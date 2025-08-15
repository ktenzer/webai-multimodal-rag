# embedding/__init__.py
import asyncio
import os, json, tempfile, warnings, logging
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame
from .element import Inputs, Outputs, Settings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_TEXT_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

embed_model = None
clip_model  = None
clip_proc   = None
_current_text_model = None
_current_clip_model = None

device = "mps" if torch.backends.mps.is_available() else "cpu"

def ensure_models(text_model_name: str, clip_model_name: str):
    global embed_model, clip_model, clip_proc, _current_text_model, _current_clip_model
    if embed_model is None or _current_text_model != text_model_name:
        print(f"Loading text embedding model: {text_model_name}")
        embed_model = SentenceTransformer(text_model_name, device=device)
        _current_text_model = text_model_name
    if clip_model is None or _current_clip_model != clip_model_name:
        print(f"Loading CLIP model: {clip_model_name}")
        clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        clip_proc  = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
        _current_clip_model = clip_model_name

class SciEmbedding:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim   = self.model.get_sentence_embedding_dimension()

    def __call__(self, texts: List[str]):
        if not texts:
            return []
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            use_multiprocessing=False
        ).tolist()

def clip_embed_image(path: str, device=None):
    with torch.no_grad():
        x = clip_proc(images=Image.open(path), return_tensors="pt")
        x = {k: v.to(device) for k, v in x.items()}
        clip_model.to(device)
        feats = clip_model.get_image_features(**x)
    return feats[0].cpu().numpy().tolist()

IDLE_TIMEOUT = 3.0  # seconds

def _stub_text(meta: dict) -> str:
    from pathlib import Path as _P
    src = meta.get("source")
    pg  = meta.get("page")
    ip  = meta.get("image_path")
    vl  = meta.get("vision_label", "")
    src_name = _P(src).name if src else (_P(ip).name if ip else "image")
    pieces = ["image"]
    if vl: pieces.append(f"({vl})")
    pieces.append("from")
    pieces.append(src_name)
    if pg is not None:
        pieces.append(f"page {pg}")
    return " ".join(pieces)

class Embedder:
    def __init__(self):
        self.text_ef = None
        self._text_model_name = None

    async def run(self, process: Process):
        outputs = process.outputs

        text_model_name = (getattr(process.settings, "text_model", None) and process.settings.text_model.value) or DEFAULT_TEXT_MODEL
        clip_model_name = (getattr(process.settings, "clip_model", None) and process.settings.clip_model.value) or DEFAULT_CLIP_MODEL

        ensure_models(text_model_name, clip_model_name)

        if self.text_ef is None or self._text_model_name != text_model_name:
            self.text_ef = SciEmbedding(text_model_name)
            self._text_model_name = text_model_name

        ingest_mode = bool(getattr(process.settings, "is_ingestion", None) and process.settings.is_ingestion.value)

        enable_clip_pixels = bool(getattr(process.settings, "enable_clip_image_embeddings", None) and process.settings.enable_clip_image_embeddings.value)
        index_blank_images = True  

        if ingest_mode:
            processed_any = False
            ait = process.inputs.wait_for_frame()

            while True:
                try:
                    if not processed_any:
                        slot, frame = await ait.__anext__()
                    else:
                        slot, frame = await asyncio.wait_for(ait.__anext__(), timeout=IDLE_TIMEOUT)
                except asyncio.TimeoutError:
                    print("Embedding (ingest): idle timeout reached; exiting.")
                    break
                except StopAsyncIteration:
                    print("Embedding (ingest): input stream closed; exiting.")
                    break

                other = getattr(frame, "other_data", None) or {}
                in_path = (other.get("file_path", "") or "").strip()
                if not in_path or not os.path.exists(in_path):
                    print(f"Embedding (ingest): path invalid: {in_path!r}; skipping.")
                    continue

                print(f"Embedding received: {in_path}")
                with open(in_path, "r", encoding="utf-8") as f:
                    bundle = json.load(f)

                txt_chunks     = bundle.get("txt_chunks", [])
                img_txt_chunks = bundle.get("img_txt_chunks", []) 
                img_docs       = bundle.get("img_docs", [])      

                # Pure text
                txt_docs  = [d["page_content"] for d in txt_chunks]
                txt_metas = [d["metadata"]     for d in txt_chunks]
                txt_embs  = self.text_ef(txt_docs)

                # Image-derived text
                img_txt_docs   = [d["page_content"] for d in img_txt_chunks]
                img_txt_metas  = [d["metadata"]     for d in img_txt_chunks]
                img_txt_embeds = self.text_ef(img_txt_docs)

                # Image "stubs", ensure images exist in <base>_images even without captions
                img_stub_docs, img_stub_metas = [], []
                if index_blank_images and img_docs:
                    for d in img_docs:
                        m = d.get("metadata", {})

                        if not m.get("image_path"):
                            continue
                        img_stub_metas.append(m)
                        img_stub_docs.append(_stub_text(m))
                img_stub_embeddings = self.text_ef(img_stub_docs)

                # Optional raw CLIP pixels
                img_embs, img_metas, img_docs_out = [], [], []
                if enable_clip_pixels and img_docs:
                    print("Embedding images (CLIP) …")
                    for d in img_docs:
                        p = d["metadata"]["image_path"]
                        img_embs.append(clip_embed_image(p, device=device))
                        img_docs_out.append("") 
                        img_metas.append(d["metadata"])

                fd, out_path = tempfile.mkstemp(prefix="embeds_", suffix=".json", dir="/tmp")
                os.close(fd)
                out_payload = {
                    # pure text
                    "txt_docs": txt_docs, "txt_metadatas": txt_metas, "txt_embeddings": txt_embs,
                    # image-derived text
                    "img_txt_docs": img_txt_docs, "img_txt_metadatas": img_txt_metas, "img_txt_embeddings": img_txt_embeds,
                    # image stub text
                    "img_stub_docs": img_stub_docs, "img_stub_metadatas": img_stub_metas, "img_stub_embeddings": img_stub_embeddings,
                    # optional legacy CLIP pixels
                    "img_docs": img_docs_out, "img_metadatas": img_metas, "img_embeddings": img_embs,
                    "prev_tmp_files": bundle.get("prev_tmp_files", []) + [in_path],
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out_payload, f, ensure_ascii=False)

                print(f"Embeddings wrote: {out_path}")
                await outputs.default.send(Frame(None, [], None, None, None, {"file_path": out_path}))
                processed_any = True

            print("Embedding (ingest): finished; exiting.")

            # Hack: Keep element running as if one element completes all will be killed by platform
            while True:
                await asyncio.sleep(1)

        # Non-ingestion (query)
        while True:
            ait = process.inputs.wait_for_frame()
            try:
                while True:
                    slot, frame = await ait.__anext__()
                    message = ""
                    other = getattr(frame, "other_data", None) or {}
                    try:
                        if isinstance(other, dict) and "api" in other:
                            user_msgs = [x for x in other.get("api", []) if isinstance(x, dict) and x.get("role") == "user"]
                            if user_msgs:
                                message = user_msgs[-1].get("content", "") or ""
                        if not message and isinstance(other, dict):
                            message = other.get("message", "") or ""
                    except Exception as e:
                        print(f"Non-ingestion: error extracting message from other_data: {e}")

                    if not message:
                        message = (getattr(frame, "text", "") or "").strip()
                    if not message:
                        continue

                    embs_vec = self.text_ef([message])[0]
                    db_path = Path(os.getenv("CHROMA_DIR", "./chroma_db/webai")).resolve()

                    req_in = other.get("requestId", None) or other.get("request_id", None)
                    try:
                        rid = int(req_in) if req_in is not None else None
                    except Exception:
                        rid = None

                    out_other = {
                        "message": message,
                        "embeddings": embs_vec,
                        "vector_index_path": str(db_path),
                    }
                    if isinstance(other.get("api"), list):
                        out_other["api"] = other["api"]
                    if rid is not None:
                        out_other["requestId"] = rid
                        out_other["request_id"] = rid

                    await outputs.default.send(Frame(None, [], None, None, None, out_other))

            except StopAsyncIteration:
                await asyncio.sleep(1)
                continue

embedder = Embedder()

process = CreateElement(Process(
    inputs=Inputs(),
    outputs=Outputs(),
    settings=Settings(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-embed000001",
        name="embedding",
        displayName="MM - Embedding",
        version="0.36.0",
        description="Embeds text + image-derived text + image stubs (optional CLIP pixels)."
    ),
    run_func=embedder.run
))


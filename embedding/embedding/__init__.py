import asyncio
import os, json, tempfile, warnings, logging
from pathlib import Path
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

# Lazy globals
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

    def __call__(self, texts):
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

class Embedder:
    def __init__(self):
        self.text_ef = None
        self._text_model_name = None

    async def run(self, process: Process):
        outputs = process.outputs

        # Read settings (with defaults)
        text_model_name = (
            getattr(process.settings, "text_model", None)
            and process.settings.text_model.value
        ) or DEFAULT_TEXT_MODEL
        clip_model_name = (
            getattr(process.settings, "clip_model", None)
            and process.settings.clip_model.value
        ) or DEFAULT_CLIP_MODEL

        # Make sure global models match the chosen settings
        ensure_models(text_model_name, clip_model_name)

        # Refresh the per-instance text encoder if needed
        if self.text_ef is None or self._text_model_name != text_model_name:
            self.text_ef = SciEmbedding(text_model_name)
            self._text_model_name = text_model_name

        ingest_mode = bool(
            getattr(process.settings, "is_ingestion", None)
            and process.settings.is_ingestion.value
        )

        # Ingestion Mode
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

                txt_chunks = bundle.get("txt_chunks", [])
                img_docs   = bundle.get("img_docs", [])

                docs  = [d["page_content"] for d in txt_chunks]
                metas = [d["metadata"]     for d in txt_chunks]

                if docs:
                    print("Embedding text …")
                    txt_embs = self.text_ef(docs)
                else:
                    print("No text chunks to embed.")
                    txt_embs = []

                # Image embeddings (optional)
                img_embs, img_metas, img_docs_out = [], [], []
                if img_docs:
                    print("Embedding images …")
                    for d in img_docs:
                        p = d["metadata"]["image_path"]
                        img_embs.append(clip_embed_image(p, device=device))
                        img_docs_out.append("") 
                        img_metas.append(d["metadata"])

                fd, out_path = tempfile.mkstemp(prefix="embeds_", suffix=".json", dir="/tmp")
                os.close(fd)
                out_payload = {
                    "txt_docs": docs, "txt_metadatas": metas, "txt_embeddings": txt_embs,
                    "img_docs": img_docs_out, "img_metadatas": img_metas, "img_embeddings": img_embs,
                    "prev_tmp_files": bundle.get("prev_tmp_files", []) + [in_path],
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out_payload, f, ensure_ascii=False)

                print(f"Embeddings wrote: {out_path}")
                await outputs.default.send(
                    Frame(None, [], None, None, None, {"file_path": out_path})
                )
                processed_any = True

            print("Embedding (ingest): finished; exiting.")
            return 


        # Non-Ingestion Mode
        while True:
            ait = process.inputs.wait_for_frame()
            try:
                while True:
                    slot, frame = await ait.__anext__()

                    # Extract message (api format or plain message or fallback to text)
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
                        continue  # nothing to embed

                    # Compute embeddings
                    embs_vec = self.text_ef([message])[0]
                    db_path = Path(os.getenv("CHROMA_DIR", "./chroma_db/webai")).resolve()

                    # Preserve upstream request id
                    req_in = other.get("requestId", None)
                    if req_in is None:
                        req_in = other.get("request_id", None)
                    rid = None
                    try:
                        rid = int(req_in) if req_in is not None else None
                    except Exception:
                        rid = None

                    # Build output payload
                    out_other = {
                        "message": message,
                        "embeddings": embs_vec,
                        "vector_index_path": str(db_path),
                    }

                    # Pass through API chat array if present
                    if isinstance(other.get("api"), list):
                        out_other["api"] = other["api"]

                    # Pass through request id in both forms
                    if rid is not None:
                        out_other["requestId"] = rid   
                        out_other["request_id"] = rid  

                    # Emit embeddings amd message to downstream (vector retrieval)
                    await outputs.default.send(
                        Frame(None, [], None, None, None, out_other)
                    )

            except StopAsyncIteration:
                await asyncio.sleep(0.25)
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
        version="0.30.0",
        description="Receives chunk file path, writes embeddings file (ingest) or emits query embeddings (non-ingest)"
    ),
    run_func=embedder.run
))

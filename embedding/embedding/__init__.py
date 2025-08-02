import os, json, tempfile, warnings, logging
import numpy as np
from pathlib import Path
import glob

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame, TextFrame

import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

from .element import Inputs, Outputs, Settings

# Quiet warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

TEXT_MODEL = "BAAI/bge-base-en-v1.5"
CLIP_MODEL = "openai/clip-vit-base-patch32"
device     = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading embedding models …")
embed_model = SentenceTransformer(TEXT_MODEL, device=device)
clip_model  = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
clip_proc   = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=True)
print("Embedding models ready\n")

class SciEmbedding:
    def __init__(self):
        self.model = SentenceTransformer(TEXT_MODEL, device=device)
        self.dim   = self.model.get_sentence_embedding_dimension()
    def __call__(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

def clip_embed_image(path: str, device=None):
    with torch.no_grad():
        x = clip_proc(images=Image.open(path), return_tensors="pt")
        x = {k: v.to(device) for k, v in x.items()}
        clip_model.to(device)
        feats = clip_model.get_image_features(**x)
    return feats[0].cpu().numpy().tolist()

import asyncio
class Embedder:
    def __init__(self):
        self.q = asyncio.Queue(4)
        self.text_ef = SciEmbedding()

    async def frame_receiver(self, _: str, frame: Frame):  
        await self.q.put(frame)

    async def run(self, process: Process):
        outputs = process.outputs

        ingest_mode = bool(getattr(process.settings, "is_ingestion", None) and process.settings.is_ingestion.value)

        while True:
            frame = await self.q.get()

            if ingest_mode:
                # INGESTION
                # Local helper so it's always in scope
                def _latest_tmp_local(pattern: str) -> str | None:
                    try:
                        candidates = sorted(
                            glob.glob(pattern),
                            key=lambda p: os.path.getmtime(p),
                            reverse=True,
                        )
                        return candidates[0] if candidates else None
                    except Exception as e:
                        print(f"_latest_tmp_local error for pattern {pattern}: {e}")
                        return None

                # Prefer TextFrame.text; fallback to newest /tmp chunks bundle
                in_path = (getattr(frame, "text", "") or "").strip()
                if not in_path:
                    in_path = _latest_tmp_local("/tmp/chunks_*.json") or ""
                    print(f"Embedding: no path in frame; fallback to latest: {in_path or 'NONE'}")

                if not in_path or not os.path.exists(in_path):
                    print("Embedding: received no usable path. Skipping.")
                    self.q.task_done()
                    continue

                print(f"Embedding received: {in_path}")

                # Load chunk bundle
                with open(in_path, "r", encoding="utf-8") as f:
                    bundle = json.load(f)

                # Expecting fields from Chunking element
                txt_chunks = bundle.get("txt_chunks", [])
                img_docs   = bundle.get("img_docs", [])

                # Prepare text docs and metas
                docs  = [d["page_content"] for d in txt_chunks]
                metas = [d["metadata"]     for d in txt_chunks]

                # Text embeddings (BGE)
                if docs:
                    print("Embedding text …")
                    txt_embs = self.text_ef(docs)
                else:
                    print("No text chunks to embed.")
                    txt_embs = []

                # Image embeddings (CLIP)
                img_embs, img_metas, img_docs_out = [], [], []
                if img_docs:
                    print("Embedding images …")
                    for d in img_docs:
                        p = d["metadata"]["image_path"]
                        img_embs.append(clip_embed_image(p, device=device))
                        img_docs_out.append("")         
                        img_metas.append(d["metadata"])

                # Write embeddings bundle and emit path
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
                await outputs.default.send(TextFrame(text=out_path))
                self.q.task_done()

            else:
                # Non-Ingestion
                message = ""
                try:
                    other = getattr(frame, "other_data", None) or {}
                    if isinstance(other, dict) and "api" in other:
                        user_msgs = [x for x in other.get("api", []) if isinstance(x, dict) and x.get("role") == "user"]
                        if user_msgs:
                            message = user_msgs[-1].get("content", "") or ""
                    if not message and isinstance(other, dict):
                        message = other.get("message", "") or ""
                except Exception as e:
                    print(f"Non-ingestion: error extracting message from other_data: {e}")

                # Fallback: allow TextFrame.text to carry the message
                if not message:
                    message = (getattr(frame, "text", "") or "").strip()

                if not message:
                    print("No message text to embed.")
                    self.q.task_done()
                    continue

                # 2) Compute embeddings using YOUR embedder (BGE) instead of `generate(...)`
                #    Result should be a plain Python list (like your sample expects).
                embs_vec = self.text_ef([message])[0]  # list[float]

                # 3) Provide a vector index path (keep same field as sample).
                db_path = Path(os.getenv("CHROMA_DIR", "./chroma_db/webai")).resolve()

                # 4) Emit a Frame with other_data (same shape as your sample)
                await outputs.default.send(
                    Frame(
                        None,           # ndframe (required positional)
                        [],             # rois
                        None,           # color_space
                        None,           # frame_id
                        None,           # headers
                        {               # other_data
                            "message": message,
                            "embeddings": embs_vec,
                            "vector_index_path": str(db_path),
                        },
                    )
                )
                self.q.task_done()

embedder = Embedder()

process = CreateElement(Process(
    inputs=Inputs(),
    outputs=Outputs(),
    settings=Settings(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-embed000001",
        name="embedding",
        displayName="MM - Embedding Element",
        version="0.16.0",
        description="Receives chunk file path, writes embeddings file, outputs that path."
    ),
    frame_receiver_func=embedder.frame_receiver,
    run_func=embedder.run
))


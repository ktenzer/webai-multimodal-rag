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

from .element import Inputs, Outputs

# Silence transformer warnings like in the monolith
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

# ---- exact helpers preserved ----
class SciEmbedding:
    def __init__(self):
        self.model = SentenceTransformer(TEXT_MODEL, device=device)
        self.dim   = self.model.get_sentence_embedding_dimension()
    def __call__(self, texts):
        # returns list[list[float]]
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
        while True:
            frame = await self.q.get()

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
                    img_docs_out.append("")         # mirrors original: documents=[""] for images
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

embedder = Embedder()

process = CreateElement(Process(
    inputs=Inputs(),
    outputs=Outputs(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-embed000001",
        name="embedding",
        displayName="MM - Embedding Element",
        version="0.14.0",
        description="Receives chunk file path, writes embeddings file, outputs that path."
    ),
    frame_receiver_func=embedder.frame_receiver,
    run_func=embedder.run
))


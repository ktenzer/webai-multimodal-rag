import asyncio
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


IDLE_TIMEOUT = 3.0  # seconds

class Embedder:
    def __init__(self):
        self.text_ef = SciEmbedding()

    async def run(self, process: Process):
        outputs = process.outputs
        ingest_mode = bool(getattr(process.settings, "is_ingestion", None) and process.settings.is_ingestion.value)

        # ---------- INGESTION MODE: auto-exit after idle ----------
        if ingest_mode:
            IDLE_TIMEOUT = 3.0  # seconds
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

                # --- expect path from Chunking via Frame.other_data["file_path"] ---
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

                img_embs, img_metas, img_docs_out = [], [], []
                if img_docs:
                    print("Embedding images …")
                    for d in img_docs:
                        p = d["metadata"]["image_path"]
                        img_embs.append(clip_embed_image(p, device=device))
                        img_docs_out.append("")   # mirrors original
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
            return  # <- important: stop here in ingestion mode

        # ---------- NON-INGESTION MODE: stay alive indefinitely ----------
        while True:
            # attach to the input stream and consume until it closes; then reattach
            ait = process.inputs.wait_for_frame()
            try:
                while True:
                    slot, frame = await ait.__anext__()

                    # extract message (api format or plain message or fallback to text)
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

                    if not message:
                        message = (getattr(frame, "text", "") or "").strip()

                    if not message:
                        # quietly ignore empty/control frames
                        continue

                    # compute embeddings for the user message
                    embs_vec = self.text_ef([message])[0]
                    db_path = Path(os.getenv("CHROMA_DIR", "./chroma_db/webai")).resolve()

                    # emit embeddings + message to downstream (vector retrieval)
                    await outputs.default.send(
                        Frame(
                            None, [], None, None, None,
                            {
                                "message": message,
                                "embeddings": embs_vec,
                                "vector_index_path": str(db_path),
                            },
                        )
                    )

            except StopAsyncIteration:
                # upstream temporarily closed; stay resident and reattach
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
        displayName="MM - Embedding Element",
        version="0.23.0",
        description="Receives chunk file path, writes embeddings file, outputs that path."
    ),
    run_func=embedder.run
))


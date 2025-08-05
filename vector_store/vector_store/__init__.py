import asyncio
import os, json, time
from pathlib import Path
import glob
from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame

import chromadb
from chromadb.config import Settings as ChromaSettings  # alias to avoid clashes
from .element import Inputs, Settings as VSSettings

IDLE_TIMEOUT = 3.0  # seconds

def _add_in_batches(
    col,
    docs: list,
    metas: list,
    embs: list,
    *,
    id_prefix: str = "t",
    max_batch: int | None = None,):

    if not docs:
        return

    n = len(docs)
    if not (len(metas) == n and len(embs) == n):
        raise ValueError(f"Lengths mismatch: docs={len(docs)} metas={len(metas)} embs={len(embs)}")

    # fallback if user hasn't provided setting
    if max_batch is None:
        max_batch = 2000

    batches = (n + max_batch - 1) // max_batch
    print(f"[VectorStore] adding {n} items to '{col.name}' in {batches} batch(es) (max_batch={max_batch})")

    for i in range(0, n, max_batch):
        j = min(i + max_batch, n)
        ids = [f"{id_prefix}{k}" for k in range(i, j)]
        col.add(
            documents=docs[i:j],
            metadatas=metas[i:j],
            embeddings=embs[i:j],
            ids=ids,
        )


def build_stores_from_payload(payload, client, max_batch: int):
    import time
    print("Embedding & writing to Chroma …")
    t0 = time.time()

    # ---- TEXT ----
    txt_col = client.get_or_create_collection("text")
    docs  = payload.get("txt_docs", []) or []
    metas = payload.get("txt_metadatas", []) or []
    embs  = payload.get("txt_embeddings", []) or []
    _add_in_batches(txt_col, docs, metas, embs, id_prefix="t", max_batch=max_batch)

    # ---- IMAGES ----
    img_embs  = payload.get("img_embeddings", []) or []
    if img_embs:
        img_col  = client.get_or_create_collection("images")
        img_docs = payload.get("img_docs", []) or []
        img_meta = payload.get("img_metadatas", []) or []
        _add_in_batches(img_col, img_docs, img_meta, img_embs, id_prefix="i", max_batch=max_batch)

    print(f"Vector DB ready ({time.time()-t0:.1f}s)\n")

class StoreWriter:
    async def run(self, process):
        user_dir = (process.settings.vector_db_folder_path.value or "").strip()
        chroma_dir = Path(user_dir).resolve()
        os.makedirs(chroma_dir, exist_ok=True)
        print(f"[VectorStore] Chroma DB directory: {chroma_dir}")

        processed_any = False
        ait = process.inputs.wait_for_frame()  # consume directly

        while True:
            try:
                if not processed_any:
                    slot, frame = await ait.__anext__()
                else:
                    slot, frame = await asyncio.wait_for(ait.__anext__(), timeout=IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                print("VectorStore: idle timeout reached; exiting.")
                break
            except StopAsyncIteration:
                print("VectorStore: input stream closed; exiting.")
                break

            other = getattr(frame, "other_data", None) or {}
            path = (other.get("file_path", "") or "").strip()

            if not path or not os.path.exists(path):
                print(f"VectorStore: invalid path received: {path!r}. Skipping.")
                continue

            print(f"VectorStore received: {path}")
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            max_batch = int(process.settings.max_batch_size.value)
            build_stores_from_payload(payload, client, max_batch)

            # cleanup
            for p in payload.get("prev_tmp_files", []) + [path]:
                try:
                    os.remove(p)
                    print(f"Deleted tmp: {p}")
                except Exception as e:
                    print(f"Cleanup failed for {p}: {e}")

            processed_any = True

        print("VectorStore: finished; exiting.")

writer = StoreWriter()

process = CreateElement(Process(
    inputs=Inputs(),
    settings=VSSettings(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-store000003",
        name="vector_store",
        displayName="MM - Vector Store Element",
        version="0.17.0",
        description="Consumes embeddings bundle filename and writes to Chroma; cleans up tmp files."
    ),
    run_func=writer.run
))


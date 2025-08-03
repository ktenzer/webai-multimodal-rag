import os, json, time
from pathlib import Path
import glob
from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame

import chromadb
from chromadb.config import Settings as ChromaSettings  # alias to avoid clashes
from .element import Inputs, Settings as VSSettings

def build_stores_from_payload(payload, client):
    print("Embedding & writing to Chroma …")
    t0 = time.time()

    txt_col = client.get_or_create_collection("text")
    docs = payload.get("txt_docs", [])
    metas= payload.get("txt_metadatas", [])
    embs = payload.get("txt_embeddings", [])
    if docs:
        txt_col.add(
            documents=docs,
            metadatas=metas,
            embeddings=embs,
            ids=[f"t{idx}" for idx in range(len(docs))]
        )

    img_docs = payload.get("img_docs", [])
    img_metas= payload.get("img_metadatas", [])
    img_embs = payload.get("img_embeddings", [])
    if img_embs:
        img_col = client.get_or_create_collection("images")
        img_col.add(
            documents=img_docs,
            metadatas=img_metas,
            embeddings=img_embs,
            ids=[f"i{idx}" for idx in range(len(img_embs))]
        )

    print(f"Vector DB ready ({time.time()-t0:.1f}s)\n")

import asyncio
class StoreWriter:
    def __init__(self):
        self.q = asyncio.Queue(4)

    async def frame_receiver(self, _: str, frame: Frame):
        await self.q.put(frame)

    async def run(self, process):
        user_dir = (process.settings.vector_db_folder_path.value or "").strip()
        chroma_dir = Path(user_dir).resolve()
        os.makedirs(chroma_dir, exist_ok=True)
        print(f"[VectorStore] Chroma DB directory: {chroma_dir}")

        while True:
            frame = await self.q.get()

            # >>> READ VIA Frame.other_data ONLY <<<
            other = getattr(frame, "other_data", None) or {}
            path = (other.get("file_path", "") or "").strip()

            if not path or not os.path.exists(path):
                print(f"VectorStore: invalid path received: {path!r}. Skipping.")
                self.q.task_done()
                continue

            print(f"VectorStore received: {path}")
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            build_stores_from_payload(payload, client)

            # Cleanup tmp files from earlier stages + this file
            for p in payload.get("prev_tmp_files", []) + [path]:
                try:
                    os.remove(p)
                    print(f"Deleted tmp: {p}")
                except Exception as e:
                    print(f"Cleanup failed for {p}: {e}")

            self.q.task_done()

writer = StoreWriter()

process = CreateElement(Process(
    inputs=Inputs(),
    settings=VSSettings(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-store000003",
        name="vector_store",
        displayName="MM - Vector Store Element",
        version="0.11.0",
        description="Consumes embeddings bundle filename and writes to Chroma; cleans up tmp files."
    ),
    frame_receiver_func=writer.frame_receiver,
    run_func=writer.run
))


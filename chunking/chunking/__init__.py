# chunking/__init__.py
import os, json, time, tempfile
from pathlib import Path
from typing import List

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document

from .element import Inputs, Outputs

def md_split(docs, chunk=800, overlap=80):
    header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])
    rc     = RecursiveCharacterTextSplitter(chunk_size=chunk, chunk_overlap=overlap)
    out: List[Document] = []
    print("Chunking text …")
    t0 = time.time()
    for doc in docs:
        for sec in header.split_text(doc.page_content):
            for ch in rc.split_text(sec.page_content):
                out.append(Document(page_content=ch, metadata=doc.metadata))
    print(f"{len(out)} chunks ({time.time()-t0:.1f}s)\n")
    return out

import asyncio
class Chunker:
    def __init__(self):
        self.q = asyncio.Queue(8)

    async def frame_receiver(self, _: str, frame: Frame):
        # Push raw frames into our own queue (avoid wait_for_frame() to prevent "Already borrowed")
        await self.q.put(frame)

    async def run(self, process: Process):
        outputs = process.outputs
        processed = set()

        while True:
            frame = await self.q.get()
            try:
                other = getattr(frame, "other_data", None) or {}
                in_path = (other.get("file_path", "") or "").strip()

                if not in_path or not Path(in_path).exists():
                    print(f"Chunking: invalid path received: {in_path!r}; skipping.")
                    continue
                if in_path in processed:
                    print(f"Chunking: duplicate path: {in_path}; skipping.")
                    continue

                print(f"Chunking processing: {in_path}")
                with open(in_path, "r", encoding="utf-8") as f:
                    bundle = json.load(f)

                text_docs = [
                    Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in bundle.get("text_docs", [])
                ]
                img_docs = bundle.get("img_docs", [])

                chunks = md_split(text_docs)
                chunks_dicts = [{"page_content": d.page_content, "metadata": d.metadata} for d in chunks]

                fd, out_path = tempfile.mkstemp(prefix="chunks_", suffix=".json", dir="/tmp")
                os.close(fd)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "txt_chunks": chunks_dicts,
                            "img_docs": img_docs,
                            "prev_tmp_files": [in_path],
                        },
                        f,
                        ensure_ascii=False,
                    )
                print(f"Chunking wrote: {out_path}")

                # Send downstream using Frame.other_data
                await outputs.default.send(
                    Frame(None, [], None, None, None, {"file_path": out_path})
                )
                processed.add(in_path)

                # We expect one OCR bundle; exit after successful processing
                break
            finally:
                self.q.task_done()

        print("Chunking: finished; exiting.")

chunker = Chunker()

process = CreateElement(Process(
    inputs=Inputs(),
    outputs=Outputs(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-chunk0000001",
        name="chunking",
        displayName="MM - Chunking Element",
        version="0.22.0",  # bump to redeploy
        description="Splits OCR text docs into chunks; outputs path to chunk bundle."
    ),
    frame_receiver_func=chunker.frame_receiver,  # <— important to avoid "Already borrowed"
    run_func=chunker.run
))

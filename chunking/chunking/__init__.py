import os, json, time, tempfile
import glob
from typing import List
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from typing import List
from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame, TextFrame  

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
        self.frame_q = asyncio.Queue(4)

    async def frame_receiver(self, _: str, frame: Frame):  
        print("DEBUG frame:", type(frame).__name__, getattr(frame, "text", None), getattr(frame, "meta", None))
        await self.frame_q.put(frame)

    async def run(self, process: Process):
        outputs = process.outputs
        processed = set()

        def _latest(pattern: str):
            xs = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p))
            return xs[-1] if xs else None

        async for (_slot, frame) in process.inputs.wait_for_frame():
            # prefer TextFrame.text; fallback to newest OCR bundle
            in_path = (getattr(frame, "text", "") or "").strip()
            if not in_path:
                in_path = _latest("/tmp/ocr_bundle_*.json") or ""
                print(f"Chunking: no path in frame; fallback to {in_path or 'NONE'}")

            if not in_path or not Path(in_path).exists() or in_path in processed:
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
                    {"txt_chunks": chunks_dicts, "img_docs": img_docs, "prev_tmp_files": [in_path]},
                    f,
                    ensure_ascii=False,
                )
            print(f"Chunking wrote: {out_path}")
            await outputs.default.send(TextFrame(text=out_path))
            processed.add(in_path)

        print("Chunking: input stream closed; exiting.")
        return


chunker = Chunker()

process = CreateElement(Process(
    inputs=Inputs(),
    outputs=Outputs(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-chunk0000001",
        name="chunking",
        displayName="MM - Chunking Element",
        version="0.20.0",
        description="Splits OCR text docs into chunks; outputs path to chunk bundle."
    ),
    run_func=chunker.run
))

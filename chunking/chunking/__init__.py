import os
import asyncio
import json
import tempfile
import time

from pathlib import Path
from typing import List

import nltk
from nltk.tokenize import TextTilingTokenizer, sent_tokenize
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame

from .element import Inputs, Outputs, Settings

IDLE_TIMEOUT = 3.0  # seconds

# Protect table data merging contiguous lines
def protect_tables(text: str) -> str:
    lines = text.splitlines()
    out, buf = [], []
    for line in lines:
        if line.strip().startswith("```") or line.strip().startswith("|"):
            buf.append(line)
        else:
            if buf:
                out.append(" ".join(buf))
                buf = []
            out.append(line)
    if buf:
        out.append(" ".join(buf))
    return "\n".join(out)

# Tokenizer for token based chunk splits
TOKENIZER = None
def token_count(text: str, model_name: str):
    global TOKENIZER
    if TOKENIZER is None or TOKENIZER.name_or_path != model_name:
        TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return len(TOKENIZER.tokenize(text))

# OG simple recursive character splitter
def md_split(
        docs: List[Document],
        chunk_size: int,
        overlap: int,
) -> List[Document]:
    header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])
    rc     = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    out: List[Document] = []
    print("Chunking text …")
    t0 = time.time()
    for doc in docs:
        for sec in header.split_text(doc.page_content):
            for ch in rc.split_text(sec.page_content):
                out.append(Document(page_content=ch, metadata=doc.metadata))
    print(f"{len(out)} chunks ({time.time()-t0:.1f}s)\n")
    return out


# Recursive Split via heading
def recursive_split_with_headings(
    docs: List[Document],
    chunk_size: int,
    overlap: int,
    token_model: str,
    max_tokens: int
) -> List[Document]:
    header = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")]
    )
    rc = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    out: List[Document] = []
    t0 = time.time()
    for doc in docs:
        text = protect_tables(doc.page_content)
        for sec in header.split_text(text):
            path = " > ".join(filter(None, (
                sec.metadata.get("h1"),
                sec.metadata.get("h2"),
                sec.metadata.get("h3"),
            )))
            for ch in rc.split_text(sec.page_content):
                chunk = (path + "\n\n" + ch).strip()
                # enforce token budget
                while token_count(chunk, token_model) > max_tokens:
                    chunk = chunk.rsplit(" ", 1)[0]
                out.append(Document(page_content=chunk, metadata=doc.metadata))
    print(f"[Chunk] recursive_split → {len(out)} chunks in {time.time()-t0:.1f}s")
    return out

# Semantic split based on sliding window
def semantic_split(
    docs: list[Document],
    model_name: str,
    window: int,
    overlap: int,
    token_model: str,
    max_tokens: int,
) -> list[Document]:
    embedder = SentenceTransformer(model_name)
    out = []

    for doc in docs:
        sents = sent_tokenize(doc.page_content)
        if not sents:
            continue

        # precompute embeddings
        embs = embedder.encode(sents, convert_to_tensor=True)

        i = 0
        while i < len(sents):
            j = min(i + window, len(sents))
            chunk_emb = embs[i:j].mean(dim=0, keepdim=True)
            # check if on average it’s still coherent
            avg_sim = float(cos_sim(chunk_emb, embs[i:j]).mean())
            text = " ".join(sents[i:j]).strip()

            # enforce token budget
            while token_count(text, token_model) > max_tokens and " " in text:
                text = text.rsplit(" ", 1)[0]

            if text:
                md = doc.metadata.copy()
                md.setdefault("split_method", "semantic_simple")
                md["avg_sim"] = round(avg_sim, 2)
                out.append(Document(page_content=text, metadata=md))

            # advance
            i += window - overlap

    return out

# Chunker
class Chunker:
    async def run(self, process: Process):
        settings = process.settings
        outputs = process.outputs
        processed = set()
        processed_any = False
        ait = process.inputs.wait_for_frame()

        while True:
            try:
                if not processed_any:
                    _, frame = await ait.__anext__()
                else:
                    _, frame = await asyncio.wait_for(ait.__anext__(), timeout=IDLE_TIMEOUT)
            except asyncio.TimeoutError:
                print("Chunking: idle timeout; exiting.")
                break
            except StopAsyncIteration:
                print("Chunking: input closed; exiting.")
                break

            other = getattr(frame, "other_data", {}) or {}
            path = (other.get("file_path","") or "").strip()
            if not path or path in processed or not Path(path).exists():
                continue

            # Load OCR bundle
            bundle = json.loads(Path(path).read_text(encoding="utf-8"))
            docs = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in bundle.get("text_docs", [])
            ]

            # Fetch common settings
            strategy      = settings.chunk_strategy.value
            token_model   = settings.llm_token_model.value  
            max_tokens    = int(settings.max_chunk_tokens.value)
            out_chunks = []

            if strategy == "semantic":
                out_chunks = semantic_split(
                    docs,
                    settings.semantic_model.value,
                    settings.semantic_window_size.value,
                    settings.semantic_overlap.value,
                    token_model,
                    max_tokens
                )
            elif strategy == "recursive":
                out_chunks = recursive_split_with_headings(
                    docs,
                    settings.chunk_size.value,
                    settings.chunk_overlap.value,
                    token_model,
                    max_tokens
                )
            else:
                out_chunks = md_split(
                    docs,
                    settings.chunk_size.value,
                    settings.chunk_overlap.value,
                )

            # Serialize
            chunks_dicts = [
                {"page_content": c.page_content, "metadata": c.metadata}
                for c in out_chunks
            ]
            out_payload = {
                "txt_chunks": chunks_dicts,
                "img_docs":   bundle.get("img_docs", []),
                "prev_tmp_files": [path]
            }

            fd, out_path = tempfile.mkstemp(prefix="chunks_", suffix=".json", dir="/tmp")
            os.close(fd)
            with open(out_path, "w", encoding="utf-8") as wf:
                json.dump(out_payload, wf, ensure_ascii=False)

            print(f"Chunking wrote: {out_path}")
            await outputs.default.send(Frame(None, [], None, None, None, {"file_path": out_path}))
            processed.add(path)
            processed_any = True

        print("Chunking: finished; exiting.")

process = CreateElement(Process(
    inputs=Inputs(),
    outputs=Outputs(),
    settings=Settings(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-chunk0000001",
        name="chunking",
        displayName="MM - Chunking",
        version="0.66.0",
        description="Splits OCR text docs into highly coherent chunks for RAG"
    ),
    run_func=Chunker().run
))



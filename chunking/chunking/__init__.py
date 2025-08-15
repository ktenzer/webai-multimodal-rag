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
from transformers import logging as hf_logging

from typing import Iterable
from transformers import PreTrainedTokenizerBase

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame

from .element import Inputs, Outputs, Settings

TOKENIZERS_PARALLELISM=False
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

def enforce_max_for_embedder(embedder, hard_max: int):
    # Align Sentence-Transformers and HF tokenizer limits
    if hasattr(embedder, "max_seq_length"):
        embedder.max_seq_length = hard_max
    if hasattr(embedder, "tokenizer"):
        try:
            embedder.tokenizer.model_max_length = hard_max  # avoids HF warnings
        except Exception:
            pass

def split_text_by_token_ids(text: str, tok, safe_max: int) -> list[str]:
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return []
    pieces = []
    for i in range(0, len(ids), safe_max):
        sub = ids[i:i+safe_max]
        pieces.append(tok.decode(sub, skip_special_tokens=True))
    return pieces

def token_id_chunks(text: str, tok, max_len: int) -> List[str]:
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        return []
    out = []
    for i in range(0, len(ids), max_len):
        sub = ids[i:i+max_len]
        out.append(tok.decode(sub, skip_special_tokens=True))
    return out

def get_embedder_tokenizer_and_limit(model_name: str, fallback: int = 512):
    tok = AutoTokenizer.from_pretrained(model_name)
    # Some tokenizers use a giant sentinel like 1e30 for "no limit"
    raw = getattr(tok, "model_max_length", None)
    if raw is None or raw > 10**8:  
        raw = fallback
    # leave headroom for specials
    safe_max = max(8, int(raw) - 2)
    # clamp tokenizer
    try:
        tok.model_max_length = int(raw)
    except Exception:
        pass
    return tok, safe_max


# Recursive Split via heading
def recursive_split_with_headings(
    docs: List[Document],
    chunk_size: int,
    overlap: int,
    token_model: str,
    max_tokens: int,
    embedder_model: str,
) -> List[Document]:
    header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])
    rc     = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # One embedder tokenizer reused for all chunks, clamped to 512
    embed_tok, safe_max = get_embedder_tokenizer_and_limit(embedder_model)

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

                # LLM-side cap first
                while token_count(chunk, token_model) > max_tokens and " " in chunk:
                    chunk = chunk.rsplit(" ", 1)[0]
                while token_count(chunk, token_model) > max_tokens and len(chunk) > 0:
                    chunk = chunk[:-50]  # fallback for no-space runs

                if not chunk:
                    continue

                # Embedder-side cap 
                parts = token_id_chunks(chunk, embed_tok, safe_max)
                if not parts:
                    continue
                for p in parts:
                    out.append(Document(page_content=p, metadata=doc.metadata))

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
    # Embedder + tokenizer, both clamped to 512
    embedder = SentenceTransformer(model_name)
    try:
        embedder.max_seq_length = 512
    except Exception:
        pass
    embed_tok, safe_max = get_embedder_tokenizer_and_limit(model_name)

    out: list[Document] = []

    for doc in docs:
        # Merge table lines first, then sentence-split
        base = protect_tables(doc.page_content)
        sents = sent_tokenize(base) or [base]

        # Token-ID split: guarantee each piece under embedder cap
        safe_sents: list[str] = []
        for s in sents:
            safe_sents.extend(token_id_chunks(s, embed_tok, safe_max))
        if not safe_sents:
            continue

        embs = embedder.encode(safe_sents, convert_to_tensor=True, show_progress_bar=False)

        i, step = 0, max(1, window - overlap)
        while i < len(safe_sents):
            j = min(i + window, len(safe_sents))
            text = " ".join(safe_sents[i:j]).strip()

            # LLM-side budget (may be a different tokenizer)
            while token_count(text, token_model) > max_tokens and " " in text:
                text = text.rsplit(" ", 1)[0]
            while token_count(text, token_model) > max_tokens and len(text) > 0:
                text = text[:-50]

            if text:
                chunk_emb = embs[i:j].mean(dim=0, keepdim=True)
                avg_sim = float(cos_sim(chunk_emb, embs[i:j]).mean())
                md = doc.metadata.copy()
                md.setdefault("split_method", "semantic_simple")
                md["avg_sim"] = round(avg_sim, 2)
                out.append(Document(page_content=text, metadata=md))

            i += step

    return out

# Chunker
class Chunker:
    async def run(self, process: Process):
        settings = process.settings
        outputs = process.outputs
        processed = set()
        processed_any = False
        ait = process.inputs.wait_for_frame()

        out_path = ""
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

            # Build pure text and image-derived text
            docs_text = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in bundle.get("text_docs", [])
            ]
            docs_image_text = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in bundle.get("image_text_docs", [])  
            ]

            # Fetch common settings
            strategy      = settings.chunk_strategy.value
            token_model   = settings.llm_token_model.value  
            max_tokens    = int(settings.max_chunk_tokens.value)

            def _split(docs: List[Document]) -> List[Document]:
                if strategy == "semantic":
                    return semantic_split(
                        docs,
                        settings.semantic_model.value,
                        settings.semantic_window_size.value,
                        settings.semantic_overlap.value,
                        token_model,
                        max_tokens,
                    )
                elif strategy == "recursive":
                    return recursive_split_with_headings(
                        docs,
                        settings.chunk_size.value,
                        settings.chunk_overlap.value,
                        token_model,
                        max_tokens,
                        embedder_model=settings.semantic_model.value,
                    )
                else:
                    return md_split(
                        docs,
                        settings.chunk_size.value,
                        settings.chunk_overlap.value,
                    )

            # Run chosen splitter on both corpora
            out_chunks_text = _split(docs_text)
            out_chunks_imgt = _split(docs_image_text)

            # Serialize
            chunks_dicts = [
                {"page_content": c.page_content, "metadata": c.metadata}
                for c in out_chunks_text
            ]
            img_chunks_dicts = [
                {"page_content": c.page_content, "metadata": c.metadata}
                for c in out_chunks_imgt
            ]

            out_payload = {
                "txt_chunks": chunks_dicts,
                "img_txt_chunks": img_chunks_dicts, 
                "img_docs":   bundle.get("img_docs", []),
                "prev_tmp_files": [path]
            }

            fd, out_path = tempfile.mkstemp(prefix="chunks_", suffix=".json", dir="/tmp")
            os.close(fd)
            with open(out_path, "w", encoding="utf-8") as wf:
                json.dump(out_payload, wf, ensure_ascii=False)

            print(f"Chunking wrote: {out_path} (text={len(chunks_dicts)}, image_text={len(img_chunks_dicts)})")
            await outputs.default.send(Frame(None, [], None, None, None, {"file_path": out_path}))
            processed.add(path)
            processed_any = True

        print("Chunking: finished; exiting.")

        # Hack: Keep element running as if one element completes all will be killed by platform
        while True:
            await asyncio.sleep(1)

process = CreateElement(Process(
    inputs=Inputs(),
    outputs=Outputs(),
    settings=Settings(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-chunk0000001",
        name="chunking",
        displayName="MM - Chunking",
        version="0.77.0", 
        description="Splits OCR text docs into highly coherent chunks for RAG (and image-derived text separately)."
    ),
    run_func=Chunker().run
))



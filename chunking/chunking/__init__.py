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

os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

def token_id_sliding_chunks_with_prefix(prefix: str, body: str, tok, max_len: int, overlap_tokens: int) -> list[str]:
    prefix_ids = tok.encode(prefix, add_special_tokens=False) if prefix else []
    spacer_ids = tok.encode("\n\n", add_special_tokens=False) if prefix else []
    body_ids   = tok.encode(body, add_special_tokens=False)

    if not body_ids:
        return []

    # reserve budget for prefix + spacer
    reserved = len(prefix_ids) + len(spacer_ids)
    if reserved >= max_len:
        # clamp prefix if it alone would overflow
        keep = max(0, max_len - len(spacer_ids))
        prefix_ids = prefix_ids[:keep]
        reserved = len(prefix_ids) + len(spacer_ids)

    payload_cap = max(1, max_len - reserved)
    # step size to create overlap
    step = max(1, payload_cap - max(0, overlap_tokens))

    out = []
    i = 0
    while i < len(body_ids):
        win = body_ids[i : i + payload_cap]
        full = (prefix_ids + spacer_ids + win) if prefix_ids else win
        out.append(tok.decode(full, skip_special_tokens=True))
        if i + payload_cap >= len(body_ids):
            break
        i += step
    return out

def count_embed_tokens(text: str, tok) -> int:
    return len(tok.encode(text, add_special_tokens=False))

def pack_sentences_with_prefix(
    prefix_tag: str,                 # short tag or "" (we’ll keep it short)
    body_text: str,
    tok,                             # embedder tokenizer
    max_len: int,                    # embedder cap (safe_max)
    overlap_sents: int = 2           # 1–2 is plenty
) -> list[str]:
    """
    Greedy sentence packer that keeps whole sentences, respects token budget,
    and (optionally) prepends a SHORT prefix tag to each chunk.
    """
    sents = sent_tokenize(body_text) or [body_text]
    chunks = []
    i = 0

    # reserve a small prefix once
    prefix = (prefix_tag.strip() + "\n") if prefix_tag else ""
    prefix_tokens = count_embed_tokens(prefix, tok) if prefix else 0
    payload_cap = max(1, max_len - prefix_tokens)

    while i < len(sents):
        cur = []
        cur_tokens = 0
        j = i
        while j < len(sents):
            cand = sents[j]
            cand_tokens = count_embed_tokens(cand, tok)
            # if single sentence longer than cap, hard-slice it (rare)
            if cand_tokens > payload_cap:
                ids = tok.encode(cand, add_special_tokens=False)[:payload_cap]
                cand = tok.decode(ids, skip_special_tokens=True)
                cand_tokens = count_embed_tokens(cand, tok)

            if cur_tokens + cand_tokens + (1 if cur else 0) > payload_cap:
                break
            cur.append(cand)
            cur_tokens += cand_tokens + (1 if cur[:-1] else 0)  # account for space/newline
            j += 1

        if not cur:
            # fallback: force include at least something
            force = sents[j] if j < len(sents) else sents[-1]
            ids = tok.encode(force, add_special_tokens=False)[:payload_cap]
            cur = [tok.decode(ids, skip_special_tokens=True)]
            j = max(j, i + 1)

        chunk_body = " ".join(cur).strip()
        chunk_text = (prefix + chunk_body) if prefix else chunk_body
        chunks.append(chunk_text)

        # sentence-overlap step
        i = j - overlap_sents if j - overlap_sents > i else j

    return chunks

def get_embedder_tokenizer_and_limit(model_name: str, fallback: int = 512):
    # We will *enforce* fallback as the real cap (safe_max) ourselves,
    # but set a HUGE model_max_length on the tokenizer to prevent HF warnings
    # when we momentarily tokenize long strings to count/split.
    tok = AutoTokenizer.from_pretrained(model_name)

    # Silence "sequence length longer than max length" warnings during counting
    try:
        tok.model_max_length = int(1e9)  # huge -> no warning while we pre-split
    except Exception:
        pass

    safe_max = int(fallback)            # the true budget we enforce downstream
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

    embed_tok, safe_max = get_embedder_tokenizer_and_limit(embedder_model)

    out: List[Document] = []
    t0 = time.time()

    # keep the tag short so it doesn’t dominate embeddings
    def make_tag(sec_md: dict) -> str:
        h1, h2, h3 = (sec_md.get("h1","") or ""), (sec_md.get("h2","") or ""), (sec_md.get("h3","") or "")
        parts = [p.strip() for p in (h1, h2, h3) if p]
        # short tag: just top 1–2 levels
        tag = "§" + "/".join(parts[:2])
        return tag[:50]  # hard limit

    for doc in docs:
        text = protect_tables(doc.page_content)
        for sec in header.split_text(text):
            tag = make_tag(sec.metadata)  # short, stable anchor (or "" to disable)
            for ch in rc.split_text(sec.page_content):
                body = ch.strip()
                if not body:
                    continue

                # Sentence-pack within the embedder budget
                pieces = pack_sentences_with_prefix(
                    prefix_tag=tag,
                    body_text=body,
                    tok=embed_tok,
                    max_len=safe_max,
                    overlap_sents=2,     # 1–2 sentences overlap
                )

                # Final LLM-side budget (gentle tail trim only if you must)
                if max_tokens and token_model:
                    for p in pieces:
                        while token_count(p, token_model) > max_tokens and len(p) > 0:
                            p = p[:-50]
                        if p:
                            out.append(Document(page_content=p, metadata=doc.metadata))
                else:
                    for p in pieces:
                        out.append(Document(page_content=p, metadata=doc.metadata))

    print(f"[Chunk] recursive_split → {len(out)} chunks in {time.time()-t0:.1f}s")
    return out


# Semantic split based on sliding window
def semantic_split(
    docs: list[Document],
    model_name: str,
    window: int,          # unused now; kept for signature compatibility
    overlap: int,         # unused now; we use sentence overlap below
    token_model: str,
    max_tokens: int,
) -> list[Document]:
    """
    Token-safe, sentence-aware splitter that PREPARES chunks for a given embedder
    without calling the embedder. This avoids the 2690>512 error entirely.
    """

    # Use the SAME HF tokenizer your embedder will use to enforce the cap now.
    embed_tok, safe_max = get_embedder_tokenizer_and_limit(model_name)

    out: list[Document] = []
    header = MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2"),("###","h3")])

    for doc in docs:
        sections = header.split_text(protect_tables(doc.page_content))

        for sec in sections:
            # short, stable tag to give the chunk some local context without polluting too much
            h1, h2 = (sec.metadata.get("h1") or ""), (sec.metadata.get("h2") or "")
            tag = ("§" + "/".join([p for p in (h1, h2) if p]))[:50]

            # Pack whole sentences under the embedder's token budget, with small sentence overlap
            pieces = pack_sentences_with_prefix(
                prefix_tag=tag,
                body_text=sec.page_content,
                tok=embed_tok,
                max_len=safe_max,
                overlap_sents=2,   # 1–2 sentence overlap is usually enough
            )
            if not pieces:
                continue

            # Hard-guard: if any piece still slipped over (safety), split by token-ids
            safe_pieces: list[str] = []
            for p in pieces:
                ids = embed_tok.encode(p, add_special_tokens=False)
                if len(ids) > safe_max:
                    for i in range(0, len(ids), safe_max):
                        sub = embed_tok.decode(ids[i:i+safe_max], skip_special_tokens=True)
                        if sub.strip():
                            safe_pieces.append(sub)
                else:
                    safe_pieces.append(p)

            # Final LLM-side cap (may be a different tokenizer)
            for text in safe_pieces:
                if max_tokens and token_model:
                    while token_count(text, token_model) > max_tokens and len(text) > 0:
                        text = text[:-50]
                if not text:
                    continue
                md = doc.metadata.copy()
                md.setdefault("split_method", "semantic_simple")
                out.append(Document(page_content=text, metadata=md))

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
        version="0.85.0", 
        description="Splits OCR text docs into highly coherent chunks for RAG (and image-derived text separately)."
    ),
    run_func=Chunker().run
))



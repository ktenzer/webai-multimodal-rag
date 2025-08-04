import os, textwrap, warnings, logging, json
from pathlib import Path
from typing import List

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import TextFrame, Frame

import torch, chromadb
from chromadb.config import Settings as ChromaSettings
import asyncio
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from rank_bm25 import BM25Okapi
import numpy as np

from .element import Inputs, Outputs, Settings as VSSettings

# Quiet warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Constants
TEXT_MODEL   = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading retrieval models …")
embed_model = SentenceTransformer(TEXT_MODEL, device=device)  # not used directly here
reranker    = CrossEncoder(RERANK_MODEL, device=device)       # used for rerank
print("Models ready\n")

def mmr_select(query_emb, doc_embs, docs, metas, k=20, weight=0.6):
    query_emb = np.asarray(query_emb, dtype=np.float32)
    doc_embs  = np.asarray(doc_embs,  dtype=np.float32)
    selected, sel_docs, sel_metas = [], [], []
    candidates = list(range(len(docs)))
    sims_query = cos_sim(torch.tensor(query_emb), torch.tensor(doc_embs)).numpy().flatten()
    while len(selected) < k and candidates:
        if not selected:
            idx = int(np.argmax(sims_query[candidates]))
        else:
            sims_selected = cos_sim(
                torch.tensor(doc_embs[candidates]),
                torch.tensor(doc_embs[selected])
            ).numpy().max(axis=1)
            mmr = weight * sims_query[candidates] - (1 - weight) * sims_selected
            idx = candidates[int(np.argmax(mmr))]
        selected.append(idx)
        sel_docs.append(docs[idx])
        sel_metas.append(metas[idx])
        candidates.remove(idx)
    return sel_docs, sel_metas

class BM25Store:
    def __init__(self, docs_lower: List[str]):
        self.model = BM25Okapi([d.split() for d in docs_lower])
    def query(self, q: str, k: int):
        idxs = self.model.get_top_n(q.split(), range(len(self.model.doc_freqs)), n=k)
        return idxs

def _shorten(t, w=600):
    return textwrap.shorten(" ".join((t or "").split()), width=w)

def _label(meta, idx):
    from pathlib import Path as _P
    # If the text came from an image, meta will usually have image_path + flags
    if meta.get("image_path") and (meta.get("caption") or meta.get("ocr") or meta.get("chartqa")):
        src = _P(meta["image_path"]).name + " (from image)"
    else:
        src = _P(meta.get("source", meta.get("image_path","unknown"))).name
    flags = [k for k in ("table_md","table_json","table_row","chartqa","ocr","caption") if meta.get(k)]
    return f"Source {idx}: {src}{' ('+'/'.join(flags)+')' if flags else ''}"

def rerank(query: str, docs: List[str], metas: List[dict], keep: int):
    scores = reranker.predict([(query, d if d else " ") for d in docs])
    best   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    return [docs[i] for i in best], [metas[i] for i in best]

class _BM25Cache:
    def __init__(self):
        self.cache = {}  # path -> (bm25, corpus_docs, corpus_metas)
    def get(self, client, path: Path):
        if path in self.cache:
            return self.cache[path]
        try:
            txt_col = client.get_collection("text")
            if txt_col.count() == 0:
                return (None, [], [])
            corpus = txt_col.get()
            corpus_docs  = corpus["documents"]
            corpus_metas = corpus["metadatas"]
            bm25 = BM25Store([d.lower() for d in corpus_docs])
            self.cache[path] = (bm25, corpus_docs, corpus_metas)
            return bm25, corpus_docs, corpus_metas
        except Exception:
            return (None, [], [])

bm25_cache = _BM25Cache()

class Retriever:
    def __init__(self):
        self.q = asyncio.Queue(16)

    async def frame_receiver(self, _: str, frame: Frame):
        await self.q.put(frame)

    async def run(self, process: Process):
        outputs = process.outputs
        while True:
            frame = await self.q.get()

            other     = getattr(frame, "other_data", None) or {}
            embeds_in = other.get("embeddings")
            q         = (other.get("message", "") or getattr(frame, "text", "") or "").strip()
            db_path   = (process.settings.vector_db_folder_path.value or "").strip()

            if embeds_in is None:
                await outputs.default.send(TextFrame(text="Vector Retrieval: missing 'embeddings' in frame.other_data."))
                self.q.task_done()
                continue

            # K settings
            try:
                K_MMR   = int(process.settings.top_k_mmr.value)
            except Exception:
                K_MMR   = 40
            try:
                K_BM25  = int(process.settings.top_k_bm25.value)
            except Exception:
                K_BM25  = 10
            try:
                K_FINAL = int(process.settings.top_k.value)
            except Exception:
                K_FINAL = 3

            # normalize embeddings
            if isinstance(embeds_in, list) and embeds_in and isinstance(embeds_in[0], (float, int)):
                query_emb = np.asarray(embeds_in, dtype=np.float32)
                query_emb_list = [query_emb.tolist()]
            else:
                query_emb_list = embeds_in
                query_emb = np.asarray(embeds_in[0], dtype=np.float32)

            # Chroma client (TEXT ONLY)
            chroma_dir = Path(os.path.expanduser(db_path)).resolve()
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            try:
                txt_col = client.get_collection("text")
            except Exception:
                txt_col = None

            if not txt_col or txt_col.count() == 0:
                await outputs.default.send(TextFrame(text=f"No text vectors found in Chroma at {chroma_dir}."))
                self.q.task_done()
                continue

            # vector search
            n_initial = max(60, K_MMR + K_BM25 + 10)
            txt = txt_col.query(
                query_embeddings=query_emb_list,
                n_results=n_initial,
                include=['documents','metadatas','embeddings']
            )
            docs_raw  = txt["documents"][0]
            metas_raw = txt["metadatas"][0]
            embs_raw  = np.array(txt["embeddings"][0], dtype=np.float32)

            # MMR
            docs_mmr, metas_mmr = mmr_select(query_emb, embs_raw, docs_raw, metas_raw, k=K_MMR, weight=0.6)

            # BM25
            bm25, corpus_docs, corpus_metas = bm25_cache.get(client, chroma_dir)
            docs_bm, metas_bm = [], []
            if q and bm25:
                idxs = bm25.query(q.lower(), k=K_BM25)
                docs_bm  = [corpus_docs[i]  for i in idxs]
                metas_bm = [corpus_metas[i] for i in idxs]

            # merge + rerank
            docs_all  = docs_mmr + docs_bm
            metas_all = metas_mmr + metas_bm
            if q and docs_all:
                keep_k = min(K_FINAL, len(docs_all))
                docs, metas = rerank(q, docs_all, metas_all, keep=keep_k)
            else:
                docs, metas = docs_all, metas_all

            # top-k final
            n = min(K_FINAL, len(docs))
            docs_out  = [docs[i]  for i in range(n)]
            metas_out = [metas[i] for i in range(n)]

            # build plain text context for LLM
            blocks = []
            for i, (d, m) in enumerate(zip(docs_out, metas_out), 1):
                snippet = _shorten(d, 600)
                blocks.append(f"{_label(m, i)}\n{snippet}")

            context_text = (
                "Use ONLY the sources below. Cite facts like (Source 2).\n"
                f"Question: {q}\n\n"
                + "\n\n".join(blocks)
            )

            # citations (no image_url / pixels)
            citations = []
            for m in metas_out:
                entry = {}
                if m.get("source"): entry["source"] = str(m["source"])
                if "page" in m and m["page"] is not None: entry["page"] = m["page"]
                if not entry and m.get("image_path"):
                    entry["source"] = str(m["image_path"])  # still a useful reference
                if entry:
                    citations.append(entry)

            # compose API messages and request id
            api_in = other.get("api") or []
            api_out = [msg for msg in api_in if isinstance(msg, dict) and msg.get("role") == "system"]
            api_out.append({"role": "user", "content": context_text})

            req_in = other.get("requestId", None) or other.get("request_id", None)
            try:
                rid = int(req_in) if req_in is not None else int(__import__("time").time() * 1000)
            except Exception:
                rid = int(__import__("time").time() * 1000)

            other_data = {
                "type": "vector_search_result",
                "value": [{"document": docs_out, "metadata": metas_out}],
                "message": other.get("message", q),
                "metadata": other.get("metadata", {}),
                "api": api_out,          # <-- text only
                "citations": citations,  # <-- no image_url
                "requestId": rid,
                "request_id": rid,
            }

            # concise debug
            try:
                print("[VectorRetrieval] summary:", {
                    "k": {"mmr": K_MMR, "bm25": K_BM25, "final": K_FINAL},
                    "raw": len(docs_raw),
                    "final": len(docs_out),
                    "images_derived": sum(1 for m in metas_out if m.get("image_path")),
                    "req": rid,
                })
            except Exception:
                pass

            await outputs.default.send(Frame(None, [], None, None, None, other_data))
            self.q.task_done()

retriever = Retriever()

process = CreateElement(Process(
    inputs=Inputs(),
    outputs=Outputs(),
    settings=VSSettings(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-retrv000003",
        name="vector_retrieval",
        displayName="MM - Vector Retrieval",
        version="0.32.0",
        description="Text-only retrieval (MMR+BM25+CrossEncoder). Sends a single text message to the LLM with labeled snippets.",
    ),
    frame_receiver_func=retriever.frame_receiver,
    run_func=retriever.run
))





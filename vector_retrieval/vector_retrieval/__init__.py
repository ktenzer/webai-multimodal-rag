import os
import textwrap
import warnings
import logging
import json
import asyncio
from pathlib import Path
from typing import List

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import TextFrame, Frame

import torch
import numpy as np                                # <-- restored
import chromadb
from chromadb.config import Settings as ChromaSettings

# for PostgresML
import psycopg2

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from rank_bm25 import BM25Okapi

from .element import Inputs, Outputs, Settings

# Quiet warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Constants
TEXT_MODEL   = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading retrieval models …")
embed_model = SentenceTransformer(TEXT_MODEL, device=device)
reranker    = CrossEncoder(RERANK_MODEL, device=device)
print("Models ready\n")

def mmr_select(query_emb, doc_embs, docs, metas, k=20, weight=0.6):
    query_emb  = np.asarray(query_emb, dtype=np.float32)
    doc_embs   = np.asarray(doc_embs,  dtype=np.float32)
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
        return self.model.get_top_n(q.split(), range(len(self.model.doc_freqs)), n=k)

def _shorten(t, w=600):
    return textwrap.shorten(" ".join((t or "").split()), width=w)

def _label(meta, idx):
    from pathlib import Path as _P
    if meta.get("image_path") and (meta.get("caption") or meta.get("ocr") or meta.get("chartqa")):
        src = _P(meta["image_path"]).name + " (from image)"
    else:
        src = _P(meta.get("source", meta.get("image_path","unknown"))).name
    flags = [k for k in ("table_md","table_json","table_row","chartqa","ocr","caption") if meta.get(k)]
    flag_str = f" ({'/'.join(flags)})" if flags else ""
    page = meta.get("page")
    page_str = f" (page {page})" if page is not None else ""
    return f"Source {idx}: {src}{flag_str}{page_str}"

def rerank(query: str, docs: List[str], metas: List[dict], keep: int):
    scores = reranker.predict([(query, d or " ") for d in docs])
    best   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    return [docs[i] for i in best], [metas[i] for i in best]

class _BM25Cache:
    def __init__(self):
        self.cache = {}
    def get(self, client, path: Path):
        if path in self.cache:
            return self.cache[path]
        try:
            col = client.get_collection("text")
            if col.count() == 0:
                return (None, [], [])
            data        = col.get()
            docs        = data["documents"]
            metas       = data["metadatas"]
            bm          = BM25Store([d.lower() for d in docs])
            self.cache[path] = (bm, docs, metas)
            return bm, docs, metas
        except:
            return (None, [], [])

bm25_cache = _BM25Cache()

class Retriever:
    def __init__(self):
        self.q = asyncio.Queue(16)

    async def frame_receiver(self, _: str, frame: Frame):
        await self.q.put(frame)

    async def run(self, process: Process):
        outputs  = process.outputs
        settings = process.settings

        while True:
            frame    = await self.q.get()
            other    = getattr(frame, "other_data", {}) or {}
            embeds_in= other.get("embeddings")
            q_text   = (other.get("message","") or getattr(frame,"text","") or "").strip()

            if embeds_in is None:
                await outputs.default.send(TextFrame(text="Missing 'embeddings'"))
                self.q.task_done()
                continue

            K_MMR   = settings.top_k_mmr.value
            K_BM25  = settings.top_k_bm25.value
            K_FINAL = settings.top_k.value
            debug   = settings.debug.value

            if isinstance(embeds_in[0], (float,int)):
                query_emb  = np.asarray(embeds_in, dtype=np.float32)
                q_emb_list = [query_emb.tolist()]
            else:
                q_emb_list = embeds_in
                query_emb  = np.asarray(embeds_in[0], dtype=np.float32)

            store_type = settings.vector_store_type.value

            # ── Chroma ─────────────────────────────────────────────────────────────
            n_initial = max(60, K_MMR + K_BM25 + 10)
            if store_type == "chromadb":
                db_dir = Path(os.path.expanduser(settings.vector_db_folder_path.value)).resolve()
                client = chromadb.PersistentClient(path=str(db_dir),
                                                   settings=ChromaSettings(anonymized_telemetry=False))
                try:
                    txt_col = client.get_collection("text")
                except:
                    txt_col = None
                if not txt_col or txt_col.count()==0:
                    await outputs.default.send(TextFrame(text=f"No text in Chroma at {db_dir}"))
                    self.q.task_done()
                    continue

                rec = txt_col.query(
                    query_embeddings=q_emb_list,
                    n_results=n_initial,
                    include=['documents','metadatas','embeddings']
                )
                docs_raw, metas_raw = rec["documents"][0], rec["metadatas"][0]
                embs_raw = np.array(rec["embeddings"][0], dtype=np.float32)

            # ── PostgresML ─────────────────────────────────────────────────────────
            else:
                host   = settings.pgml_host.value
                port   = settings.pgml_port.value
                db     = settings.pgml_db.value
                user   = settings.pgml_user.value
                pwd    = settings.pgml_password.value or None
                table  = settings.postgres_table_name.value

                # compute query embedding the same way as for Chroma
                q_emb = embed_model.encode(q_text, convert_to_numpy=True)

                # n_initial must match the Chroma branch
                n_initial = max(60, K_MMR + K_BM25 + 10)

                conn = psycopg2.connect(
                    host=host, port=port, user=user, password=pwd, dbname=db
                )
                cur = conn.cursor()
                cur.execute(
                    f"""
                    SELECT document, metadata, embedding
                    FROM {table}
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                    """,
                    (
                        q_emb.tolist(),   # your float list
                        n_initial,        # same as Chroma
                    ),
                )
                rows = cur.fetchall()
                cur.close()
                conn.close()

                docs_raw, metas_raw, emb_list = [], [], []
                for doc, meta, emb in rows:
                    docs_raw.append(doc)
                    metas_raw.append(meta)
                    # parse any string-encoded vectors
                    if isinstance(emb, str):
                        try:
                            arr = json.loads(emb)
                        except Exception:
                            import ast
                            arr = ast.literal_eval(emb)
                    else:
                        arr = emb
                    emb_list.append(arr)

                embs_raw = np.asarray(emb_list, dtype=np.float32)

            # MMR
            docs_mmr, metas_mmr = mmr_select(query_emb, embs_raw, docs_raw, metas_raw,
                                             k=K_MMR, weight=0.4)

            # BM25 lexical
            bm25, corpus_docs, corpus_metas = bm25_cache.get(
                client if store_type=="chroma" else None,
                Path(settings.vector_db_folder_path.value)
            )
            docs_bm, metas_bm = [], []
            if q_text and bm25:
                idxs = bm25.query(q_text.lower(), k=K_BM25)
                docs_bm  = [corpus_docs[i]  for i in idxs]
                metas_bm = [corpus_metas[i] for i in idxs]

            # merge + rerank
            docs_all, metas_all = docs_mmr + docs_bm, metas_mmr + metas_bm
            if q_text and docs_all:
                scores = reranker.predict([(q_text, d or " ") for d in docs_all])
                if max(scores) < 0.35 and docs_bm:
                    docs, metas = docs_bm[:K_FINAL], metas_bm[:K_FINAL]
                else:
                    docs, metas = rerank(q_text, docs_all, metas_all, keep=K_FINAL)
            else:
                docs, metas = docs_all, metas_all

            # final top-K
            n = min(K_FINAL, len(docs))
            docs_out  = docs[:n]
            metas_out = metas[:n]

            # build prompt
            blocks = [f"{_label(m,i+1)}\n{_shorten(d,600)}"
                      for i,(d,m) in enumerate(zip(docs_out, metas_out))]
            citation_lines = [
                f"{Path(m.get('source') or m.get('image_path','')).name}"
                + (f", page {m['page']}" if m.get("page") is not None else "")
                for m in metas_out
            ]
            context = "\n".join([
                "You are given the following sources. Answer using ONLY these.",
                "Do NOT use inline citations.",
                f"Question: {q_text}",
                *blocks,
                "",
                "Citation(s):",
                *citation_lines
            ])

            req_id = other.get("request_id") or other.get("requestId")
            if req_id is None:
                req_id = int(asyncio.get_event_loop().time() * 1000)

            other_data = {
                "type": "vector_search_result",
                "value": [{"document": docs_out, "metadata": metas_out}],
                "message": q_text,
                "api": [{"role": "user", "content": context}],
                # include both keys so the API element finds it
                "request_id": req_id,
                "requestId": req_id,
            }

            if debug:
                print("[VectorRetrieval DEBUG]", {
                    "store": store_type,
                    "mmr":   len(docs_mmr),
                    "bm25":  len(docs_bm),
                    "final": len(docs_out),
                })

            await outputs.default.send(Frame(None,[],None,None,None,other_data))
            self.q.task_done()

retriever = Retriever()

process = CreateElement(
    Process(
        inputs=Inputs(),
        outputs=Outputs(),
        settings=Settings(),
        metadata=ProcessMetadata(
            id="2a7a0b6a-7b84-4c57-8f1c-retrv000003",
            name="vector_retrieval",
            displayName="MM - Vector Retrieval",
            version="0.52.0",
            description="Retrieves via Chroma or PostgresML",
        ),
        frame_receiver_func=retriever.frame_receiver,
        run_func=retriever.run,
    )
)







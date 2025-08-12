import os
import time
import textwrap
import warnings
import logging
import json
import asyncio
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import TextFrame, Frame

import torch
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings

import psycopg2

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from rank_bm25 import BM25Okapi

from .element import Inputs, Outputs, Settings

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

RERANK_MODEL = "BAAI/bge-reranker-base"

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Helpers
def mmr_select(query_emb, doc_embs, docs, metas, k=20, weight=0.6):
    query_emb  = np.asarray(query_emb, dtype=np.float32)
    doc_embs   = np.asarray(doc_embs,  dtype=np.float32)
    if len(docs) == 0 or doc_embs.size == 0:
        return [], []
    selected, sel_docs, sel_metas = [], [], []
    candidates = list(range(len(docs)))
    sims_query = cos_sim(torch.tensor(query_emb), torch.tensor(doc_embs)).numpy().flatten()
    while len(selected) < k and candidates:
        if not selected:
            local_idx = int(np.argmax(sims_query[candidates]))
            idx = candidates[local_idx]
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

def rerank(query: str, docs: List[str], metas: List[dict], keep: int, reranker_model: CrossEncoder):
    scores = reranker_model.predict([(query, d or " ") for d in docs])
    best   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    return [docs[i] for i in best], [metas[i] for i in best]

class _BM25Cache:
    def __init__(self):
        self.cache = {}
    def get(self, client, base: str, db_dir: Path):
        key = (db_dir, base)
        if key in self.cache:
            return self.cache[key]
        try:
            cols = []
            for name in (f"{base}_text", f"{base}_images"):
                try:
                    c = client.get_collection(name)
                    if c and c.count() > 0:
                        cols.append(c)
                except:
                    pass
            docs, metas = [], []
            for c in cols:
                data  = c.get()
                docs += data["documents"]
                metas += data["metadatas"]
            if not docs:
                return (None, [], [])
            bm = BM25Store([d.lower() for d in docs])
            self.cache[key] = (bm, docs, metas)
            return bm, docs, metas
        except Exception:
            return (None, [], [])

bm25_cache = _BM25Cache()

def _base(settings) -> str:
    return settings.postgres_table_name.value or "default"

def _combine_results_chroma(client, base: str, q_emb_list, n_initial: int):
    import numpy as _np
    docs_all, metas_all, embs_all = [], [], []
    for col_name in (f"{base}_text", f"{base}_images"):
        try:
            col = client.get_collection(col_name)
        except Exception:
            col = None
        if not col or col.count() == 0:
            continue

        rec = col.query(
            query_embeddings=q_emb_list,
            n_results=n_initial,
            include=["documents", "metadatas", "embeddings"],
        )

        def _first(lst, default):
            if lst is None:
                return default
            try:
                return lst[0] if len(lst) > 0 else default
            except Exception:
                return default

        docs  = _first(rec.get("documents"), [])
        metas = _first(rec.get("metadatas"), [])
        embs  = _first(rec.get("embeddings"), [])

        if isinstance(embs, _np.ndarray):
            embs = embs.tolist()
        else:
            embs = list(embs) if embs else []

        if docs:  docs_all.extend(docs)
        if metas: metas_all.extend(metas)
        if embs:  embs_all.extend(embs)

    if embs_all:
        return docs_all, metas_all, _np.asarray(embs_all, dtype=_np.float32)
    else:
        return docs_all, metas_all, _np.zeros((0, 0), dtype=_np.float32)

def _query_pg_table(conn, table: str, q_emb: list, limit: int):
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT document, metadata, embedding
            FROM {table}
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (q_emb, limit),
        )
        rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        cur.close()
    docs, metas, emb_list = [], [], []
    for doc, meta, emb in rows:
        docs.append(doc); metas.append(meta)
        if isinstance(emb, str):
            try:
                arr = json.loads(emb)
            except Exception:
                import ast
                arr = ast.literal_eval(emb)
        else:
            arr = emb
        emb_list.append(arr)
    return docs, metas, emb_list

# Attach images from the same (source, page) as retrieved text
def _collect_page_pairs(metas: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    pairs = []
    for m in metas:
        src = m.get("source")
        page = m.get("page")
        if src and (page is not None):
            try:
                pairs.append((str(src), int(page)))
            except Exception:
                continue
    seen = set(); out = []
    for p in pairs:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _images_for_pages_chroma(client, base: str, page_pairs: List[Tuple[str, int]],
                             max_per_page: int = 3, max_total: int = 8) -> List[Dict[str, Any]]:
    imgs: List[Dict[str, Any]] = []
    try:
        col = client.get_collection(f"{base}_images")
    except Exception:
        col = None
    if not col:
        return imgs

    for (src, pg) in page_pairs:
        try:
            res = col.get(where={"source": src, "page": pg})
        except Exception:
            data = col.get()
            metas = data.get("metadatas", [])
            docs  = data.get("documents", [])
            keep = []
            for md, _d in zip(metas, docs):
                if isinstance(md, dict) and md.get("source") == src and md.get("page") == pg:
                    keep.append(md)
            res = {"metadatas": keep}

        md_list = res.get("metadatas") or []
        if md_list and isinstance(md_list[0], list):
            md_list = md_list[0]

        added = 0
        for md in md_list:
            if not isinstance(md, dict):
                continue
            ip = md.get("image_path")
            if not ip:
                continue
            imgs.append({"source": src, "page": pg, "image_path": ip})
            added += 1
            if added >= max_per_page:
                break
        if len(imgs) >= max_total:
            break
    dedup, seen = [], set()
    for it in imgs:
        k = it["image_path"]
        if k in seen: continue
        seen.add(k); dedup.append(it)
    return dedup

def _images_for_pages_pg(pg_conn_info: Dict[str, Any], base: str, page_pairs: List[Tuple[str, int]],
                         max_per_page: int = 3, max_total: int = 8) -> List[Dict[str, Any]]:
    imgs: List[Dict[str, Any]] = []
    if not page_pairs:
        return imgs
    conn = psycopg2.connect(**pg_conn_info)
    try:
        cur = conn.cursor()
        for (src, pg) in page_pairs:
            try:
                cur.execute(
                    f"""
                    SELECT metadata
                    FROM {base}_images
                    WHERE (metadata->>'source') = %s
                      AND (metadata->>'page')::int = %s
                    LIMIT %s
                    """,
                    (src, int(pg), max_per_page)
                )
                rows = cur.fetchall()
            except Exception:
                rows = []
            for (meta,) in rows:
                ip = (meta or {}).get("image_path")
                if ip:
                    imgs.append({"source": src, "page": int(pg), "image_path": ip})
            if len(imgs) >= max_total:
                break
        dedup, seen = [], set()
        for it in imgs:
            k = it["image_path"]
            if k in seen: continue
            seen.add(k); dedup.append(it)
        return dedup
    finally:
        try: cur.close()
        except: pass
        conn.close()

# Enrich image entries with absolute path, file://URL and Markdown link 
def _enrich_images(imgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in imgs:
        ip = it.get("image_path")
        if not ip:
            continue
        try:
            ap = str(Path(ip).expanduser().resolve())
            file_url = Path(ap).as_uri() 
        except Exception:
            ap = str(Path(ip).expanduser())
            file_url = ("file://" + ap) if ap.startswith("/") else ("file:///" + ap)
        fname = Path(ap).name
        md_link = f"[{fname}]({file_url})"
        out.append({**it, "abs_path": ap, "file_url": file_url, "filename": fname, "md_link": md_link})
    return out

# Group & merge snippets by (source, page)
def _merge_by_page(docs: List[str], metas: List[dict], max_chars: int = 1600) -> Tuple[List[str], List[dict]]:
    key = lambda m: (m.get("source"), m.get("page"))
    groups: Dict[Tuple[str,int], List[int]] = {}
    order: List[Tuple[str,int]] = []
    for i, m in enumerate(metas):
        k = key(m)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(i)

    merged_docs, merged_metas = [], []
    for k in order:
        idxs = groups[k]
        text = " ".join(docs[i] or "" for i in idxs).strip()
        if len(text) > max_chars:
            text = text[:max_chars]
            if " " in text:
                text = text.rsplit(" ", 1)[0]
        merged_docs.append(text)
        merged_metas.append(metas[idxs[0]])
    return merged_docs, merged_metas

# Procedural intent
_PROCEDURAL_RE = re.compile(r"\b(install|installation|steps|procedure|instructions?|set ?up|assemble|mount|tighten|remove|replace|adjust)\b", re.I)
def _is_procedural(q: str) -> bool:
    return bool(_PROCEDURAL_RE.search(q or ""))

# Retriever
class Retriever:
    def __init__(self):
        self.q = asyncio.Queue(16)
        self.reranker: CrossEncoder | None = None

    async def frame_receiver(self, _: str, frame: Frame):
        await self.q.put(frame)

    async def run(self, process: Process):
        outputs  = process.outputs
        settings = process.settings

        # Lazy-load retrieval models (once per process)
        if self.reranker is None:
            print("Loading retrieval models …")
            self.reranker    = CrossEncoder(RERANK_MODEL, device=device)
            print("Models ready\n")

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

            if _is_procedural(q_text):
                K_FINAL = max(K_FINAL, 6)
                K_MMR   = max(K_MMR, 24)

            if isinstance_embeds := isinstance(embeds_in[0], (float,int)):
                query_emb  = np.asarray(embeds_in, dtype=np.float32)
                q_emb_list = [query_emb.tolist()]
            else:
                q_emb_list = embeds_in
                query_emb  = np.asarray(embeds_in[0], dtype=np.float32)

            store_type = settings.vector_store_type.value
            n_initial = max(60, K_MMR + K_BM25 + 10)

            # ChromaDB
            if store_type == "chromadb":
                db_dir = Path(os.path.expanduser(settings.vector_db_folder_path.value)).resolve()
                client = chromadb.PersistentClient(path=str(db_dir), settings=ChromaSettings(anonymized_telemetry=False))

                base = _base(settings)
                docs_raw, metas_raw, embs_raw = _combine_results_chroma(client, base, q_emb_list, n_initial)

                if not docs_raw or embs_raw.size == 0:
                    await outputs.default.send(TextFrame(text=f"No candidates found in Chroma for base '{base}'"))
                    self.q.task_done()
                    continue

                bm25, corpus_docs, corpus_metas = bm25_cache.get(client, base, db_dir)

            # PostgresML
            else:
                host   = settings.pgml_host.value
                port   = settings.pgml_port.value
                db     = settings.pgml_db.value
                user   = settings.pgml_user.value
                pwd    = settings.pgml_password.value or None
                base   = _base(settings)

                # Loading specific embedding model
                q_emb = query_emb.tolist()
                pg_conn_info = {"host": host, "port": port, "user": user, "password": pwd, "dbname": db}

                conn = psycopg2.connect(**pg_conn_info)
                try:
                    docs_raw, metas_raw, emb_list = [], [], []
                    for tbl in (f"{base}_text", f"{base}_images"):
                        d, m, e = _query_pg_table(conn, tbl, q_emb, n_initial)
                        docs_raw += d; metas_raw += m; emb_list += e
                finally:
                    conn.close()

                if not docs_raw:
                    await outputs.default.send(TextFrame(text=f"No data in PostgresML for base '{base}'"))
                    self.q.task_done()
                    continue

                embs_raw = np.asarray(emb_list, dtype=np.float32)
                bm25 = None
                corpus_docs = []; corpus_metas = []

            # MMR over ANN candidates
            docs_mmr, metas_mmr = mmr_select(query_emb, embs_raw, docs_raw, metas_raw, k=K_MMR, weight=0.4)

            # BM25 lexical (Chroma-only cache)
            docs_bm, metas_bm = [], []
            if q_text and bm25:
                idxs = bm25.query(q_text.lower(), k=K_BM25)
                docs_bm  = [corpus_docs[i]  for i in idxs]
                metas_bm = [corpus_metas[i] for i in idxs]

            # Merge and rerank
            docs_all, metas_all = docs_mmr + docs_bm, metas_mmr + metas_bm
            if q_text and docs_all:
                scores = self.reranker.predict([(q_text, d or " ") for d in docs_all])
                if max(scores) < 0.35 and docs_bm:
                    docs, metas = docs_bm[:K_FINAL], metas_bm[:K_FINAL]
                else:
                    docs, metas = rerank(q_text, docs_all, metas_all, keep=K_FINAL, reranker_model=self.reranker)
            else:
                docs, metas = docs_all, metas_all

            # Merge snippets by (source,page)
            docs_merged, metas_merged = _merge_by_page(docs, metas, max_chars=1600)

            # Attach images by (source, page) and enrich with file URLs
            page_pairs = _collect_page_pairs(metas_merged)
            images_attached: List[Dict[str, Any]] = []
            if store_type == "chromadb":
                images_attached = _images_for_pages_chroma(client, base, page_pairs)
            else:
                images_attached = _images_for_pages_pg(pg_conn_info, base, page_pairs)
            images_attached = _enrich_images(images_attached)

            # Build prompt blocks and citations
            blocks = []
            for i, (d, m) in enumerate(zip(docs_merged, metas_merged), 1):
                snippet = _shorten(d, 600)
                blocks.append(f"{_label(m, i)}\n{snippet}")

            # Always surface images if we have them, as Markdown links (clickable)
            if images_attached:
                lines = []
                for im in images_attached[:8]:
                    lines.append(f"- {im['md_link']} — {Path(im['source']).name}, page {im['page']}")
                blocks.append("\nRelated figure(s):\n" + "\n".join(lines))

            # Unique citations
            seen_cites = set()
            citation_lines = []
            for m in metas_merged:
                src = m.get("source") or m.get("image_path", "")
                page = m.get("page")
                cite = (Path(src).name if src else "unknown", page)
                if cite in seen_cites:
                    continue
                seen_cites.add(cite)
                if page is not None:
                    citation_lines.append(f"{cite[0]}, page {page}")
                else:
                    citation_lines.append(f"{cite[0]}")

            context = "\n".join([
                "You are given the following sources. Answer the question using ONLY these sources.",
                "Do **not** put inline citations like (Source 1).",
                "At the very end of your answer, include a “Citation(s):” section listing each source and page.",
                "If related figures are listed, include them as bullet lines like: - [filename](file:///…).",
                "",
                f"Question: {q_text}",
                "",
                *blocks,
                "",
                "Citation(s):\n",
                *citation_lines
            ])

            # Citations array 
            citations = []
            for m in metas_merged:
                entry = {}
                if m.get("source"): entry["source"] = str(m["source"])
                if "page" in m and m["page"] is not None: entry["page"] = m["page"]
                if not entry and m.get("image_path"):
                    entry["source"] = str(m["image_path"])
                if entry:
                    citations.append(entry)

            # Request id passthrough
            req_in = (
                other.get("requestId", None)
                or other.get("request_id", None)
                or (other.get("metadata", {}) or {}).get("requestId", None)
                or (other.get("metadata", {}) or {}).get("request_id", None)
            )
            if isinstance(req_in, int):
                rid = req_in
            elif isinstance(req_in, str) and req_in.isdigit():
                rid = int(req_in)
            else:
                rid = 0

            api_in  = other.get("api") or []
            api_out = [msg for msg in api_in if isinstance(msg, dict) and msg.get("role") == "system"]
            if images_attached:
                api_out.append({
                    "role": "system",
                    "content": "If related figures are provided, include them as bullet lines like: - [filename](file:///…)."
                })
            api_out.append({"role": "user", "content": context})

            other_data = {
                "type": "vector_search_result",
                "value": [{"document": docs_merged, "metadata": metas_merged}],
                "message": other.get("message", q_text),
                "metadata": other.get("metadata", {}),
                "api": api_out,
                "requestId": rid,
                "request_id": rid,
                "images": images_attached,
            }

            if debug:
                print("[VectorRetrieval DEBUG] summary:", {
                    "k": {"mmr": settings.top_k_mmr.value, "bm25": settings.top_k_bm25.value, "final": settings.top_k.value},
                    "raw_candidates": len(docs_raw),
                    "final_docs": len(docs_merged),
                    "attached_images": len(images_attached),
                    "request_id": rid,
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
            version="0.67.0",
            description="Retrieves from <base>_text and <base>_images; merges page snippets; surfaces figures as Markdown links to file://."
        ),
        frame_receiver_func=retriever.frame_receiver,
        run_func=retriever.run,
    )
)












import os, textwrap, warnings, logging, json
from pathlib import Path
from typing import List

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import TextFrame, Frame

import torch, chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import EmbeddingFunction
import asyncio
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

from rank_bm25 import BM25Okapi
import numpy as np

from .element import Inputs, Outputs, Settings as VSSettings

# Quiet warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Constants (defaults; can be overridden by settings)
TEXT_MODEL   = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
CLIP_MODEL   = "openai/clip-vit-base-patch32"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading retrieval models …")
embed_model = SentenceTransformer(TEXT_MODEL, device=device)  # not used for query here
reranker    = CrossEncoder(RERANK_MODEL, device=device)       # used for rerank
clip_model  = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
clip_proc   = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=True)
print("Models ready\n")

# CLIP text EF
class CLIPTextEF(EmbeddingFunction):
    def __init__(self):
        self.proc, self.model = clip_proc, clip_model
    def name(self): return f"clip-text-{CLIP_MODEL}"
    def dimensions(self): return 512
    def __call__(self, texts):
        with torch.no_grad():
            inp = self.proc(text=texts, return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items()}
            self.model.to(device)
            return self.model.get_text_features(**inp).cpu().numpy().tolist()

clip_text_ef = CLIPTextEF()

# MMR
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

# BM25
class BM25Store:
    def __init__(self, docs_lower: List[str]):
        self.model = BM25Okapi([d.split() for d in docs_lower])
    def query(self, q: str, k: int):
        idxs = self.model.get_top_n(q.split(), range(len(self.model.doc_freqs)), n=k)
        return idxs

def shorten(t, w=200): return textwrap.shorten(" ".join((t or "").split()), width=w)

def label(meta, idx):
    from pathlib import Path as _P
    if meta.get("image_path") and meta.get("caption"):
        src = _P(meta["image_path"]).name + " (image)"
    else:
        src = _P(meta.get("source", meta.get("image_path","unknown"))).name
    flags = [k for k in ("table_md","table_json","table_row","chartqa","ocr") if meta.get(k)]
    return f"Source {idx}: {src}{' ('+'/'.join(flags)+')' if flags else ''}"

def rerank(query: str, docs: List[str], metas: List[dict], keep: int):
    scores = reranker.predict([(query, d if d else " ") for d in docs])
    best   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    return [docs[i] for i in best], [metas[i] for i in best]

# Cache BM25 per DB path
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

            # ---- Read embeddings + DB path + optional message ----
            other     = getattr(frame, "other_data", None) or {}
            embeds_in = other.get("embeddings")  # list[float] OR list[list[float]]
            q         = (other.get("message", "") or getattr(frame, "text", "") or "").strip()
            db_path   = (process.settings.vector_db_folder_path.value or "").strip()

            if embeds_in is None:
                await outputs.default.send(TextFrame(text="Vector Retrieval: missing 'embeddings' in frame.other_data."))
                self.q.task_done()
                continue

            # Read K settings (with safe defaults)
            try:
                K_MMR   = int(process.settings.top_k_mmr.value)
            except Exception:
                K_MMR   = 40
            try:
                K_BM25  = int(process.settings.top_k_bm25.value)
            except Exception:
                K_BM25  = 10
            try:
                K_IMAGE = int(process.settings.top_k_image.value)
            except Exception:
                K_IMAGE = 2
            try:
                K_FINAL = int(process.settings.top_k.value)
            except Exception:
                K_FINAL = 3

            # Normalize embeddings for Chroma (list[list[float]])
            if isinstance(embeds_in, list) and embeds_in and isinstance(embeds_in[0], (float, int)):
                query_emb = np.asarray(embeds_in, dtype=np.float32)
                query_emb_list = [query_emb.tolist()]
            else:
                query_emb_list = embeds_in
                query_emb = np.asarray(embeds_in[0], dtype=np.float32)

            # Open client for this DB path
            chroma_dir = Path(os.path.expanduser(db_path)).resolve()
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=ChromaSettings(anonymized_telemetry=False),
            )

            # Collections
            try:
                txt_col = client.get_collection("text")
            except Exception:
                txt_col = None
            try:
                img_col = client.get_collection("images")
            except Exception:
                img_col = None

            if not txt_col or txt_col.count() == 0:
                await outputs.default.send(TextFrame(text=f"No text vectors found in Chroma at {chroma_dir}."))
                self.q.task_done()
                continue

            # Initial vector search using provided embeddings
            txt = txt_col.query(
                query_embeddings=query_emb_list,
                n_results=60,
                include=['documents', 'metadatas', 'embeddings']
            )
            docs_raw  = txt["documents"][0]
            metas_raw = txt["metadatas"][0]
            embs_raw  = np.array(txt["embeddings"][0], dtype=np.float32)

            # MMR diversity over top-60
            docs_mmr, metas_mmr = mmr_select(query_emb, embs_raw, docs_raw, metas_raw, k=K_MMR, weight=0.6)

            # BM25 lexical retrieval (only if we have text and a BM25 index)
            bm25, corpus_docs, corpus_metas = bm25_cache.get(client, chroma_dir)
            docs_bm, metas_bm = [], []
            if q and bm25:
                idxs = bm25.query(q.lower(), k=K_BM25)
                docs_bm  = [corpus_docs[i]  for i in idxs]
                metas_bm = [corpus_metas[i] for i in idxs]

            # Merge & CrossEncoder rerank (requires q)
            docs_all  = docs_mmr + docs_bm
            metas_all = metas_mmr + metas_bm
            if q and docs_all:
                keep_k = min(K_FINAL, len(docs_all))
                docs, metas = rerank(q, docs_all, metas_all, keep=keep_k)
            else:
                docs, metas = docs_all, metas_all

            # CLIP image retrieval (optional, requires q)
            if q and img_col and img_col.count() > 0 and K_IMAGE > 0:
                clip_vec = clip_text_ef([q])[0]
                img = img_col.query(query_embeddings=[clip_vec], n_results=K_IMAGE)
                docs  += img["documents"][0]
                metas += img["metadatas"][0]

            # Top-k slice (defensive; respect final top_k)
            n = min(K_FINAL, len(docs))
            docs_out  = [docs[i]  for i in range(n)]
            metas_out = [metas[i] for i in range(n)]

            # Build context text for LLM
            def _label(meta, idx):
                from pathlib import Path as _P
                if meta.get("image_path") and meta.get("caption"):
                    src = _P(meta["image_path"]).name + " (image)"
                else:
                    src = _P(meta.get("source", meta.get("image_path","unknown"))).name
                flags = [k for k in ("table_md","table_json","table_row","chartqa","ocr") if meta.get(k)]
                return f"Source {idx}: {src}{' ('+'/'.join(flags)+')' if flags else ''}"

            def _shorten(t, w=600):
                import textwrap as _tw
                return _tw.shorten(" ".join((t or "").split()), width=w)

            blocks = []
            for i, (d, m) in enumerate(zip(docs_out, metas_out), 1):
                content = "<image>" if m.get("image_path") else _shorten(d, 600)
                blocks.append(f"{_label(m, i)}\n{content}")

            context_text = (
                "Use ONLY the sources below to answer.\n"
                f"Question: {q}\n\n"
                + "\n\n".join(blocks)
                + "\n\nCite facts like (Source 2)."
            )

            # Citations for UI
            citations = []
            for m in metas_out:
                src = m.get("source") or m.get("image_path")
                if not src:
                    continue
                entry = {"source": str(src)}
                if "page" in m and m["page"] is not None:
                    entry["page"] = m["page"]
                citations.append(entry)

            # Compose API messages and request id
            api_in = other.get("api") or []
            api_out = []
            for msg in api_in:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    api_out.append(msg)
            api_out.append({"role": "user", "content": context_text})

            req_in = other.get("requestId", None)
            if req_in is None:
                req_in = other.get("request_id", None)
            try:
                rid = int(req_in) if req_in is not None else int(__import__("time").time() * 1000)
            except Exception:
                rid = int(__import__("time").time() * 1000)

            # Payload
            other_data = {
                "type": "vector_search_result",
                "value": [
                    {
                        "document": docs_out,
                        "metadata": metas_out,
                    }
                ],
                "message": other.get("message", q),
                "metadata": other.get("metadata", {}),
                "api": api_out,
                "citations": citations,
                "requestId": rid,
                "request_id": rid,
            }

            # Debug summary
            try:
                print("[VectorRetrieval] sending other_data summary:", {
                    "type": other_data["type"],
                    "n_docs": len(other_data["value"][0]["document"]),
                    "has_api": isinstance(other_data.get("api"), list),
                    "n_citations": len(other_data.get("citations", [])),
                    "message": other_data.get("message", ""),
                    "requestId": other_data.get("requestId"),
                    "K_MMR": K_MMR, "K_BM25": K_BM25, "K_IMAGE": K_IMAGE, "K_FINAL": K_FINAL,
                })
            except Exception as e:
                print(f"[VectorRetrieval] preview failed: {e}")

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
        version="0.22.0",
        description="Consumes embeddings; performs MMR, BM25, CrossEncoder rerank, and CLIP; outputs structured results with requestId. K-values are user-configurable.",
    ),
    frame_receiver_func=retriever.frame_receiver,
    run_func=retriever.run
))




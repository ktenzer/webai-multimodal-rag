import os, sys, time, textwrap, warnings, logging, json
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import TextFrame, Frame

import torch, chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction

from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

from rank_bm25 import BM25Okapi
import numpy as np
from openai import OpenAI

# ---- env check (kept) ----
load_dotenv(override=True)
BASE_URL, API_KEY = os.getenv("OPENAI_BASE_URL"), os.getenv("OPENAI_API_KEY")
if not BASE_URL or not API_KEY:
    sys.exit("ERROR: OPENAI_BASE_URL / OPENAI_API_KEY missing in .env")
oa = OpenAI(base_url=BASE_URL, api_key=API_KEY)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

CHROMA_DIR = Path("./chroma_db/webai")
TEXT_MODEL    = "BAAI/bge-base-en-v1.5"
RERANK_MODEL  = "BAAI/bge-reranker-base"
CLIP_MODEL    = "openai/clip-vit-base-patch32"
TOP_K_MMR     = 40
TOP_K_BM25    = 10
TOP_K_IMAGE   = 2
TOP_K_RERANK  = 4
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading retrieval models …")
embed_model = SentenceTransformer(TEXT_MODEL, device=device)
reranker    = CrossEncoder(RERANK_MODEL, device=device)
clip_model  = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
clip_proc   = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=True)
print("Models ready\n")

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

def shorten(t, w=200): return textwrap.shorten(" ".join(t.split()), width=w)
def label(meta, idx):
    from pathlib import Path
    if meta.get("image_path") and meta.get("caption"):
        src = Path(meta["image_path"]).name + " (image)"
    else:
        src = Path(meta.get("source", meta.get("image_path","unknown"))).name
    flags = [k for k in ("table_md","table_json","table_row","chartqa","ocr") if meta.get(k)]
    return f"Source {idx}: {src}{' ('+'/'.join(flags)+')' if flags else ''}"

def rerank(query: str, docs: List[str], metas: List[dict], keep=TOP_K_RERANK):
    scores = reranker.predict([(query, d if d else " ") for d in docs])
    best   = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:keep]
    return [docs[i] for i in best], [metas[i] for i in best]

import asyncio
class Retriever:
    def __init__(self):
        self.q = asyncio.Queue(16)
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
        try:
            self.txt_col = self.client.get_collection("text")
        except Exception:
            self.txt_col = None
        try:
            self.img_col = self.client.get_collection("images")
        except Exception:
            self.img_col = None
        self.bm25 = None
        if self.txt_col and self.txt_col.count() > 0:
            corpus_docs  = self.txt_col.get()["documents"]
            self.corpus_metas = self.txt_col.get()["metadatas"]
            self.bm25 = BM25Store([d.lower() for d in corpus_docs])
        else:
            print("No text vectors found in Chroma.")

    async def frame_receiver(self, _: str, frame: Frame): 
        await self.q.put(frame)

    async def run(self, process: Process):
        outputs = process.outputs
        while True:
            frame = await self.q.get()
            q = (frame.text or "").strip()
            if not q:
                self.q.task_done()
                continue

            txt = self.txt_col.query(
                query_texts=[q],
                n_results=60,
                include=['documents', 'metadatas', 'embeddings']
            )
            docs_raw   = txt["documents"][0]
            metas_raw  = txt["metadatas"][0]
            embs_raw   = np.array(txt["embeddings"][0])

            query_emb = embed_model.encode(q, convert_to_numpy=True)
            docs_mmr, metas_mmr = mmr_select(query_emb, embs_raw, docs_raw, metas_raw, k=TOP_K_MMR, weight=0.6)

            idxs = self.bm25.query(q.lower(), k=10) if self.bm25 else []
            corpus_docs  = self.txt_col.get()["documents"] if self.bm25 else []
            corpus_metas = self.txt_col.get()["metadatas"] if self.bm25 else []
            docs_bm  = [corpus_docs[i]  for i in idxs]
            metas_bm = [corpus_metas[i] for i in idxs]

            docs_all  = docs_mmr + docs_bm
            metas_all = metas_mmr + metas_bm
            docs, metas = rerank(q, docs_all, metas_all, keep=TOP_K_BM25)

            if self.img_col and self.img_col.count() > 0:
                clip_vec = clip_text_ef([q])[0]
                img = self.img_col.query(query_embeddings=[clip_vec], n_results=TOP_K_IMAGE)
                docs  += img["documents"][0]
                metas += img["metadatas"][0]

            print("\n🔎 Retrieved context:")
            for i, (d, m) in enumerate(zip(docs, metas), 1):
                snippet = "<image>" if m.get("image_path") else shorten(d)
                print(f"  {i}. {label(m, i)} — {snippet}")
            print("")

            blocks = []
            for i, (d, m) in enumerate(zip(docs, metas), 1):
                content = f"<image at {m['image_path']}>" if m.get("image_path") else d
                blocks.append(f"{label(m, i)}\n{content}")

            prompt = (
                "You are a helpful assistant. Use ONLY the sources below.\n\n"
                f"User question: {q}\n\n" + "\n\n".join(blocks) +
                "\n\nCite facts like (Source 2)."
            )

            ans = oa.chat.completions.create(
                model="webai",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            ).choices[0].message.content

            await outputs.default.send(TextFrame(text=ans))
            self.q.task_done()

retriever = Retriever()

process = CreateElement(Process(
    inputs=__import__('vector_retrieval_element.element', fromlist=['Inputs']).Inputs(),
    outputs=__import__('vector_retrieval_element.element', fromlist=['Outputs']).Outputs(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-retrv000001",
        name="vector_retrieval",
        displayName="Vector Retrieval Element",
        version="0.1.0",
        description="Semantic retrieval + rerank + LLM answer, from Chroma."
    ),
    frame_receiver_func=retriever.frame_receiver,
    run_func=retriever.run
))

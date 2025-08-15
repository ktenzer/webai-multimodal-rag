# ocr/__init__.py
import os, mimetypes, warnings, logging, re, json, tempfile
import asyncio
from pathlib import Path
from typing import List

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame
from PIL import Image
import torch
import easyocr
import pdfplumber
from unstructured.partition.auto import partition

try:
    import fitz 
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False

from transformers import (
    CLIPModel, CLIPProcessor,
    BlipProcessor, BlipForConditionalGeneration,
    Pix2StructProcessor, Pix2StructForConditionalGeneration,
)

from .element import Outputs, Settings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

VISION_LABELS = [
    "chart", "diagram", "table", "screenshot", "document", "natural photo"
]
ALNUM_RE = re.compile(r"[A-Za-z0-9]")

# Models
CLIP_MODEL    = "openai/clip-vit-base-patch32"
BLIP_MODEL    = "Salesforce/blip-image-captioning-base"
CHARTQA_MODEL = "google/pix2struct-base"
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Lazy handles
clip_model = clip_proc = blip_model = blip_proc = p2s_model = p2s_proc = None

_easyocr_gpu = torch.cuda.is_available()
_ocr_langs   = [s.strip() for s in os.getenv("OCR_LANGS", "en").split(",") if s.strip()]
ocr_model    = easyocr.Reader(_ocr_langs, gpu=_easyocr_gpu, verbose=False)

# Helpers
def blip_caption(path: Path, device=None) -> str:
    img = Image.open(path).convert("RGB")
    inputs = blip_proc(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    blip_model.to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=40)
    return blip_proc.decode(out[0], skip_special_tokens=True)

def chartqa_caption(path: Path, device=None,
                    prompt: str = "Generate a CSV-like representation of the data in this chart:"):
    img = Image.open(path).convert("RGB")
    inputs = p2s_proc(images=img, text=prompt, return_tensors="pt").to(device or device)
    with torch.no_grad():
        out = p2s_model.generate(**inputs, max_new_tokens=128)
    return p2s_proc.decode(out[0], skip_special_tokens=True).strip()

def doctr_ocr(path: Path, conf_threshold: float = 0.30) -> str:
    try:
        results = ocr_model.readtext(str(path))
        parts = [text for (_bbox, text, conf) in results if (conf is None) or (conf >= conf_threshold)]
        return " ".join(parts)
    except Exception as e:
        print(f"EasyOCR failed on {path.name}: {e}")
        return ""
    
def ocr_pdf(path: Path):
    print(f"OCR {path.name}")
    elems = partition(filename=str(path), strategy="fast", languages=["eng"])
    return "\n".join(e.text for e in elems if e.text)

def safe_blip_caption(path: Path, enable: bool) -> str:
    if not enable:
        return ""
    try:
        return blip_caption(path, device=device)
    except Exception:
        return ""

def safe_chart_caption(path: Path, enable: bool) -> str:
    if not enable:
        return ""
    try:
        vtype = classify_visual_type(str(path))
        if vtype in {"chart", "diagram", "table"}:
            return chartqa_caption(path, device=device)
    except Exception:
        pass
    return ""

def safe_doctr_ocr(path: Path) -> str:
    try:
        return doctr_ocr(path)
    except Exception as e:
        return ""

def csv_to_sentences(raw_csv, hdr):
    out = []
    import csv, io
    for row in csv.reader(io.StringIO(raw_csv)):
        if row == hdr: continue
        out.append("Row -> " + ", ".join(f"{h.strip()}: {v.strip()}" for h, v in zip(hdr, row)))
    return out

def chunk_text(text: str, max_words: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += max_words - overlap
    return chunks

def tables_docs(path: Path):
    docs = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    for tbl in page.extract_tables():
                        hdr, *rows = tbl
                        hdr = ["" if c is None else str(c) for c in hdr]
                        md = "| " + " | ".join(hdr) + " |\n" + "|---" * len(hdr) + "|\n"
                        for r in rows:
                            md += "| " + " | ".join("" if c is None else str(c) for c in r) + " |\n"
                        docs.append({"page_content": md, "metadata": {"source": str(path), "page": page_num, "table_md": True}})
                        docs.append({"page_content": json.dumps({"headers": hdr, "rows": rows}, ensure_ascii=False),
                                     "metadata": {"source": str(path), "page": page_num, "table_json": True}})
                        raw_csv = "\n".join([",".join(hdr)] + [",".join("" if c is None else str(c) for c in r) for r in rows])
                        for sent in csv_to_sentences(raw_csv, hdr):
                            docs.append({"page_content": sent, "metadata": {"source": str(path), "page": page_num, "table_row": True}})
                except Exception as e:
                    print(f"Skipping table extraction on {path.name} page {page_num}: {e}")
    except Exception as e:
        print(f"Failed to open {path.name} with pdfplumber: {e}")
    return docs

def classify_visual_type(image_path: str) -> str:
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = clip_proc(text=[f"a {lbl}" for lbl in VISION_LABELS], images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            clip_model.to(device)
            image_emb = clip_model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
            text_emb  = clip_model.get_text_features(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
            image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
            text_emb  = torch.nn.functional.normalize(text_emb,  dim=-1)
            sims = (image_emb @ text_emb.T).squeeze(0)
            idx = int(torch.argmax(sims).cpu().item())
        return VISION_LABELS[idx]
    except Exception as e:
        return "natural photo"

def has_text(s: str, min_chars: int) -> bool:
    return bool(ALNUM_RE.search(s)) and (len(s.strip()) >= min_chars)

# Loaders
def load_docs(docs_dir: Path, img_store: Path | None, use_blip: bool, use_chartqa: bool, extract_images: bool):
    text_docs, img_docs = [], []

    for fp in docs_dir.rglob("*"):
        if fp.is_dir():
            continue
        mime = mimetypes.guess_type(fp)[0] or ""

        # Standalone images
        if mime.startswith("image"):
            img_docs.append({"page_content": "", "metadata": {"image_path": str(fp)}})
            cap   = safe_blip_caption(fp, enable=use_blip)
            chart = safe_chart_caption(fp, enable=use_chartqa)
            ocr   = safe_doctr_ocr(fp)
            combo = " • ".join(t for t in (cap, chart, ocr) if t)
            if combo:
                text_docs.append({
                    "page_content": combo,
                    "metadata": {
                        "image_path": str(fp),
                        "caption": bool(cap),
                        "chartqa": bool(chart),
                        "ocr": bool(ocr),
                        "source_type": "image_text"
                    }
                })

        # PDFs
        elif fp.suffix.lower() == ".pdf":
            try:
                with pdfplumber.open(str(fp)) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        txt = page.extract_text() or ""
                        if txt.strip():
                            for sub in chunk_text(txt, max_words=500, overlap=100):
                                text_docs.append({
                                "page_content": sub,
                                "metadata": {"source": str(fp), "page": page_num, "source_type": "text"}
                                })
            except Exception:
                text = ocr_pdf(fp)
                text_docs.append({
                    "page_content": text,
                    "metadata": {"source": str(fp), "page": None, "source_type": "text"}
                })

            # Tables
            for d in tables_docs(fp):
                d.setdefault("metadata", {}).setdefault("source_type", "text")
                text_docs.append(d)

            # Embedded images — now keep even if no OCR text (based on size/visual class)
            if img_store is not None and extract_images:
                extracted = extract_pdf_images(fp, img_store)
                img_docs.extend(extracted)

                for img_doc in extracted:
                    img_fp = Path(img_doc["metadata"]["image_path"])
                    cap   = safe_blip_caption(img_fp, enable=use_blip)
                    chart = safe_chart_caption(img_fp, enable=use_chartqa)
                    ocr   = safe_doctr_ocr(img_fp)

                    combo = " • ".join(t for t in (cap, chart, ocr) if t)
                    if combo:
                        meta = {**img_doc["metadata"], "caption": bool(cap),
                                "chartqa": bool(chart), "ocr": bool(ocr),
                                "source_type": "image_text"}
                        text_docs.append({
                            "page_content": combo,
                            "metadata": meta
                        })

                print(f"[OCR] image_store_folder -> {img_store}")

        # Plain text
        elif fp.suffix.lower() in {".txt", ".md"}:
            print(f"Load {fp.name}")
            text_docs.append({"page_content": fp.read_text(), "metadata": {"source": str(fp), "source_type": "text"}})

    print(f"Loaded {len(text_docs)} text docs & {len(img_docs)} images\n")
    return text_docs, img_docs

def extract_pdf_images(pdf_path: Path, out_dir: Path) -> list[dict]:
    out: list[dict] = []
    if not _HAVE_PYMUPDF:
        print(f"[OCR] PyMuPDF not available; skipping embedded images for {pdf_path.name}")
        return out

    MIN_OCR_CHARS = 6
    CONF_THRESHOLD = 0.30
    MIN_DIM = 48
    MIN_AREA = 16000  # ~126x126

    def _has_alnum_and_len(s: str) -> bool:
        s = (s or "").strip()
        return bool(ALNUM_RE.search(s)) and (len(s) >= MIN_OCR_CHARS)

    out_dir.mkdir(parents=True, exist_ok=True)

    import io, hashlib
    try:
        import numpy as _np
    except Exception:
        raise RuntimeError("numpy is required for OCR filtering of embedded images")

    doc = fitz.open(pdf_path)
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            images = page.get_images(full=True) or []
            for img_idx, img in enumerate(images):
                xref = img[0]
                rects = page.get_image_rects(xref) or [page.rect] 
                for r_idx, rect in enumerate(rects):
                    try:
                        mat = fitz.Matrix(2, 2)
                        pm = page.get_pixmap(clip=rect, matrix=mat, alpha=False, colorspace=fitz.csRGB)
                        png_bytes = pm.tobytes("png")
                        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    except Exception as e:
                        print(f"[OCR] rect render failed (xref={xref}, page={pno+1}): {e}")
                        continue

                    w, h = pil_img.size
                    area_ok = (min(w, h) >= MIN_DIM) and (w * h >= MIN_AREA)

                    # OCR text
                    try:
                        ocr_results = ocr_model.readtext(_np.array(pil_img))
                        parts = [t for (_bbox, t, conf) in ocr_results
                                 if (conf is None) or (conf >= CONF_THRESHOLD)]
                        ocr_text = " ".join(parts)
                    except Exception:
                        ocr_text = ""

                    keep = _has_alnum_and_len(ocr_text)
                    vision_label = None

                    # If no OCR text, keep if it's big or looks like a chart/diagram/table
                    if not keep and area_ok:
                        try:
                            vision_label = classify_visual_type(_save_temp(pil_img))
                        except Exception:
                            vision_label = None
                        if vision_label in {"chart","diagram","table"}:
                            keep = True

                    if not keep:
                        continue

                    name = hashlib.sha1(f"{pdf_path.name}:{pno}:{xref}:{r_idx}".encode()).hexdigest() + ".png"
                    fp = out_dir / name
                    pil_img.save(fp, format="PNG", optimize=True)

                    meta = {"image_path": str(fp), "source": str(pdf_path), "page": pno + 1}
                    if vision_label:
                        meta["vision_label"] = vision_label

                    out.append({"page_content": "", "metadata": meta})
    finally:
        doc.close()

    print(f"[OCR] {pdf_path.name}: saved {len(out)} images to {out_dir}")
    return out

# Classify without persisting temp to disk long-term
def _save_temp(pil_img: Image.Image) -> str:
    import tempfile
    p = Path(tempfile.mkstemp(suffix=".png")[1])
    pil_img.save(p, format="PNG")
    return str(p)

def caption_extracted_images(img_docs: list[dict]) -> list[dict]:
    text_entries: list[dict] = []
    for d in img_docs:
        try:
            ip = Path(d["metadata"]["image_path"])
            cap   = blip_caption(ip, device=device) if ip.exists() else ""
            chart = chartqa_caption(ip, device=device) if ip.exists() else ""
            ocr   = doctr_ocr(ip)
            combo = " • ".join(t for t in (cap, chart, ocr) if t)
            text_entries.append({
                "page_content": combo,
                "metadata": {
                    "image_path": str(ip),
                    "caption": bool(cap),
                    "chartqa": bool(chart),
                    "ocr": bool(ocr),
                    "source_type": "image_text",
                    **({k: v for k, v in d.get("metadata", {}).items() if k in ("source", "page","vision_label")})
                }
            })
        except Exception as e:
            print(f"Captioning extracted image failed: {e}")
    return text_entries

class OCRElement:
    async def run(self, process: Process):
        def _norm(s: str) -> str:
            return " ".join((s or "").split()).lower()

        def dedupe_text_docs(docs: list[dict], min_chars: int = 12) -> list[dict]:
            seen, out = set(), []
            for d in docs:
                txt = d.get("page_content") or ""
                if len(txt.strip()) < min_chars:
                    continue
                meta = d.get("metadata", {})
                anchor = meta.get("source") or meta.get("image_path") or "na"
                key = (anchor, meta.get("page"), hash(_norm(txt)))
                if key in seen:
                    continue
                seen.add(key)
                out.append(d)
            return out
        
        settings, outputs = process.settings, process.outputs

        data_path = settings.dataset_folder_path.value
        if not data_path or not Path(data_path).exists():
            raise Exception(f"Invalid dataset_folder_path: {data_path}")
        docs_dir = Path(data_path).resolve()

        img_store_val = getattr(settings, "image_store_folder", None)
        if not img_store_val or not img_store_val.value:
            raise Exception("image_store_folder must be set in element settings.")
        img_store = Path(img_store_val.value).expanduser().resolve()
        img_store.mkdir(parents=True, exist_ok=True)

        use_blip    = bool(getattr(settings, "use_blip", None) and settings.use_blip.value)
        use_chartqa = bool(getattr(settings, "use_chartqa", None) and settings.use_chartqa.value)
        extract_images = bool(getattr(settings, "extract_images", None) and settings.extract_images.value)

        global clip_model, clip_proc, blip_model, blip_proc, p2s_model, p2s_proc
        if extract_images or use_blip or use_chartqa:
            print("Loading image models…")
            clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
            clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=True)
            blip_proc  = BlipProcessor.from_pretrained(BLIP_MODEL)
            blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)
            p2s_proc   = Pix2StructProcessor.from_pretrained(CHARTQA_MODEL, torch_dtype=torch.float16)
            p2s_model  = Pix2StructForConditionalGeneration.from_pretrained(CHARTQA_MODEL, torch_dtype=torch.float16).to(device)
            print("Image models loaded\n")

        text_docs_all, img_docs = load_docs(docs_dir, img_store=img_store, use_blip=use_blip, use_chartqa=use_chartqa, extract_images=extract_images)

        # Split into pure text vs image-derived text
        text_docs = []
        image_text_docs = []
        for d in text_docs_all:
            if d.get("metadata", {}).get("source_type") == "image_text" or "image_path" in d.get("metadata", {}):
                image_text_docs.append(d)
            else:
                text_docs.append(d)

        # Dedupe
        text_docs       = dedupe_text_docs(text_docs)
        image_text_docs = dedupe_text_docs(image_text_docs)

        # Bundle (images are stored on disk, not passed downstream as binaries)
        fd, tmp_path = tempfile.mkstemp(prefix="ocr_bundle_", suffix=".json", dir="/tmp")
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "text_docs": text_docs,
                    "image_text_docs": image_text_docs,
                    "img_docs": img_docs,
                    "source_dir": str(docs_dir)
                },
                f,
                ensure_ascii=False
            )
        print(f"OCR wrote bundle: {tmp_path} (text + image-derived text)")

        await outputs.default.send(
            Frame(
                None, [], None, None, None,
                {"file_path": tmp_path}
            )
        )

        # Hack: Keep element running as if one element completes all will be killed by platform
        while True:
            await asyncio.sleep(1)


ocr = OCRElement()

process = CreateElement(Process(
    settings=Settings(),
    outputs=Outputs(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-ocr000000001",
        name="ocr",
        displayName="MM - OCR",
        version="0.53.0",
        description="Scans a folder, OCRs PDFs (and images), outputs path to JSON bundle with docs split by modality."
    ),
    run_func=ocr.run
))



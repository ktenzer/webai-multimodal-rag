import os, sys, time, textwrap, mimetypes, warnings, logging, csv, io, re, json, tempfile
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import Frame
import hashlib
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

# ---- original env/quieting bits (kept) ----
load_dotenv(override=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

VISION_LABELS = [
    "chart", "diagram", "table", "screenshot", "document", "natural photo"
]
ALNUM_RE = re.compile(r"[A-Za-z0-9]")

# ---- models per original ----
CLIP_MODEL    = "openai/clip-vit-base-patch32"
BLIP_MODEL    = "Salesforce/blip-image-captioning-base"
CHARTQA_MODEL = "google/deplot"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Loading OCR/caption models …")
clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
clip_proc  = CLIPProcessor.from_pretrained(CLIP_MODEL, use_fast=True)

blip_proc  = BlipProcessor.from_pretrained(BLIP_MODEL)
blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(device)

p2s_proc   = Pix2StructProcessor.from_pretrained(CHARTQA_MODEL)
p2s_model  = Pix2StructForConditionalGeneration.from_pretrained(CHARTQA_MODEL).to(device)

_easyocr_gpu = torch.cuda.is_available()
_ocr_langs   = [s.strip() for s in os.getenv("OCR_LANGS", "en").split(",") if s.strip()]
ocr_model    = easyocr.Reader(_ocr_langs, gpu=_easyocr_gpu, verbose=False)
print("Models ready\n")

# ---- helper functions from original ----
def blip_caption(path: Path, device=None) -> str:
    img = Image.open(path).convert("RGB")
    inputs = blip_proc(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    blip_model.to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=40)
    return blip_proc.decode(out[0], skip_special_tokens=True)

def chartqa_caption(path: Path, device=None,
                    prompt: str = "Generate the underlying data table of the figure below:"):
    """Return a deplot answer for chart-like images. Requires a text 'prompt'."""
    img = Image.open(path).convert("RGB")   # ensure 3 channels
    inputs = p2s_proc(images=img, text=prompt, return_tensors="pt").to(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
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
    # Use 'fast' instead of 'high_res' to avoid pdf2image/Poppler; keeps everything else the same.
    elems = partition(filename=str(path), strategy="fast", languages=["eng"])
    return "\n".join(e.text for e in elems if e.text)

def safe_blip_caption(path: Path, enable: bool) -> str:
    """BLIP caption guarded by a boolean toggle."""
    if not enable:
        return ""
    try:
        return blip_caption(path, device=device)
    except Exception:
        return ""

def safe_chart_caption(path: Path, enable: bool) -> str:
    """Run ChartQA only if enabled AND the image looks chart/diagram/table-like."""
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
        # print(f"OCR on image failed for {path.name}: {e}")
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
    """Return best label from _VISION_LABELS using CLIP zero-shot."""
    try:
        img = Image.open(image_path).convert("RGB")
        inputs = clip_proc(text=[f"a {lbl}" for lbl in VISION_LABELS], images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            clip_model.to(device)
            image_emb = clip_model.get_image_features(pixel_values=inputs["pixel_values"].to(device))
            text_emb  = clip_model.get_text_features(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
            # cosine similarity
            image_emb = torch.nn.functional.normalize(image_emb, dim=-1)
            text_emb  = torch.nn.functional.normalize(text_emb,  dim=-1)
            sims = (image_emb @ text_emb.T).squeeze(0)  # (num_labels,)
            idx = int(torch.argmax(sims).cpu().item())
        return VISION_LABELS[idx]
    except Exception as e:
        # If classification fails, return a safe default
        return "natural photo"

def has_text(s: str, min_chars: int) -> bool:
    return bool(ALNUM_RE.search(s)) and (len(s.strip()) >= min_chars)

def load_docs(docs_dir: Path, img_store: Path | None, use_blip: bool, use_chartqa: bool, extract_images: bool):
    """
    Builds:
      - text_docs: list of {"page_content": <text>, "metadata": {...}}
      - img_docs:  list of {"page_content": "", "metadata": {"image_path": ...}}
    We always create a text entry for images from OCR (+optional captions/chart).
    """
    text_docs, img_docs = [], []

    for fp in docs_dir.rglob("*"):
        if fp.is_dir():
            continue
        mime = mimetypes.guess_type(fp)[0] or ""

        # ---- Stand-alone image files ----
        if mime.startswith("image"):
            # keep image record (if you still want to archive), but we will not send images downstream
            img_docs.append({"page_content": "", "metadata": {"image_path": str(fp)}})

            # create the text doc for this image (caption + OCR + chart)
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
                    }
                })

        # ---- PDFs (extract text + tables + images) ----
        elif fp.suffix.lower() == ".pdf":
            try:
                with pdfplumber.open(str(fp)) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        txt = page.extract_text() or ""
                        if txt.strip():
                            for sub in chunk_text(txt, max_words=500, overlap=100):
                                text_docs.append({
                                "page_content": sub,
                                "metadata": {"source": str(fp), "page": page_num}
                                })
            except Exception:
                # fallback to single-chunk if pdfplumber fails
                text = ocr_pdf(fp)
                text_docs.append({
                    "page_content": text,
                    "metadata": {"source": str(fp), "page": None}
                })

            # tables (they already include 'page' in metadata)
            text_docs.extend(tables_docs(fp))

            # embedded images (existing logic unchanged)
            if img_store is not None:
                if extract_images and img_store is not None:
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
                                    "chartqa": bool(chart), "ocr": bool(ocr)}
                            text_docs.append({
                                "page_content": combo,
                                "metadata": meta
                            })

                print(f"[OCR] image_store_folder -> {img_store}")

        # ---- Plain text files ----
        elif fp.suffix.lower() in {".txt", ".md"}:
            print(f"Load {fp.name}")
            text_docs.append({"page_content": fp.read_text(), "metadata": {"source": str(fp)}})

    print(f"Loaded {len(text_docs)} text docs & {len(img_docs)} images\n")
    return text_docs, img_docs

def extract_pdf_images(pdf_path: Path, out_dir: Path) -> list[dict]:
    """
    Extract embedded images from a PDF, saving only those that contain usable
    alphanumeric text (via EasyOCR). Robust to missing / exotic colorspaces:
    renders the image region from the page into RGB first.

    Returns: [{ "page_content": "", "metadata": { "image_path", "source", "page" } }]
    """
    out: list[dict] = []
    if not _HAVE_PYMUPDF:
        print(f"[OCR] PyMuPDF not available; skipping embedded images for {pdf_path.name}")
        return out

    # strict keep policy
    MIN_OCR_CHARS = 6
    CONF_THRESHOLD = 0.30

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
    kept = skipped = 0
    try:
        for pno in range(len(doc)):
            page = doc[pno]
            images = page.get_images(full=True)
            if not images:
                continue

            for img_idx, img in enumerate(images):
                xref = img[0]

                # Prefer rendering the exact image rect(s) from the page (always RGB).
                rects = page.get_image_rects(xref)
                if not rects:
                    rects = []

                # we may have multiple rects reusing the same xref; iterate them
                used_any_rect = False
                for r_idx, rect in enumerate(rects):
                    try:
                        mat = fitz.Matrix(2, 2)  # 2x scale for better OCR; adjust if needed
                        pm = page.get_pixmap(clip=rect, matrix=mat, alpha=False, colorspace=fitz.csRGB)
                        png_bytes = pm.tobytes("png")
                        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    except Exception as e:
                        # If rect render somehow fails, try next rect or fall back later
                        print(f"[OCR] rect render failed (xref={xref}, page={pno+1}): {e}")
                        continue

                    used_any_rect = True

                    # Skip tiny assets / icons quickly
                    if min(pil_img.size) < 16 or (pil_img.size[0] * pil_img.size[1]) < 400:
                        skipped += 1
                        continue

                    # OCR to decide keep/skip
                    try:
                        ocr_results = ocr_model.readtext(_np.array(pil_img))
                        parts = [t for (_bbox, t, conf) in ocr_results
                                 if (conf is None) or (conf >= CONF_THRESHOLD)]
                        ocr_text = " ".join(parts)
                    except Exception:
                        ocr_text = ""

                    if not _has_alnum_and_len(ocr_text):
                        skipped += 1
                        continue  # DO NOT SAVE

                    # Save passing images (RGB PNG via PIL)
                    name = hashlib.sha1(f"{pdf_path.name}:{pno}:{xref}:{r_idx}".encode()).hexdigest() + ".png"
                    fp = out_dir / name
                    pil_img.save(fp, format="PNG", optimize=True)
                    kept += 1
                    print(f"[OCR] saved embedded image (has text) -> {fp}")

                    out.append({
                        "page_content": "",
                        "metadata": {
                            "image_path": str(fp),
                            "source": str(pdf_path),
                            "page": pno + 1,
                        },
                    })

                if used_any_rect:
                    # We already handled all rects for this xref; next image
                    continue

                # ---- Fallback: no rects (rare). Render the whole page and crop the bbox of the xref if possible,
                # or finally try to export the raw pixmap and PIL-convert it.
                try:
                    # Try raw export -> PIL as a last resort (can still fail; catch)
                    pix = fitz.Pixmap(doc, xref)
                    try:
                        png_bytes = pix.tobytes("png")  # may raise unsupported colorspace
                        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                    except Exception:
                        # Rasterize the whole page and hope the image occupies most of it
                        mat = fitz.Matrix(2, 2)
                        pm = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
                        png_bytes = pm.tobytes("png")
                        pil_img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
                except Exception as e:
                    print(f"[OCR] fallback render failed (xref={xref}, page={pno+1}): {e}")
                    continue

                # Skip tiny assets
                if min(pil_img.size) < 16 or (pil_img.size[0] * pil_img.size[1]) < 400:
                    skipped += 1
                    continue

                # OCR + keep
                try:
                    ocr_results = ocr_model.readtext(_np.array(pil_img))
                    parts = [t for (_bbox, t, conf) in ocr_results
                             if (conf is None) or (conf >= CONF_THRESHOLD)]
                    ocr_text = " ".join(parts)
                except Exception:
                    ocr_text = ""

                if not _has_alnum_and_len(ocr_text):
                    skipped += 1
                    continue

                name = hashlib.sha1(f"{pdf_path.name}:{pno}:{xref}:fb".encode()).hexdigest() + ".png"
                fp = out_dir / name
                pil_img.save(fp, format="PNG", optimize=True)
                kept += 1
                print(f"[OCR] saved embedded image (fallback, has text) -> {fp}")

                out.append({
                    "page_content": "",
                    "metadata": {
                        "image_path": str(fp),
                        "source": str(pdf_path),
                        "page": pno + 1,
                    },
                })

    finally:
        doc.close()

    print(f"[OCR] {pdf_path.name}: kept {kept} / skipped {skipped} (saved to {out_dir})")
    return out

def caption_extracted_images(img_docs: list[dict]) -> list[dict]:
    """
    For each extracted image, generate a text entry (caption/chartqa/ocr) so it is
    searchable by text retrieval, mirroring the behavior for 'native' images.
    Returns a list of text-doc dicts in the same shape as other text_docs.
    """
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
                    "caption": True,
                    "chartqa": bool(chart),
                    "ocr": bool(ocr),
                    # also preserve originating PDF context if present:
                    **({k: v for k, v in d.get("metadata", {}).items() if k in ("source", "page")})
                }
            })
        except Exception as e:
            print(f"Captioning extracted image failed: {e}")
    return text_entries

class OCRElement:
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

        # Input documents directory
        data_path = settings.dataset_folder_path.value
        if not data_path or not Path(data_path).exists():
            raise Exception(f"Invalid dataset_folder_path: {data_path}")
        docs_dir = Path(data_path).resolve()

        # Where to store extracted/normalized images
        img_store_val = getattr(settings, "image_store_folder", None)
        if not img_store_val or not img_store_val.value:
            raise Exception("image_store_folder must be set in element settings.")
        img_store = Path(img_store_val.value).expanduser().resolve()
        img_store.mkdir(parents=True, exist_ok=True)

        # NEW: read toggles from element settings
        use_blip    = bool(getattr(settings, "use_blip", None) and settings.use_blip.value)
        use_chartqa = bool(getattr(settings, "use_chartqa", None) and settings.use_chartqa.value)
        extract_images = bool(getattr(settings, "extract_images", None) and settings.extract_images.value)

        # Build text_docs and img_docs
        text_docs, img_docs = load_docs(docs_dir, img_store=img_store, use_blip=use_blip, use_chartqa=use_chartqa, extract_images=extract_images)

        # Optional: dedupe text
        text_docs = dedupe_text_docs(text_docs)

        # Write tmp json bundle — IMPORTANT: do not pass images downstream
        fd, tmp_path = tempfile.mkstemp(prefix="ocr_bundle_", suffix=".json", dir="/tmp")
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {"text_docs": text_docs, "img_docs": [], "source_dir": str(docs_dir)},
                f,
                ensure_ascii=False
            )
        print(f"OCR wrote bundle: {tmp_path} (images omitted; text-only indexing)")

        # Emit path via Frame.other_data
        await outputs.default.send(
            Frame(
                None, [], None, None, None,
                {"file_path": tmp_path}
            )
        )
ocr = OCRElement()

process = CreateElement(Process(
    settings=Settings(),
    outputs=Outputs(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-ocr000000001",
        name="ocr",
        displayName="MM - OCR Element",
        version="0.41.0",
        description="Scans a folder, OCRs PDFs (and images), outputs path to JSON bundle with docs."
    ),
    run_func=ocr.run
))

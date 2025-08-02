import os, sys, time, textwrap, mimetypes, warnings, logging, csv, io, re, json, tempfile
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from webai_element_sdk.element import CreateElement
from webai_element_sdk.process import Process, ProcessMetadata
from webai_element_sdk.comms.messages import TextFrame

from PIL import Image
import torch
import easyocr
import pdfplumber
from unstructured.partition.auto import partition

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

def chartqa_caption(path: Path, device=None) -> str:
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    img = Image.open(path).convert("RGB")
    inputs = p2s_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = p2s_model.generate(**inputs, max_new_tokens=64)
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

def csv_to_sentences(raw_csv, hdr):
    out = []
    import csv, io
    for row in csv.reader(io.StringIO(raw_csv)):
        if row == hdr: continue
        out.append("Row -> " + ", ".join(f"{h.strip()}: {v.strip()}" for h, v in zip(hdr, row)))
    return out

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

def load_docs(docs_dir: Path):
    text, images = [], []
    for fp in docs_dir.rglob("*"):
        if fp.is_dir(): continue
        mime = mimetypes.guess_type(fp)[0] or ""
        if mime.startswith("image"):
            images.append({"page_content": "", "metadata": {"image_path": str(fp)}})
            cap   = blip_caption(fp)
            chart = chartqa_caption(fp)
            ocr   = doctr_ocr(fp)
            combo = " • ".join(t for t in (cap, chart, ocr) if t)
            text.append({"page_content": combo,
                         "metadata": {"image_path": str(fp), "caption": True, "chartqa": bool(chart), "ocr": bool(ocr)}})
        elif fp.suffix.lower() == ".pdf":
            text.append({"page_content": ocr_pdf(fp), "metadata": {"source": str(fp)}})
            text.extend(tables_docs(fp))
        elif fp.suffix.lower() in {".txt", ".md"}:
            print(f"Load {fp.name}")
            text.append({"page_content": fp.read_text(), "metadata": {"source": str(fp)}})
    print(f"Loaded {len(text)} text docs & {len(images)} images\n")
    return text, images

class OCRElement:
    async def run(self, process: Process):
        settings, outputs = process.settings, process.outputs
        data_path = settings.dataset_folder_path.value
        if not data_path or not Path(data_path).exists():
            raise Exception(f"Invalid dataset_folder_path: {data_path}")
        docs_dir = Path(data_path).resolve()
        text_docs, img_docs = load_docs(docs_dir)

        # write tmp json bundle
        fd, tmp_path = tempfile.mkstemp(prefix="ocr_bundle_", suffix=".json", dir="/tmp")
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump({"text_docs": text_docs, "img_docs": img_docs, "source_dir": str(docs_dir)}, f, ensure_ascii=False)
        print(f"OCR wrote bundle: {tmp_path}")
        await outputs.default.send(TextFrame(text=tmp_path)) 

ocr = OCRElement()

process = CreateElement(Process(
    settings=Settings(),
    outputs=Outputs(),
    metadata=ProcessMetadata(
        id="2a7a0b6a-7b84-4c57-8f1c-ocr000000001",
        name="ocr",
        displayName="MM - OCR Element",
        version="0.18.0",
        description="Scans a folder, OCRs PDFs (and images), outputs path to JSON bundle with docs."
    ),
    run_func=ocr.run
))

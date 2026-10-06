"""Agent 1 — Document Ingestion & Routing.

Looks at what arrived and sends it to the engine that can read it. Nothing
here is clever; it is the part of the system that has to be boring, because
every later agent assumes the text in front of it is really the document's
text.

Three roads:

  PDF    -> PyMuPDF reads the text layer directly. A scanned PDF has no text
            layer, so each page is rendered and sent down the OCR road.
  IMAGE  -> Tesseract. Present in this project's packages.txt so it exists on
            Streamlit Cloud too; if it is missing the file is still accepted
            and the failure is reported honestly rather than faked.
  TABLE  -> pandas, for CSV and Excel. No OCR, no language model, no guessing.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

logger = logging.getLogger("omniparse.ingest")

PDF_TYPES = (".pdf",)
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
TABLE_TYPES = (".csv", ".tsv", ".xlsx", ".xls")

SUPPORTED = PDF_TYPES + IMAGE_TYPES + TABLE_TYPES


def route(filename: str) -> str:
    """The Document Type Router. Returns 'pdf', 'image', 'table' or ''."""
    lowered = (filename or "").lower()
    if lowered.endswith(PDF_TYPES):
        return "pdf"
    if lowered.endswith(IMAGE_TYPES):
        return "image"
    if lowered.endswith(TABLE_TYPES):
        return "table"
    return ""


# ---------------------------------------------------------------------------
def _ocr(image) -> str:
    try:
        import pytesseract
    except ImportError:
        raise RuntimeError(
            "OCR engine (Tesseract) is not installed in this environment. "
            "Text PDFs, CSV and Excel files still work.")
    try:
        return pytesseract.image_to_string(image)
    except Exception as exc:   # noqa: BLE001 — tesseract binary missing
        raise RuntimeError(f"OCR failed: {exc}") from exc


def read_pdf(data: bytes) -> tuple:
    """Return (text, method). Falls back to OCR when there is no text layer."""
    import pymupdf

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)
        if len(text.strip()) >= 40:
            return text, "pdf-text-layer"

        # A scan. Render each page and read it the slow way.
        from PIL import Image

        pages = []
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            pages.append(_ocr(Image.open(io.BytesIO(pix.tobytes("png")))))
        return "\n".join(pages), "pdf-ocr"


def read_image(data: bytes) -> tuple:
    from PIL import Image

    return _ocr(Image.open(io.BytesIO(data))), "ocr"


def read_table(data: bytes, filename: str) -> tuple:
    """Return (DataFrame, method)."""
    lowered = filename.lower()
    if lowered.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data)), "pandas-excel"
    sep = "\t" if lowered.endswith(".tsv") else ","
    return pd.read_csv(io.BytesIO(data), sep=sep), "pandas-csv"


# ---------------------------------------------------------------------------
def ingest(data: bytes, filename: str) -> dict:
    """One file in, one payload out.

    {"kind": "pdf"|"image"|"table", "text": str|None, "frame": DataFrame|None,
     "method": str}
    """
    kind = route(filename)
    if not kind:
        raise ValueError(
            f"{filename}: unsupported file type. "
            f"Accepted: {', '.join(SUPPORTED)}")

    if kind == "table":
        frame, method = read_table(data, filename)
        return {"kind": kind, "text": None, "frame": frame, "method": method}

    text, method = read_pdf(data) if kind == "pdf" else read_image(data)
    if not text.strip():
        raise ValueError(
            f"{filename}: no readable text was found in this document.")
    return {"kind": kind, "text": text, "frame": None, "method": method}

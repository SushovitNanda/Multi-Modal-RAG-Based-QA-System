"""
Document ingestion utilities for PDFs, tables, and embedded images with OCR.
"""

from __future__ import annotations

import os
import sys
import warnings
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import fitz
import pdfplumber
from PIL import Image
import pytesseract
import camelot

from langchain_community.document_loaders import PyMuPDFLoader

# Suppress non-critical PDF processing warnings
warnings.filterwarnings("ignore", message=".*Cannot set gray non-stroke color.*")
warnings.filterwarnings("ignore", message=".*invalid float value.*")
warnings.filterwarnings("ignore", message=".*does not lie in column range.*")
warnings.filterwarnings("ignore", category=UserWarning, module="camelot")
# Suppress all warnings from PyMuPDF/fitz
warnings.filterwarnings("ignore", category=UserWarning, module="fitz")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="fitz")
# Suppress warnings from langchain PyMuPDFLoader
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="langchain")


def extract_text_with_langchain(pdf_path: str) -> List[Dict[str, Any]]:
    """Use LangChain's PyMuPDFLoader to grab page-wise text."""
    try:
        # Redirect stderr to suppress C-level warnings from PyMuPDF
        stderr_buffer = StringIO()
        with warnings.catch_warnings(), redirect_stderr(stderr_buffer):
            # Suppress all warnings during PDF loading
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", message=".*")
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
        page_texts = []
        for d in docs:
            page_num = d.metadata.get("page")
            page_texts.append({"page": page_num, "text": d.page_content, "source": pdf_path})
        return page_texts
    except Exception as e:
        # Return empty list if extraction fails, will fall back to pdfplumber
        return []


def extract_tables_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract tabular data via Camelot (tries lattice, then stream)."""
    def _read(flavor: str):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="camelot")
                warnings.filterwarnings("ignore", message=".*does not lie in column range.*")
                return camelot.read_pdf(pdf_path, pages="all", flavor=flavor)
        except Exception:
            return []

    tables = _read("lattice")
    if not tables:
        tables = _read("stream")

    records: List[Dict[str, Any]] = []
    for table in tables:
        try:
            df = table.df
            records.append(
                {
                    "page": int(table.page),
                    "table_text": df.to_csv(index=False),
                    "source": pdf_path,
                }
            )
        except Exception:
            continue
    return records


def extract_images_and_ocr(pdf_path: str, output_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract embedded images using PyMuPDF and run pytesseract OCR.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images: List[Dict[str, Any]] = []
    
    try:
        # Redirect stderr to suppress C-level warnings from PyMuPDF
        stderr_buffer = StringIO()
        with warnings.catch_warnings(), redirect_stderr(stderr_buffer):
            # Suppress all warnings during PDF processing
            warnings.simplefilter("ignore")
            warnings.filterwarnings("ignore", message=".*")
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pdf = fitz.open(pdf_path)
            
            for page_index in range(len(pdf)):
                try:
                    page = pdf[page_index]
                    for img_index, img in enumerate(page.get_images(full=True)):
                        try:
                            xref = img[0]
                            base_image = pdf.extract_image(xref)
                            img_bytes = base_image["image"]
                            ext = base_image.get("ext", "png")
                            img_name = f"{Path(pdf_path).stem}_p{page_index+1}_img{img_index+1}.{ext}"
                            img_path = output_dir / img_name
                            with img_path.open("wb") as f:
                                f.write(img_bytes)
                            try:
                                pil_img = Image.open(img_path).convert("RGB")
                                ocr_text = pytesseract.image_to_string(pil_img)
                            except Exception:
                                ocr_text = ""
                            images.append(
                                {
                                    "page": page_index + 1,
                                    "image_path": str(img_path),
                                    "ocr_text": ocr_text,
                                    "source": pdf_path,
                                }
                            )
                        except Exception:
                            # Skip individual image extraction errors
                            continue
                except Exception:
                    # Skip page-level errors
                    continue
            pdf.close()
    except Exception:
        # Return empty list if PDF cannot be opened
        pass
    
    return images


def extract_chart_metadata_from_page_text(text: str) -> Dict[str, str]:
    """Simple heuristic to capture chart / figure metadata clues."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    captions = []
    for line in lines[:50]:
        if line.lower().startswith(("figure", "fig.", "chart", "table")):
            captions.append(line)
    for line in lines:
        lower = line.lower()
        if any(key in lower for key in ["axis", "legend", "scale", "units", "x-axis", "y-axis"]):
            captions.append(line)
    return {"captions": " | ".join(captions)}


def ingest_documents(raw_dir: Path, processed_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run ingestion over PDFs in raw_dir and return dict containing page_texts, tables, images.
    """
    files = [str(p) for p in raw_dir.glob("*") if p.suffix.lower() in [".pdf", ".docx"]]
    page_texts_all: List[Dict[str, Any]] = []
    table_records_all: List[Dict[str, Any]] = []
    image_records_all: List[Dict[str, Any]] = []

    for fpath in files:
        try:
            # Redirect stderr to suppress C-level warnings from PyMuPDF
            stderr_buffer = StringIO()
            with warnings.catch_warnings(), redirect_stderr(stderr_buffer):
                # Suppress all warnings during PDF processing
                warnings.simplefilter("ignore")
                warnings.filterwarnings("ignore", message=".*")
                page_texts = extract_text_with_langchain(fpath)
        except Exception:
            page_texts = []
            try:
                # Redirect stderr to suppress C-level warnings from pdfplumber
                stderr_buffer = StringIO()
                with warnings.catch_warnings(), redirect_stderr(stderr_buffer):
                    # Suppress all warnings during PDF processing
                    warnings.simplefilter("ignore")
                    warnings.filterwarnings("ignore", message=".*")
                    with pdfplumber.open(fpath) as pdf:
                        for i, page in enumerate(pdf.pages):
                            page_texts.append({"page": i + 1, "text": page.extract_text() or "", "source": fpath})
            except Exception:
                page_texts = []

        page_texts_all.extend(page_texts)
        table_records_all.extend(extract_tables_pdf(fpath))
        image_records_all.extend(extract_images_and_ocr(fpath, processed_dir / "images"))

    return {
        "pages": page_texts_all,
        "tables": table_records_all,
        "images": image_records_all,
    }



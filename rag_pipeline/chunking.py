"""
Chunking utilities to create LangChain-friendly segments with metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def chunk_documents(
    page_records: List[Dict[str, Any]],
    table_records: List[Dict[str, Any]],
    image_records: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    docs_for_split: List[str] = []
    meta_map: List[Dict[str, Any]] = []

    def _append(records: List[Dict[str, Any]], rec_type: str):
        for rec in records:
            content = ""
            if rec_type == "page_text":
                content = clean_text(rec.get("text"))
            elif rec_type == "table":
                content = clean_text(rec.get("table_text"))
            elif rec_type == "image_ocr":
                content = clean_text(rec.get("ocr_text"))
            if not content:
                continue
            docs_for_split.append(content)
            metadata = {"type": rec_type, "page": rec.get("page"), "source": rec.get("source")}
            if rec_type == "image_ocr":
                metadata["image_path"] = rec.get("image_path")
            meta_map.append(metadata)

    _append(page_records, "page_text")
    _append(table_records, "table")
    _append(image_records, "image_ocr")

    all_chunks: List[Dict[str, Any]] = []
    chunk_id = 0
    for idx, full_text in enumerate(docs_for_split):
        dummy_doc = type("D", (), {})()
        dummy_doc.page_content = full_text
        dummy_doc.metadata = meta_map[idx]
        splitted = splitter.split_documents([dummy_doc])
        for s in splitted:
            all_chunks.append(
                {
                    "id": f"chunk_{chunk_id}",
                    "type": s.metadata.get("type"),
                    "page": s.metadata.get("page"),
                    "content": s.page_content,
                    "metadata": s.metadata,
                }
            )
            chunk_id += 1

    return all_chunks



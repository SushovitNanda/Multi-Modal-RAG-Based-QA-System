"""
Top-level orchestration for ingestion, chunking, embeddings, and artifact loading.
"""

from __future__ import annotations

import json
import warnings
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from typing import List, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Suppress PDF processing warnings
warnings.filterwarnings("ignore", message=".*Cannot set gray non-stroke color.*")
warnings.filterwarnings("ignore", message=".*invalid float value.*")

from . import config
from .chunking import chunk_documents
from .embeddings import (
    build_sbert_and_faiss,
    build_tfidf,
    get_avg_word2vec_embeddings,
    load_pickle,
    load_word2vec_model,
    persist_doc_texts,
    save_pickle,
)
from .ingestion import ingest_documents
from .retrieval import RetrievalArtifacts


def _save_chunks(chunks: List[dict]) -> None:
    try:
        config.PROCESSED_JSON.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    except (IOError, OSError) as e:
        print(f"Warning: Could not save chunks to {config.PROCESSED_JSON}: {e}")
        raise


def _load_chunks() -> Optional[List[dict]]:
    if not config.PROCESSED_JSON.exists():
        return None
    try:
        return json.loads(config.PROCESSED_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError, OSError) as e:
        print(f"Warning: Could not load chunks from {config.PROCESSED_JSON}: {e}")
        return None


def load_existing_artifacts() -> Optional[RetrievalArtifacts]:
    required = [
        config.PROCESSED_JSON,
        config.FAISS_INDEX_DIR / "index.faiss",
        config.FAISS_INDEX_DIR / "doc_embs.npy",
        config.TFIDF_MODEL_PATH,
        config.TFIDF_MATRIX_PATH,
    ]
    if not all(path.exists() for path in required):
        return None

    chunks = _load_chunks()
    if chunks is None:
        return None

    try:
        tfidf_vectorizer = load_pickle(config.TFIDF_MODEL_PATH)
    except Exception:
        return None
    
    tfidf_matrix = None
    try:
        from scipy import sparse

        tfidf_matrix = sparse.load_npz(config.TFIDF_MATRIX_PATH)
    except Exception:
        return None

    w2v_doc_embs = np.load(config.W2V_EMB_PATH) if config.W2V_EMB_PATH.exists() else None
    w2v_model = load_word2vec_model(config.WORD2VEC_NAME) if config.W2V_EMB_PATH.exists() else None
    sbert_doc_embs = np.load(config.FAISS_INDEX_DIR / "doc_embs.npy")
    faiss_index = faiss.read_index(str(config.FAISS_INDEX_DIR / "index.faiss"))
    sbert_model = SentenceTransformer(config.SBERT_MODEL_NAME, device=config.DEVICE)
    
    return RetrievalArtifacts(
        chunks=chunks,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        w2v_model=w2v_model,
        w2v_doc_embs=w2v_doc_embs,
        sbert_model=sbert_model,
        sbert_doc_embs=sbert_doc_embs,
        faiss_index=faiss_index,
    )


def build_pipeline_and_index(rebuild: bool = False) -> RetrievalArtifacts:
    if not rebuild:
        existing = load_existing_artifacts()
        if existing:
            return existing

    # Suppress PDF warnings during ingestion
    stderr_buffer = StringIO()
    with warnings.catch_warnings(), redirect_stderr(stderr_buffer):
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore", message=".*")
        ingestion_outputs = ingest_documents(config.RAW_DOC_DIR, config.PROCESSED_DIR)
    chunks = chunk_documents(
        ingestion_outputs["pages"],
        ingestion_outputs["tables"],
        ingestion_outputs["images"],
        config.CHUNK_SIZE,
        config.CHUNK_OVERLAP,
    )
    
    # Handle empty chunks
    if not chunks or len(chunks) == 0:
        raise ValueError("No documents were processed. Please ensure PDF files exist in the data/raw directory.")
    
    _save_chunks(chunks)

    doc_texts = [chunk["content"] for chunk in chunks]
    persist_doc_texts(doc_texts)

    tfidf_vectorizer, tfidf_matrix = build_tfidf(doc_texts)
    save_pickle(tfidf_vectorizer, config.TFIDF_MODEL_PATH)
    from scipy import sparse

    sparse.save_npz(config.TFIDF_MATRIX_PATH, tfidf_matrix)

    w2v_model = load_word2vec_model(config.WORD2VEC_NAME)
    if w2v_model is not None:
        w2v_doc_embs = get_avg_word2vec_embeddings(w2v_model, doc_texts)
        np.save(config.W2V_EMB_PATH, w2v_doc_embs)
    else:
        w2v_doc_embs = None

    sbert_model, faiss_index, sbert_doc_embs = build_sbert_and_faiss(
        doc_texts, model_name=config.SBERT_MODEL_NAME, faiss_index_path=config.FAISS_INDEX_DIR
    )

    return RetrievalArtifacts(
        chunks=chunks,
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        w2v_model=w2v_model,
        w2v_doc_embs=w2v_doc_embs,
        sbert_model=sbert_model,
        sbert_doc_embs=sbert_doc_embs,
        faiss_index=faiss_index,
    )



"""
Embedding builders and persistence helpers.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import scipy.sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as gensim_api
import faiss

from . import config


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (IOError, OSError, pickle.PickleError) as e:
        print(f"Warning: Could not load pickle from {path}: {e}")
        raise


def build_tfidf(doc_texts: List[str]) -> Tuple[TfidfVectorizer, Any]:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50000)
    matrix = tfidf.fit_transform(doc_texts)
    return tfidf, matrix


def load_word2vec_model(name: str = config.WORD2VEC_NAME):
    try:
        return gensim_api.load(name)
    except Exception:
        return None


def get_avg_word2vec_embeddings(w2v_model, texts: List[str]) -> np.ndarray:
    if w2v_model is None:
        return np.zeros((len(texts), 300))
    embs = []
    for text in texts:
        tokens = [w for w in text.split() if w]
        vectors = []
        for token in tokens:
            try:
                vectors.append(w2v_model[token.lower()])
            except KeyError:
                continue
        if vectors:
            embs.append(np.mean(vectors, axis=0))
        else:
            embs.append(np.zeros(w2v_model.vector_size))
    return np.vstack(embs)


def build_sbert_and_faiss(
    texts: List[str],
    model_name: str = config.SBERT_MODEL_NAME,
    faiss_index_path: Path = config.FAISS_INDEX_DIR,
):
    sbert = SentenceTransformer(model_name, device=config.DEVICE)
    doc_embs = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = doc_embs.shape[1]
    faiss.normalize_L2(doc_embs)
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs)
    faiss_index_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_index_path / "index.faiss"))
    np.save(faiss_index_path / "doc_embs.npy", doc_embs)
    return sbert, index, doc_embs


def persist_doc_texts(doc_texts: List[str]) -> None:
    try:
        config.DOC_TEXTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with config.DOC_TEXTS_PATH.open("w", encoding="utf-8") as f:
            json.dump(doc_texts, f, indent=2)
    except (IOError, OSError) as e:
        print(f"Warning: Could not save doc texts to {config.DOC_TEXTS_PATH}: {e}")
        raise


def cosine_sim_matrix(vectorizer: TfidfVectorizer, tfidf_matrix, query: str) -> np.ndarray:
    q_vec = vectorizer.transform([query])
    return cosine_similarity(q_vec, tfidf_matrix).reshape(-1)



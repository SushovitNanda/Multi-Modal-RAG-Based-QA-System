"""
Hybrid retrieval, reranking, and helper utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

from . import config
from .embeddings import cosine_sim_matrix, get_avg_word2vec_embeddings


@dataclass
class RetrievalArtifacts:
    chunks: List[Dict]
    tfidf_vectorizer: any
    tfidf_matrix: any
    w2v_model: any
    w2v_doc_embs: np.ndarray | None
    sbert_model: SentenceTransformer
    sbert_doc_embs: np.ndarray
    faiss_index: faiss.Index


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.max() - scores.min() <= 1e-9:
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def reciprocal_rank_fusion(rankings: List[np.ndarray], k: int = 60) -> np.ndarray:
    """
    Reciprocal Rank Fusion (RRF) combines multiple ranked lists.
    
    Args:
        rankings: List of arrays where each array contains document indices in ranked order
        k: RRF constant (typical values: 20-100, default 60)
    
    Returns:
        Combined RRF scores for all documents
    """
    if not rankings:
        return np.array([])
    
    # Filter out empty rankings and get max doc index
    valid_rankings = [rank for rank in rankings if len(rank) > 0]
    if not valid_rankings:
        return np.array([])
    
    n_docs = max(max(rank) for rank in valid_rankings) + 1
    rrf_scores = np.zeros(n_docs)
    
    for ranking in valid_rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            if 0 <= doc_idx < n_docs:
                rrf_scores[doc_idx] += 1.0 / (k + rank)
    
    return rrf_scores


def hybrid_retrieval(
    query: str,
    artifacts: RetrievalArtifacts,
    top_k: int = config.TOP_K_CANDIDATES,
    use_rrf: bool = config.USE_RRF,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Hybrid retrieval with optional RRF and cross-modal support.
    """
    # Handle empty query
    if not query or not query.strip():
        n_docs = len(artifacts.chunks)
        return np.array([], dtype=int), np.array([]), {
            "tfidf": np.zeros(n_docs) if n_docs > 0 else np.array([]),
            "w2v": np.zeros(n_docs) if n_docs > 0 else np.array([]),
            "sbert": np.zeros(n_docs) if n_docs > 0 else np.array([]),
            "clip": np.zeros(n_docs) if n_docs > 0 else np.array([]),
            "fused": np.zeros(n_docs) if n_docs > 0 else np.array([]),
        }
    
    n_docs = len(artifacts.chunks)
    
    # Handle empty document collection
    if n_docs == 0:
        return np.array([], dtype=int), np.array([]), {
            "tfidf": np.array([]),
            "w2v": np.array([]),
            "sbert": np.array([]),
            "clip": np.array([]),
            "fused": np.array([]),
        }
    
    # 1. TF-IDF retrieval
    tfidf_scores = cosine_sim_matrix(artifacts.tfidf_vectorizer, artifacts.tfidf_matrix, query)
    tfidf_ranking = np.argsort(tfidf_scores)[::-1]
    
    # 2. Word2Vec retrieval
    if artifacts.w2v_model is not None and artifacts.w2v_doc_embs is not None:
        q_w2v = get_avg_word2vec_embeddings(artifacts.w2v_model, [query])[0]
        q_norm = np.linalg.norm(q_w2v)
        doc_norms = np.linalg.norm(artifacts.w2v_doc_embs, axis=1)

        w2v_scores = np.zeros(n_docs)
        valid_mask = (doc_norms > 0) & (q_norm > 0)
        if valid_mask.any():
            dot_products = artifacts.w2v_doc_embs[valid_mask] @ q_w2v
            w2v_scores[valid_mask] = dot_products / (doc_norms[valid_mask] * q_norm)
        w2v_ranking = np.argsort(w2v_scores)[::-1]
    else:
        w2v_scores = np.zeros(n_docs)
        w2v_ranking = np.arange(n_docs)

    # 3. SBERT retrieval
    q_sbert = artifacts.sbert_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_sbert)
    sbert_scores = np.dot(artifacts.sbert_doc_embs, q_sbert.reshape(-1))
    sbert_ranking = np.argsort(sbert_scores)[::-1]

    # Normalize component scores for display
    tfidf_norm = _normalize(tfidf_scores)
    w2v_norm = _normalize(w2v_scores)
    sbert_norm = _normalize(sbert_scores)

    # Choose fusion method: RRF or weighted sum
    if use_rrf:
        # Reciprocal Rank Fusion
        rankings = [tfidf_ranking, w2v_ranking, sbert_ranking]
        fused = reciprocal_rank_fusion(rankings, k=config.RRF_K)
        # If RRF returns empty array (shouldn't happen), fall back to weighted sum
        if len(fused) == 0:
            fused = (
                config.HYBRID_WEIGHTS.tfidf * tfidf_norm
                + config.HYBRID_WEIGHTS.word2vec * w2v_norm
                + config.HYBRID_WEIGHTS.sbert * sbert_norm
            )
    else:
        # Weighted sum (original method)
        fused = (
            config.HYBRID_WEIGHTS.tfidf * tfidf_norm
            + config.HYBRID_WEIGHTS.word2vec * w2v_norm
            + config.HYBRID_WEIGHTS.sbert * sbert_norm
        )

    if len(fused) == 0:
        # Return empty results if no documents
        return np.array([], dtype=int), np.array([]), {
            "tfidf": np.zeros(n_docs) if n_docs > 0 else np.array([]),
            "w2v": np.zeros(n_docs) if n_docs > 0 else np.array([]),
            "sbert": np.zeros(n_docs) if n_docs > 0 else np.array([]),
            "fused": np.zeros(n_docs) if n_docs > 0 else np.array([]),
        }
    
    top_idxs = np.argsort(fused)[::-1][:top_k]
    top_scores = fused[top_idxs]
    component_scores = {
        "tfidf": tfidf_norm,
        "w2v": w2v_norm,
        "sbert": sbert_norm,
        "fused": fused,
    }
    return top_idxs, top_scores, component_scores


def cross_encoder_rerank(
    query: str,
    candidate_texts: List[str],
    top_k: int = config.TOP_K_FINAL,
    cross_encoder_model: CrossEncoder | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rerank candidates using cross-encoder model.
    
    Args:
        query: User query text
        candidate_texts: List of candidate text passages
        top_k: Number of top results to return
        cross_encoder_model: Pre-loaded cross-encoder model
    
    Returns:
        order: Indices of top-k candidates (sorted by score, highest first)
        raw_scores: Raw cross-encoder scores (can be negative)
        normalized_scores: Scores normalized to [0, 1] range for threshold comparison
    """
    # Handle empty candidate list
    if not candidate_texts or len(candidate_texts) == 0:
        return np.array([], dtype=int), np.array([]), np.array([])
    
    cross_encoder = cross_encoder_model or CrossEncoder(config.CROSS_ENCODER_NAME, device=config.DEVICE)
    pairs = [[query, text] for text in candidate_texts]
    raw_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    
    # Ensure raw_scores is a numpy array
    raw_scores = np.asarray(raw_scores)
    
    # Normalize scores to [0, 1] for threshold comparison
    if len(raw_scores) == 0:
        return np.array([], dtype=int), np.array([]), np.array([])
    
    if raw_scores.max() - raw_scores.min() > 1e-9:
        normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        normalized_scores = np.ones_like(raw_scores) * 0.5  # All same score
    
    order = np.argsort(raw_scores)[::-1][:top_k]
    # Ensure order indices are valid
    order = order[order < len(raw_scores)]
    return order, raw_scores[order], normalized_scores[order]



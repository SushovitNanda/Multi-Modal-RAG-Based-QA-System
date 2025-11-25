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
    clip_model: Optional[SentenceTransformer] = None  # For cross-modal retrieval
    clip_image_embs: Optional[np.ndarray] = None  # Image embeddings from CLIP


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
    n_docs = max(max(rank) for rank in rankings) + 1 if rankings else 0
    rrf_scores = np.zeros(n_docs)
    
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
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
    n_docs = len(artifacts.chunks)
    
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

    # 4. Cross-modal CLIP retrieval (if available)
    clip_scores = None
    clip_ranking = None
    if artifacts.clip_model is not None and artifacts.clip_image_embs is not None:
        # Encode query text with CLIP
        query_emb = artifacts.clip_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_emb)
        # Compute similarity with image embeddings
        clip_scores = np.dot(artifacts.clip_image_embs, query_emb.reshape(-1))
        clip_ranking = np.argsort(clip_scores)[::-1]

    # Normalize component scores for display
    tfidf_norm = _normalize(tfidf_scores)
    w2v_norm = _normalize(w2v_scores)
    sbert_norm = _normalize(sbert_scores)
    clip_norm = _normalize(clip_scores) if clip_scores is not None else np.zeros(n_docs)

    # Choose fusion method: RRF or weighted sum
    if use_rrf:
        # Reciprocal Rank Fusion
        rankings = [tfidf_ranking, w2v_ranking, sbert_ranking]
        if clip_ranking is not None:
            rankings.append(clip_ranking)
        fused = reciprocal_rank_fusion(rankings, k=config.RRF_K)
    else:
        # Weighted sum (original method)
        fused = (
            config.HYBRID_WEIGHTS.tfidf * tfidf_norm
            + config.HYBRID_WEIGHTS.word2vec * w2v_norm
            + config.HYBRID_WEIGHTS.sbert * sbert_norm
        )
        if clip_scores is not None:
            # Add CLIP with small weight if available
            fused += 0.1 * clip_norm

    top_idxs = np.argsort(fused)[::-1][:top_k]
    top_scores = fused[top_idxs]
    component_scores = {
        "tfidf": tfidf_norm,
        "w2v": w2v_norm,
        "sbert": sbert_norm,
        "clip": clip_norm if clip_scores is not None else np.zeros(n_docs),
        "fused": fused,
    }
    return top_idxs, top_scores, component_scores


def cross_encoder_rerank(
    query: str,
    candidate_texts: List[str],
    top_k: int = config.TOP_K_FINAL,
    cross_encoder_model: CrossEncoder | None = None,
    artifacts: Optional[RetrievalArtifacts] = None,
    candidate_chunk_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rerank candidates using cross-encoder model with optional cross-modal support.
    
    Args:
        query: User query text
        candidate_texts: List of candidate text passages
        top_k: Number of top results to return
        cross_encoder_model: Pre-loaded cross-encoder model
        artifacts: Retrieval artifacts (for cross-modal reranking)
        candidate_chunk_indices: Original chunk indices for candidates (for CLIP mapping)
    
    Returns:
        order: Indices of top-k candidates (sorted by score, highest first)
        raw_scores: Raw cross-encoder scores (can be negative)
        normalized_scores: Scores normalized to [0, 1] range for threshold comparison
    """
    cross_encoder = cross_encoder_model or CrossEncoder(config.CROSS_ENCODER_NAME, device=config.DEVICE)
    pairs = [[query, text] for text in candidate_texts]
    raw_scores = cross_encoder.predict(pairs, show_progress_bar=False)
    
    # Cross-modal reranking: boost scores for image chunks if CLIP similarity is high
    if (artifacts is not None and artifacts.clip_model is not None and 
        artifacts.clip_image_embs is not None and candidate_chunk_indices is not None):
        # Encode query with CLIP
        query_clip_emb = artifacts.clip_model.encode([query], convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(query_clip_emb)
        
        # Boost scores for image chunks based on CLIP similarity
        for cand_idx, chunk_idx in enumerate(candidate_chunk_indices):
            if chunk_idx < len(artifacts.chunks):
                chunk = artifacts.chunks[chunk_idx]
                if chunk.get("type") == "image_ocr" and chunk_idx < len(artifacts.clip_image_embs):
                    clip_sim = np.dot(artifacts.clip_image_embs[chunk_idx], query_clip_emb.reshape(-1))
                    # Boost cross-encoder score with CLIP similarity (weighted combination)
                    raw_scores[cand_idx] = 0.7 * raw_scores[cand_idx] + 0.3 * clip_sim * 5.0  # Scale CLIP score
    
    # Normalize scores to [0, 1] for threshold comparison
    if raw_scores.max() - raw_scores.min() > 1e-9:
        normalized_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        normalized_scores = np.ones_like(raw_scores) * 0.5  # All same score
    
    order = np.argsort(raw_scores)[::-1][:top_k]
    return order, raw_scores[order], normalized_scores[order]



"""
IR evaluation helpers (MRR, NDCG, Precision/Recall).
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def build_label_matrix(chunks: List[Dict]) -> np.ndarray:
    pages = sorted({c.get("page") for c in chunks if c.get("page") is not None})
    page_to_idx = {p: i for i, p in enumerate(pages)}
    labels = np.zeros((len(chunks), len(pages)), dtype=int)
    for idx, chunk in enumerate(chunks):
        page = chunk.get("page")
        if page is None:
            continue
        labels[idx, page_to_idx[page]] = 1
    return labels


def calculate_ir_metrics_for_query(similarities: np.ndarray, query_label_indices: np.ndarray, labels_mat: np.ndarray, k: int = 5):
    ranked = np.argsort(similarities)[::-1]
    relevant_set = set()
    for qidx in query_label_indices:
        if qidx < 0:
            continue
        labels = np.where(labels_mat[qidx] == 1)[0]
        for lab in labels:
            relevant = set(np.where(labels_mat[:, lab] == 1)[0].tolist())
            relevant_set.update(relevant)
    for qidx in query_label_indices:
        relevant_set.discard(qidx)
    if not relevant_set:
        return {"mrr": 0.0, "ndcg": 0.0, "ndcg_at_k": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0, "f1_at_k": 0.0}

    top_k = ranked[:k]
    retrieved_relevant_k = len(set(top_k) & relevant_set)
    precision_k = retrieved_relevant_k / k
    recall_k = retrieved_relevant_k / len(relevant_set)
    f1_k = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0

    cutoff = min(len(ranked), max(1, len(relevant_set) * 2))
    retrieved_docs = ranked[:cutoff]
    retrieved_relevant = len(set(retrieved_docs) & relevant_set)
    precision = retrieved_relevant / cutoff if cutoff > 0 else 0.0
    recall = retrieved_relevant / len(relevant_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mrr = 0.0
    for rank_idx, doc_idx in enumerate(top_k, 1):
        if doc_idx in relevant_set:
            mrr = 1.0 / rank_idx
            break

    dcg = 0.0
    for i, doc_idx in enumerate(top_k, 1):
        rel = 1 if doc_idx in relevant_set else 0
        dcg += rel / math.log2(i + 1)
    ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_set), k) + 1))
    ndcg_k = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    cutoff_overall = min(len(ranked), 100)
    dcg_overall = 0.0
    for i, doc_idx in enumerate(ranked[:cutoff_overall], 1):
        rel = 1 if doc_idx in relevant_set else 0
        dcg_overall += rel / math.log2(i + 1)
    ideal_overall = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_set), cutoff_overall) + 1))
    ndcg_overall = dcg_overall / ideal_overall if ideal_overall > 0 else 0.0

    return {"mrr": mrr, "ndcg": ndcg_overall, "ndcg_at_k": ndcg_k, "precision": precision, "recall": recall, "f1": f1, "precision_at_k": precision_k, "recall_at_k": recall_k, "f1_at_k": f1_k}



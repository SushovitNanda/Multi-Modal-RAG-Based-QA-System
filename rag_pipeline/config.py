"""
Centralized configuration for the multi-modal RAG QA system.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DOC_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_PATH = PROJECT_ROOT / "Prompt.json"
ASSIGNMENT_DOC_LOCAL_PATH = PROJECT_ROOT / "multi-modal_rag_qa_assignment.docx"
API_KEY_PATH = PROJECT_ROOT / "Api_key.txt"
HF_TOKEN_PATH = PROJECT_ROOT / "Hf_token.txt"
HF_CHAT_MODEL = os.environ.get("HF_CHAT_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

PROCESSED_JSON = PROCESSED_DIR / "processed_chunks.json"
TFIDF_MODEL_PATH = PROCESSED_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_PATH = PROCESSED_DIR / "tfidf_matrix.npz"
FAISS_INDEX_DIR = PROCESSED_DIR / "faiss_index"
SBERT_EMB_PATH = PROCESSED_DIR / "sbert_doc_embeddings.npy"
W2V_EMB_PATH = PROCESSED_DIR / "w2v_doc_embs.npy"
CLIP_EMB_PATH = PROCESSED_DIR / "clip_image_embeddings.npy"
DOC_TEXTS_PATH = PROCESSED_DIR / "doc_texts.json"

WORD2VEC_NAME = "word2vec-google-news-300"
SBERT_MODEL_NAME = "all-mpnet-base-v2"
CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Vision-text model for cross-modal retrieval (CLIP)
CLIP_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K_CANDIDATES = 20
TOP_K_FINAL = 5
RELEVANCE_THRESHOLD = 0.30  # Minimum rerank score to proceed with LLM generation
USE_RRF = True  # Use Reciprocal Rank Fusion for hybrid search
RRF_K = 60  # RRF constant (typical values: 20-100)


@dataclass
class HybridWeights:
    tfidf: float = 0.35
    word2vec: float = 0.30
    sbert: float = 0.15
    cross_encoder: float = 0.20

    def validate(self) -> None:
        total = self.tfidf + self.word2vec + self.sbert + self.cross_encoder
        if not 0.99 <= total <= 1.01:
            raise ValueError("Hybrid weights must sum to 1.0 per Prompt.json guidance.")


@dataclass
class PromptInstructions:
    role: str
    name: str
    purpose: str
    behavior_rules: dict
    rag_pipeline: dict
    langchain_components: dict
    answer_format: dict
    failure_modes: dict


def load_prompt_instructions() -> PromptInstructions:
    with PROMPT_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return PromptInstructions(**data)


def load_openai_key() -> str | None:
    if not API_KEY_PATH.exists():
        return None
    key = API_KEY_PATH.read_text(encoding="utf-8").strip()
    return key or None


def load_hf_token() -> str | None:
    token = os.environ.get("HF_API_TOKEN")
    if token:
        return token
    if HF_TOKEN_PATH.exists():
        return HF_TOKEN_PATH.read_text(encoding="utf-8").strip() or None
    return None


def list_supported_modalities(prompt: PromptInstructions) -> List[str]:
    return prompt.rag_pipeline.get("modalities_supported", [])


HYBRID_WEIGHTS = HybridWeights()
HYBRID_WEIGHTS.validate()

# Detect device: use CUDA if available and not forced to CPU
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1" else "cpu"
except ImportError:
    DEVICE = "cpu"



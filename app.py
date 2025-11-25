"""
Streamlit UI for the multi-modal RAG QA chatbot.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import warnings

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sentence_transformers import CrossEncoder

# Suppress warnings about sampling parameters when do_sample=False
warnings.filterwarnings("ignore", message=".*do_sample.*temperature.*")
warnings.filterwarnings("ignore", message=".*do_sample.*top_p.*")
warnings.filterwarnings("ignore", message=".*do_sample.*top_k.*")
warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")

# Import from rag_pipeline package (file is now at root level)
import rag_pipeline.config as config
from rag_pipeline.pipeline import build_pipeline_and_index
from rag_pipeline.retrieval import cross_encoder_rerank, hybrid_retrieval


def _format_citation(chunk: dict) -> str:
    meta = chunk.get("metadata", {})
    source = Path(meta.get("source", "unknown")).name
    page = chunk.get("page", "?")
    chunk_id = chunk.get("id")
    return f"(source: {source}, page: {page}, chunk: {chunk_id})"


def _context_with_citations(chunks: List[dict]) -> str:
    pieces = []
    for idx, chunk in enumerate(chunks, 1):
        citation = _format_citation(chunk)
        pieces.append(f"[Chunk {idx}] {chunk['content']}\n{citation}")
    return "\n\n---\n\n".join(pieces)


def _ensure_llm_model(model_name: str):
    """Load and cache the LLM model and tokenizer in session state."""
    cache_key = f"llm_model_{model_name}"
    if cache_key not in st.session_state:
        hf_token = config.load_hf_token()
        if not hf_token:
            raise ValueError("Hugging Face token required to load model. Add it to Hf_token.txt or set HF_API_TOKEN.")
        
        with st.spinner(f"Loading {model_name} (this may take a few minutes on first run)..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                torch_dtype=torch.float16 if config.DEVICE == "cuda" else torch.float32,
                device_map="auto" if config.DEVICE == "cuda" else None,
                low_cpu_mem_usage=True,
            )
            if config.DEVICE == "cpu":
                model = model.to(config.DEVICE)
            model.eval()
            st.session_state[cache_key] = {"model": model, "tokenizer": tokenizer}
    return st.session_state[cache_key]["model"], st.session_state[cache_key]["tokenizer"]


def _call_llm(question: str, context: str, model_name: str) -> str:
    try:
        model, tokenizer = _ensure_llm_model(model_name)
    except Exception as exc:
        return f"Failed to load model: {exc}"
    
    instructions = config.load_prompt_instructions()
    behavior = instructions.behavior_rules
    system_prompt = (
        f"You are {instructions.name}, whose purpose is {instructions.purpose}. "
        f"{behavior.get('strict_grounding')} "
        f"{behavior.get('citation_requirement')} "
        f"{behavior.get('confidence_policy')}"
    )
    user_prompt = (
        "You are given multiple knowledge chunks (each tagged as [Chunk N]) that were retrieved for the user query. "
        "Write a comprehensive, detailed, and nuanced answer that thoroughly integrates the evidence from these chunks. "
        "Your answer should:\n"
        "- Be substantial and informative (aim for 3-5 sentences minimum, more if the context is rich)\n"
        "- Synthesize information from multiple chunks when relevant\n"
        "- Provide specific details and examples from the retrieved passages\n"
        "- Reference the chunk IDs in parentheses like (source: file, page: #, chunk: id) when citing specific information\n"
        "- If no chunk contains relevant details, respond strictly with 'No relevant information found in the ingested documents.'\n\n"
        "Context (retrieved passages):\n"
        f"{context}\n\n"
        f"User question: {question}\n\n"
        "Provide a detailed, comprehensive answer based on the context above:"
    )
    
    # Use tokenizer's chat template if available (works for Llama 3, Qwen, etc.)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback to manual formatting for models without chat template
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
        
        # Set pad_token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        with torch.no_grad():
            # Create a clean generation config for greedy decoding (deterministic)
            # Increased max_new_tokens for longer, more detailed responses
            gen_config = GenerationConfig(
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
            )
        # Decode only the newly generated tokens
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated_text.strip()
    except Exception as exc:
        return f"LLM generation failed: {exc}"


def _ensure_cross_encoder() -> CrossEncoder:
    if "cross_encoder" not in st.session_state:
        st.session_state["cross_encoder"] = CrossEncoder(config.CROSS_ENCODER_NAME, device=config.DEVICE)
    return st.session_state["cross_encoder"]


def _answer_question(query: str, model_name: str, relevance_threshold: float = config.RELEVANCE_THRESHOLD, use_rrf: bool = config.USE_RRF) -> Dict[str, Any]:
    artifacts = _ensure_artifacts()
    candidate_idxs, candidate_scores, comp_scores = hybrid_retrieval(query, artifacts, use_rrf=use_rrf)
    candidates = [artifacts.chunks[i] for i in candidate_idxs]

    cross_encoder = _ensure_cross_encoder()
    order, rerank_scores_raw, rerank_scores_norm = cross_encoder_rerank(
        query,
        [c["content"] for c in candidates],
        cross_encoder_model=cross_encoder,
        top_k=config.TOP_K_FINAL,  # Ensure we get exactly top 5
        artifacts=artifacts,  # Pass artifacts for cross-modal reranking
        candidate_chunk_indices=candidate_idxs,  # Pass chunk indices for CLIP mapping
    )
    final_docs = [candidates[i] for i in order]

    # Check relevance threshold using normalized scores (0-1 range)
    if not final_docs or len(rerank_scores_norm) == 0:
        answer = "No relevant information found in the ingested documents."
        context = ""
    elif rerank_scores_norm[0] < relevance_threshold:
        # Top passage relevance is too low - return early without calling LLM
        answer = "No relevant information found in the ingested documents."
        context = ""
    else:
        # Use exactly top 5 passages (already limited by TOP_K_FINAL)
        context = _context_with_citations(final_docs[:config.TOP_K_FINAL])
        answer = _call_llm(query, context if context.strip() else "No context", model_name=model_name)

    # Get chunk IDs for candidates (before reranking)
    candidate_chunk_ids = [candidates[i].get("id", f"idx_{candidate_idxs[i]}") for i in range(len(candidates))]
    
    return {
        "answer": answer,
        "final_docs": final_docs[:config.TOP_K_FINAL],  # Return exactly top 5
        "candidate_idxs": candidate_idxs,
        "candidate_chunk_ids": candidate_chunk_ids,  # Chunk IDs for candidates
        "candidate_scores": candidate_scores,
        "component_scores": comp_scores,
        "rerank_scores_raw": rerank_scores_raw[:config.TOP_K_FINAL] if len(rerank_scores_raw) > 0 else [],
        "rerank_scores_norm": rerank_scores_norm[:config.TOP_K_FINAL] if len(rerank_scores_norm) > 0 else [],
    }


def _ensure_artifacts(rebuild: bool = False):
    if rebuild or "artifacts" not in st.session_state:
        artifacts = build_pipeline_and_index(rebuild=rebuild)
        st.session_state["artifacts"] = artifacts
    return st.session_state["artifacts"]


def run():
    st.set_page_config(page_title="Multi-Modal RAG QA", layout="wide")
    st.title("Multi-Modal RAG QA ChatBot")

    prompt_instructions = config.load_prompt_instructions()
    st.sidebar.header("Configuration")
    st.sidebar.write(f"Supported modalities: {', '.join(config.list_supported_modalities(prompt_instructions))}")
    model_name = st.sidebar.text_input("HF chat model", value=config.HF_CHAT_MODEL)
    use_rrf = st.sidebar.checkbox("Use RRF (Reciprocal Rank Fusion)", value=config.USE_RRF, 
                                   help="Combine retrieval methods using RRF instead of weighted sum")
    relevance_threshold = st.sidebar.slider(
        "Relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.RELEVANCE_THRESHOLD,
        step=0.05,
        help="Minimum normalized rerank score (0-1) to generate answer. Uses cross-encoder scores normalized to [0,1]. Below this, returns 'No relevant information found'.",
    )
    rebuild = st.sidebar.checkbox(
        "Rebuild processed data", 
        value=False,
        help="If checked, forces complete rebuild of all embeddings and indices from scratch. If unchecked, loads from cache if available."
    )
    if st.sidebar.button("Build / Load pipeline"):
        with st.spinner("Preparing hybrid pipeline (ingestion → embeddings → indices)..."):
            _ensure_artifacts(rebuild=rebuild)
        
        # Pre-load LLM model so it's ready when user asks questions
        with st.spinner(f"Loading LLM model ({model_name})..."):
            try:
                _ensure_llm_model(model_name)
                st.success("Pipeline and LLM ready!")
            except Exception as e:
                st.warning(f"Pipeline ready, but LLM loading failed: {e}. It will be loaded on first query.")
                st.success("Pipeline ready (LLM will load on first query).")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "retrieval_debug" not in st.session_state:
        st.session_state["retrieval_debug"] = None

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask about the ingested documents")
    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state["chat_history"].append({"role": "user", "content": user_query})

        with st.spinner("Retrieving and composing answer..."):
            if "artifacts" not in st.session_state:
                _ensure_artifacts()
            result = _answer_question(user_query, model_name=model_name, relevance_threshold=relevance_threshold, use_rrf=use_rrf)

        with st.chat_message("assistant"):
            st.markdown(result["answer"])
        st.session_state["chat_history"].append({"role": "assistant", "content": result["answer"]})
        st.session_state["retrieval_debug"] = result

    debug = st.session_state.get("retrieval_debug")
    if debug:
        with st.expander("Show supporting chunks and scores"):
            st.markdown("**Top Passages (after rerank)**")
            rerank_scores_raw = debug.get("rerank_scores_raw", [])
            rerank_scores_norm = debug.get("rerank_scores_norm", [])
            for rank, doc in enumerate(debug["final_docs"], 1):
                if rank <= len(rerank_scores_norm):
                    score_text = f" (rerank: raw={rerank_scores_raw[rank-1]:.3f}, norm={rerank_scores_norm[rank-1]:.3f})"
                else:
                    score_text = ""
                st.markdown(f"**Rank {rank}**{score_text} {_format_citation(doc)}")
                st.write(doc["content"][:800] + ("..." if len(doc["content"]) > 800 else ""))
                st.write("---")

            st.markdown("**Fusion Component Scores (top candidates from hybrid retrieval)**")
            st.caption("Note: 'doc_idx' is the index into chunks array. 'chunk_id' is the actual chunk identifier.")
            top_n = min(len(debug["candidate_idxs"]), config.TOP_K_CANDIDATES)
            doc_slice = debug["candidate_idxs"][:top_n]
            chunk_ids = debug.get("candidate_chunk_ids", [f"idx_{idx}" for idx in doc_slice])[:top_n]
            
            score_dict = {
                "doc_idx": doc_slice,
                "chunk_id": chunk_ids,
                "fused_score": debug["candidate_scores"][:top_n],
                "tfidf": debug["component_scores"]["tfidf"][doc_slice],
                "w2v": debug["component_scores"]["w2v"][doc_slice],
                "sbert": debug["component_scores"]["sbert"][doc_slice],
            }
            # Add CLIP scores if available
            if "clip" in debug["component_scores"]:
                score_dict["clip"] = debug["component_scores"]["clip"][doc_slice]
            df_scores = pd.DataFrame(score_dict)
            st.dataframe(df_scores)
            
            # Show rerank scores separately
            rerank_scores_norm = debug.get("rerank_scores_norm")
            if rerank_scores_norm is not None and len(rerank_scores_norm) > 0:
                st.markdown("**Cross-Encoder Rerank Scores (final top 5)**")
                rerank_df = pd.DataFrame({
                    "rank": range(1, len(debug["final_docs"]) + 1),
                    "chunk_id": [doc.get("id", "unknown") for doc in debug["final_docs"]],
                    "rerank_raw": debug.get("rerank_scores_raw", [])[:len(debug["final_docs"])],
                    "rerank_norm": rerank_scores_norm[:len(debug["final_docs"])],
                })
                st.dataframe(rerank_df)
                st.caption("'rerank_raw' = raw cross-encoder score (can be negative). 'rerank_norm' = normalized [0,1] for threshold comparison.")


if __name__ == "__main__":
    run()


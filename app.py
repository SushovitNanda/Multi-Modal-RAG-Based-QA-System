"""
Streamlit UI for the multi-modal RAG QA chatbot.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Any

import warnings

# Set up stderr filtering BEFORE any other imports to catch C-level warnings
class FilteredStderr:
    """Filter stderr to suppress PDF processing warnings from C-level code."""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.filter_patterns = [
            "Cannot set gray non-stroke color",
            "invalid float value",
            "does not lie in column range",
        ]
    
    def write(self, text):
        # Filter out unwanted messages
        if any(pattern in text for pattern in self.filter_patterns):
            return  # Suppress the message
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        # Forward any other attributes to original stderr
        return getattr(self.original_stderr, name)

# Apply stderr filter globally
sys.stderr = FilteredStderr(sys.stderr)

import numpy as np
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
# Suppress PDF processing warnings from PyMuPDF
warnings.filterwarnings("ignore", message=".*Cannot set gray non-stroke color.*")
warnings.filterwarnings("ignore", message=".*invalid float value.*")
warnings.filterwarnings("ignore", category=UserWarning, module="fitz")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="fitz")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="camelot")

# Import from rag_pipeline package (file is now at root level)
import rag_pipeline.config as config
from rag_pipeline.pipeline import build_pipeline_and_index
from rag_pipeline.retrieval import cross_encoder_rerank, hybrid_retrieval
from rag_pipeline.summarization import generate_summary, generate_topic_briefing


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
    import time
    start_time = time.time()
    
    artifacts = _ensure_artifacts()
    retrieval_start = time.time()
    candidate_idxs, candidate_scores, comp_scores = hybrid_retrieval(query, artifacts, use_rrf=use_rrf)
    retrieval_time = time.time() - retrieval_start
    
    # Validate candidate indices
    if len(candidate_idxs) == 0:
        return {
            "answer": "No relevant information found in the ingested documents.",
            "final_docs": [],
            "candidate_idxs": [],
            "candidate_chunk_ids": [],
            "candidate_scores": [],
            "component_scores": comp_scores,
            "rerank_scores_raw": [],
            "rerank_scores_norm": [],
        }
    
    # Filter out invalid indices
    valid_idxs = [i for i in candidate_idxs if 0 <= i < len(artifacts.chunks)]
    if len(valid_idxs) == 0:
        return {
            "answer": "No relevant information found in the ingested documents.",
            "final_docs": [],
            "candidate_idxs": [],
            "candidate_chunk_ids": [],
            "candidate_scores": [],
            "component_scores": comp_scores,
            "rerank_scores_raw": [],
            "rerank_scores_norm": [],
        }
    
    candidates = [artifacts.chunks[i] for i in valid_idxs]

    cross_encoder = _ensure_cross_encoder()
    rerank_start = time.time()
    order, rerank_scores_raw, rerank_scores_norm = cross_encoder_rerank(
        query,
        [c["content"] for c in candidates],
        cross_encoder_model=cross_encoder,
        top_k=config.TOP_K_FINAL,  # Ensure we get exactly top 5
    )
    rerank_time = time.time() - rerank_start
    
    # Validate order indices
    valid_order = [i for i in order if 0 <= i < len(candidates)]
    final_docs = [candidates[i] for i in valid_order]

    # Check relevance threshold using normalized scores (0-1 range)
    llm_time = 0.0
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
        llm_start = time.time()
        answer = _call_llm(query, context if context.strip() else "No context", model_name=model_name)
        llm_time = time.time() - llm_start
    
    total_time = time.time() - start_time

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
        "latency": {
            "retrieval": retrieval_time,
            "rerank": rerank_time,
            "llm": llm_time,
            "total": total_time,
        },
    }


def _ensure_artifacts(rebuild: bool = False):
    if rebuild or "artifacts" not in st.session_state:
        artifacts = build_pipeline_and_index(rebuild=rebuild)
        st.session_state["artifacts"] = artifacts
    return st.session_state["artifacts"]


def _show_summary(summary_result: Dict[str, Any]):
    """Display summary/briefing in the main area."""
    st.header("📄 Summary / Briefing")
    st.markdown(f"**Type:** {summary_result.get('type', 'unknown').title()}")
    if summary_result.get('focus_topic'):
        st.markdown(f"**Focus Topic:** {summary_result['focus_topic']}")
    st.markdown(f"**Chunks Used:** {summary_result.get('chunks_used', 0)} / {summary_result.get('total_chunks', 0)}")
    st.divider()
    st.markdown(summary_result.get('summary', 'No summary generated'))


def _show_evaluation_dashboard():
    """Display evaluation dashboard with latency metrics."""
    st.header("📊 Evaluation Dashboard")
    
    # Latency metrics
    st.subheader("⏱️ Latency Metrics")
    latency_history = st.session_state.get("latency_history", [])
    if latency_history:
        latency_df = pd.DataFrame(latency_history)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Total", f"{latency_df['total'].mean():.3f}s")
        with col2:
            st.metric("Avg Retrieval", f"{latency_df['retrieval'].mean():.3f}s")
        with col3:
            st.metric("Avg Rerank", f"{latency_df['rerank'].mean():.3f}s")
        with col4:
            st.metric("Avg LLM", f"{latency_df['llm'].mean():.3f}s")
        
        # Latency chart
        st.line_chart(latency_df[['retrieval', 'rerank', 'llm', 'total']])
        
        # Latency table
        with st.expander("View Latency History"):
            st.dataframe(latency_df)
    else:
        st.info("No latency data yet. Ask some questions to see metrics.")
    
    # Clear data button
    if st.button("Clear All Metrics"):
        st.session_state["latency_history"] = []
        st.rerun()


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
            artifacts = _ensure_artifacts(rebuild=rebuild)
            
            # Verify multi-modal framework
            chunk_types = {}
            for chunk in artifacts.chunks:
                chunk_type = chunk.get("type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            # Display multi-modal statistics
            st.sidebar.success("Pipeline loaded successfully!")
            with st.sidebar.expander("Multi-Modal Statistics"):
                st.write(f"**Total chunks:** {len(artifacts.chunks)}")
                st.write(f"**Text chunks:** {chunk_types.get('page_text', 0)}")
                st.write(f"**Table chunks:** {chunk_types.get('table', 0)}")
                st.write(f"**Image chunks:** {chunk_types.get('image_ocr', 0)}")
        
        # Pre-load LLM model so it's ready when user asks questions
        with st.spinner(f"Loading LLM model ({model_name})..."):
            try:
                _ensure_llm_model(model_name)
                st.success("Pipeline and LLM ready!")
            except Exception as e:
                st.warning(f"Pipeline ready, but LLM loading failed: {e}. It will be loaded on first query.")
                st.success("Pipeline ready (LLM will load on first query).")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "retrieval_debug" not in st.session_state:
        st.session_state["retrieval_debug"] = None
    if "latency_history" not in st.session_state:
        st.session_state["latency_history"] = []  # Store latency for each query

    # Summarization section
    st.sidebar.header("📄 Summarization")
    summary_type = st.sidebar.selectbox(
        "Summary Type",
        ["document", "briefing", "executive"],
        help="Choose the type of summary to generate"
    )
    focus_topic = st.sidebar.text_input(
        "Focus Topic (optional)",
        placeholder="e.g., 'economic growth'",
        help="Generate summary focused on a specific topic"
    )
    if st.sidebar.button("Generate Summary"):
        if "artifacts" not in st.session_state:
            st.sidebar.warning("Please build/load pipeline first")
        else:
            try:
                artifacts = _ensure_artifacts()
                model, tokenizer = _ensure_llm_model(model_name)
                with st.spinner(f"Generating {summary_type} summary..."):
                    if focus_topic and focus_topic.strip():
                        summary_result = generate_topic_briefing(
                            focus_topic.strip(), artifacts, model, tokenizer
                        )
                    else:
                        summary_result = generate_summary(
                            artifacts, model, tokenizer, summary_type=summary_type
                        )
                st.sidebar.success("Summary generated!")
                st.session_state["summary"] = summary_result
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error generating summary: {e}")
    
    # Evaluation Dashboard section
    st.sidebar.header("📊 Evaluation Dashboard")
    if st.sidebar.button("View Dashboard"):
        st.session_state["show_dashboard"] = True
        st.rerun()
    
    # Main content area - check which view to show
    if st.session_state.get("show_dashboard", False):
        _show_evaluation_dashboard()
        if st.button("← Back to Chat"):
            st.session_state["show_dashboard"] = False
            st.rerun()
    elif st.session_state.get("summary"):
        _show_summary(st.session_state["summary"])
        if st.button("← Back to Chat"):
            st.session_state["summary"] = None
            st.rerun()
    else:
        # Regular chat interface
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
                
                # Store latency
                if "latency" in result:
                    st.session_state["latency_history"].append({
                        "query": user_query[:50] + "..." if len(user_query) > 50 else user_query,
                        **result["latency"]
                    })

            with st.chat_message("assistant"):
                st.markdown(result["answer"])
                # Show latency info
                if "latency" in result:
                    with st.expander("⏱️ Performance"):
                        lat = result["latency"]
                        st.write(f"**Total:** {lat['total']:.3f}s | **Retrieval:** {lat['retrieval']:.3f}s | **Rerank:** {lat['rerank']:.3f}s | **LLM:** {lat['llm']:.3f}s")
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
            # Show chunk types for top candidates
            artifacts = _ensure_artifacts()
            chunk_types = []
            for idx in doc_slice:  # Show types for all candidates
                if 0 <= idx < len(artifacts.chunks):
                    chunk_type = artifacts.chunks[idx].get("type", "unknown")
                    chunk_types.append(chunk_type)
                else:
                    chunk_types.append("invalid")
            if chunk_types and len(chunk_types) == len(doc_slice):
                score_dict["type"] = chunk_types
            
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


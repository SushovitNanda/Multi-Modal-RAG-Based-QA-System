"""
CLI version of the Multi-Modal RAG QA chatbot.
Provides command-line interface for querying documents without Streamlit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import rag_pipeline.config as config
from rag_pipeline.pipeline import build_pipeline_and_index
from rag_pipeline.retrieval import cross_encoder_rerank, hybrid_retrieval
from sentence_transformers import CrossEncoder
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def _format_citation(chunk: dict) -> str:
    """Format citation string for a chunk."""
    meta = chunk.get("metadata", {})
    source = Path(meta.get("source", "unknown")).name
    page = chunk.get("page", "?")
    chunk_id = chunk.get("id")
    return f"(source: {source}, page: {page}, chunk: {chunk_id})"


def _context_with_citations(chunks: list[dict]) -> str:
    """Build context string with citations."""
    pieces = []
    for idx, chunk in enumerate(chunks, 1):
        citation = _format_citation(chunk)
        pieces.append(f"[Chunk {idx}] {chunk['content']}\n{citation}")
    return "\n\n---\n\n".join(pieces)


def _load_llm_model(model_name: str):
    """Load LLM model and tokenizer."""
    hf_token = config.load_hf_token()
    if not hf_token:
        raise ValueError("Hugging Face token required. Add it to Hf_token.txt or set HF_API_TOKEN.")
    
    print(f"Loading {model_name}...")
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
    return model, tokenizer


def _call_llm(question: str, context: str, model, tokenizer) -> str:
    """Generate answer using LLM."""
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
    
    # Use tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    with torch.no_grad():
        gen_config = GenerationConfig(
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        outputs = model.generate(**inputs, generation_config=gen_config)
    
    generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated_text.strip()


def answer_question(
    query: str,
    artifacts,
    model,
    tokenizer,
    cross_encoder,
    relevance_threshold: float = config.RELEVANCE_THRESHOLD,
    use_rrf: bool = config.USE_RRF,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Answer a question using the RAG pipeline."""
    candidate_idxs, candidate_scores, comp_scores = hybrid_retrieval(query, artifacts, use_rrf=use_rrf)
    
    # Validate candidate indices to prevent IndexError
    if len(candidate_idxs) == 0:
        return {
            "answer": "No relevant information found in the ingested documents.",
            "final_docs": [],
            "rerank_scores": [],
        }
    
    # Filter out invalid indices
    valid_idxs = [i for i in candidate_idxs if 0 <= i < len(artifacts.chunks)]
    if len(valid_idxs) == 0:
        return {
            "answer": "No relevant information found in the ingested documents.",
            "final_docs": [],
            "rerank_scores": [],
        }
    
    candidates = [artifacts.chunks[i] for i in valid_idxs]

    order, rerank_scores_raw, rerank_scores_norm = cross_encoder_rerank(
        query,
        [c["content"] for c in candidates],
        cross_encoder_model=cross_encoder,
        top_k=config.TOP_K_FINAL,
    )
    
    # Validate order indices to prevent IndexError
    valid_order = [i for i in order if 0 <= i < len(candidates)]
    final_docs = [candidates[i] for i in valid_order]

    if not final_docs or len(rerank_scores_norm) == 0:
        answer = "No relevant information found in the ingested documents."
        context = ""
    elif rerank_scores_norm[0] < relevance_threshold:
        answer = "No relevant information found in the ingested documents."
        context = ""
    else:
        context = _context_with_citations(final_docs[:config.TOP_K_FINAL])
        answer = _call_llm(query, context if context.strip() else "No context", model, tokenizer)

    if verbose:
        print("\n" + "="*80)
        print("TOP RETRIEVED PASSAGES:")
        print("="*80)
        for rank, doc in enumerate(final_docs[:config.TOP_K_FINAL], 1):
            print(f"\n[Rank {rank}] {_format_citation(doc)}")
            print(f"Rerank score (norm): {rerank_scores_norm[rank-1]:.3f}" if rank <= len(rerank_scores_norm) else "")
            print(f"Content: {doc['content'][:500]}...")
        print("="*80 + "\n")

    return {
        "answer": answer,
        "final_docs": final_docs[:config.TOP_K_FINAL],
        "rerank_scores": rerank_scores_norm[:config.TOP_K_FINAL] if len(rerank_scores_norm) > 0 else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal RAG QA Chatbot (CLI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single query
  python cli.py "What is the main topic of the document?"
  
  # Interactive mode
  python cli.py --interactive
  
  # With verbose output showing retrieved passages
  python cli.py "Your question" --verbose
  
  # Rebuild pipeline from scratch
  python cli.py --rebuild "Your question"
        """
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Question to ask about the documents (optional if using --interactive)"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (keep asking questions)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild all embeddings and indices from scratch"
    )
    parser.add_argument(
        "--model",
        default=config.HF_CHAT_MODEL,
        help=f"LLM model to use (default: {config.HF_CHAT_MODEL})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.RELEVANCE_THRESHOLD,
        help=f"Relevance threshold (default: {config.RELEVANCE_THRESHOLD})"
    )
    parser.add_argument(
        "--use-rrf",
        action="store_true",
        default=config.USE_RRF,
        help="Use Reciprocal Rank Fusion for retrieval"
    )
    parser.add_argument(
        "--no-rrf",
        action="store_true",
        help="Disable RRF (use weighted sum instead)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show retrieved passages and scores"
    )

    args = parser.parse_args()

    # Handle RRF flag
    use_rrf = args.use_rrf and not args.no_rrf

    # Check if query provided or interactive mode
    if not args.query and not args.interactive:
        parser.error("Either provide a query or use --interactive mode")

    print("="*80)
    print("Multi-Modal RAG QA Chatbot (CLI)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Relevance threshold: {args.threshold}")
    print(f"Use RRF: {use_rrf}")
    print("="*80 + "\n")

    # Load pipeline
    print("Loading pipeline...")
    artifacts = build_pipeline_and_index(rebuild=args.rebuild)
    print("✓ Pipeline loaded\n")

    # Load cross-encoder
    print("Loading cross-encoder...")
    cross_encoder = CrossEncoder(config.CROSS_ENCODER_NAME, device=config.DEVICE)
    print("✓ Cross-encoder loaded\n")

    # Load LLM
    try:
        model, tokenizer = _load_llm_model(args.model)
        print("✓ LLM loaded\n")
    except Exception as e:
        print(f"✗ Failed to load LLM: {e}")
        sys.exit(1)

    # Interactive mode
    if args.interactive:
        print("Entering interactive mode. Type 'quit' or 'exit' to stop.\n")
        while True:
            try:
                query = input("Question: ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                if not query:
                    continue
                
                print("\nProcessing...")
                result = answer_question(
                    query, artifacts, model, tokenizer, cross_encoder,
                    relevance_threshold=args.threshold,
                    use_rrf=use_rrf,
                    verbose=args.verbose
                )
                print("\nAnswer:")
                print("-" * 80)
                print(result["answer"])
                print("-" * 80 + "\n")
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n✗ Error: {e}\n")
    else:
        # Single query mode
        print(f"Question: {args.query}\n")
        print("Processing...")
        result = answer_question(
            args.query, artifacts, model, tokenizer, cross_encoder,
            relevance_threshold=args.threshold,
            use_rrf=use_rrf,
            verbose=args.verbose
        )
        print("\nAnswer:")
        print("-" * 80)
        print(result["answer"])
        print("-" * 80)


if __name__ == "__main__":
    main()


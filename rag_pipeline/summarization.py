"""
Summarization and briefing generation utilities.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

from . import config
from .retrieval import hybrid_retrieval, RetrievalArtifacts


def generate_summary(
    artifacts: RetrievalArtifacts,
    model,
    tokenizer,
    summary_type: str = "document",
    max_chunks: int = 50,
    focus_topic: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a summary or briefing of the ingested documents.
    
    Args:
        artifacts: Retrieval artifacts
        model: LLM model
        tokenizer: LLM tokenizer
        summary_type: Type of summary ("document", "briefing", "executive")
        max_chunks: Maximum number of chunks to include
        focus_topic: Optional topic to focus the summary on
    
    Returns:
        Dict with summary text and metadata
    """
    # Select chunks for summarization
    if focus_topic:
        # Use retrieval to find relevant chunks for the topic
        candidate_idxs, _, _ = hybrid_retrieval(
            focus_topic, artifacts, top_k=max_chunks, use_rrf=True
        )
        selected_chunks = [artifacts.chunks[i] for i in candidate_idxs[:max_chunks]]
    else:
        # Use diverse sampling from all chunks
        import random
        all_indices = list(range(len(artifacts.chunks)))
        selected_indices = random.sample(all_indices, min(max_chunks, len(all_indices)))
        selected_chunks = [artifacts.chunks[i] for i in selected_indices]
    
    # Combine chunk contents
    chunk_texts = []
    for chunk in selected_chunks:
        chunk_type = chunk.get("type", "unknown")
        content = chunk.get("content", "")
        page = chunk.get("page", "?")
        chunk_texts.append(f"[{chunk_type}, page {page}]: {content}")
    
    combined_text = "\n\n".join(chunk_texts)
    
    # Generate summary prompt based on type
    if summary_type == "briefing":
        prompt = (
            "Generate a concise briefing document based on the following extracted content from PDF documents. "
            "The briefing should:\n"
            "- Be structured and well-organized\n"
            "- Highlight key points and findings\n"
            "- Include important statistics, figures, or data\n"
            "- Be suitable for quick reference\n\n"
            "Content:\n" + combined_text[:8000] + "\n\n"
            "Generate the briefing:"
        )
    elif summary_type == "executive":
        prompt = (
            "Generate an executive summary based on the following document content. "
            "The summary should:\n"
            "- Be high-level and strategic\n"
            "- Focus on main conclusions and recommendations\n"
            "- Be concise (2-3 paragraphs)\n"
            "- Suitable for executive decision-making\n\n"
            "Content:\n" + combined_text[:6000] + "\n\n"
            "Generate the executive summary:"
        )
    else:  # document summary
        prompt = (
            "Generate a comprehensive summary of the following document content. "
            "The summary should:\n"
            "- Cover all major topics and themes\n"
            "- Maintain important details and context\n"
            "- Be well-structured with clear sections\n"
            "- Include key information from tables and figures\n\n"
            "Content:\n" + combined_text[:10000] + "\n\n"
            "Generate the summary:"
        )
    
    # Generate summary using LLM
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(config.DEVICE)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        summary_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        return {
            "summary": summary_text.strip(),
            "type": summary_type,
            "chunks_used": len(selected_chunks),
            "total_chunks": len(artifacts.chunks),
            "focus_topic": focus_topic,
        }
    except Exception as e:
        return {
            "summary": f"Error generating summary: {e}",
            "type": summary_type,
            "chunks_used": len(selected_chunks),
            "total_chunks": len(artifacts.chunks),
            "focus_topic": focus_topic,
        }


def generate_topic_briefing(
    topic: str,
    artifacts: RetrievalArtifacts,
    model,
    tokenizer,
    max_chunks: int = 30,
) -> Dict[str, Any]:
    """
    Generate a focused briefing on a specific topic.
    
    Args:
        topic: Topic to generate briefing for
        artifacts: Retrieval artifacts
        model: LLM model
        tokenizer: LLM tokenizer
        max_chunks: Maximum chunks to retrieve
    
    Returns:
        Dict with briefing text and metadata
    """
    return generate_summary(
        artifacts, model, tokenizer,
        summary_type="briefing",
        max_chunks=max_chunks,
        focus_topic=topic,
    )


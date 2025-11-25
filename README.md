# Multi-Modal RAG QA Chatbot

A comprehensive Retrieval-Augmented Generation (RAG) system for question-answering over multi-modal documents (PDFs with text, tables, images, charts). The system uses hybrid retrieval combining TF-IDF, Word2Vec, SBERT embeddings, and cross-modal CLIP embeddings, followed by cross-encoder reranking and LLM-based answer generation.

##  Features

- **Multi-Modal Document Processing**
  - Text extraction from PDFs
  - Table extraction using Camelot
  - Image extraction with OCR (Tesseract)
  - Chart metadata detection

- **Hybrid Retrieval System**
  - **TF-IDF**: Term frequency-inverse document frequency for keyword matching
  - **Word2Vec**: Semantic word embeddings for lexical similarity
  - **SBERT**: Dense semantic embeddings for contextual understanding
  - **CLIP**: Cross-modal vision-text embeddings for image retrieval
  - **Reciprocal Rank Fusion (RRF)**: Optional rank-based fusion method
  - **Weighted Sum**: Score-based fusion with configurable weights

- **Advanced Reranking**
  - Cross-encoder reranking for precision
  - Cross-modal reranking (CLIP + cross-encoder) for image chunks
  - Relevance threshold filtering to prevent hallucinations

- **LLM Integration**
  - Support for Hugging Face instruct models (Qwen, Llama, etc.)
  - Local model loading using Transformers library
  - ChatML-compatible prompt formatting
  - Configurable generation parameters

- **User Interfaces**
  - **Streamlit Web UI**: Interactive chatbot with real-time retrieval visualization
  - **CLI Interface**: Command-line tool for batch processing and scripting

##  Requirements

### System Requirements
- Python 3.8+
- Windows/Linux/macOS
- CUDA-capable GPU (optional, for faster inference)

### Dependencies
All dependencies are listed in `requirements.txt`. Key packages include:
- `streamlit` - Web UI framework
- `transformers` - Hugging Face transformers for LLM
- `sentence-transformers` - Embedding models (SBERT, CLIP)
- `faiss-cpu` - Vector similarity search
- `langchain` - Document processing
- `PyMuPDF`, `pdfplumber` - PDF processing
- `camelot-py` - Table extraction
- `pytesseract` - OCR
- `torch` - Deep learning framework

##  Installation

### Option 1: Automatic Installation (Windows)

1. Double-click `run_chatbot.bat`
2. The script will automatically:
   - Check for Python installation
   - Install missing dependencies
   - Launch the Streamlit chatbot

### Option 2: Manual Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face token:**
   - Create `Hf_token.txt` in the root directory
   - Add your Hugging Face API token (get it from https://huggingface.co/settings/tokens)
   - Or set environment variable: `export HF_API_TOKEN=your_token_here`

4. **Place documents:**
   - Add PDF files to `data/raw/` directory

##  Quick Start

### Streamlit Web Interface

```bash
streamlit run app.py
```

The chatbot will open in your default web browser at `http://localhost:8501`

**First-time setup:**
1. In the sidebar, click **"Build / Load pipeline"**
2. Wait for the pipeline to build (first run may take several minutes)
3. Start asking questions!

### CLI Interface

**Single query:**
```bash
python cli.py "What is the main topic of the document?"
```

**Interactive mode:**
```bash
python cli.py --interactive
```

**With verbose output:**
```bash
python cli.py "Your question" --verbose
```

**Rebuild pipeline:**
```bash
python cli.py --rebuild "Your question"
```

**CLI Options:**
- `--interactive, -i`: Run in interactive mode
- `--rebuild`: Force rebuild of all embeddings
- `--model MODEL`: Specify LLM model (default: Qwen/Qwen2.5-0.5B-Instruct)
- `--threshold FLOAT`: Relevance threshold (default: 0.30)
- `--use-rrf`: Enable Reciprocal Rank Fusion
- `--no-rrf`: Disable RRF (use weighted sum)
- `--verbose, -v`: Show retrieved passages and scores

##  Project Structure

```
RAG-Based QA System/
├── app.py                      # Streamlit web interface
├── cli.py                      # Command-line interface
├── run_chatbot.bat            # Windows launcher script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── Prompt.json                 # LLM prompt instructions
├── Hf_token.txt               # Hugging Face API token
├── Api_key.txt                # OpenAI API key (optional)
├── data/
│   ├── raw/                   # Place PDFs here
│   └── processed/             # Generated embeddings and indices
│       ├── processed_chunks.json
│       ├── faiss_index/
│       ├── images/
│       └── ...
└── rag_pipeline/              # Core pipeline modules
    ├── __init__.py
    ├── config.py              # Configuration and paths
    ├── ingestion.py           # Document ingestion (PDF, tables, images)
    ├── chunking.py            # Text chunking utilities
    ├── embeddings.py          # Embedding generation (TF-IDF, W2V, SBERT)
    ├── retrieval.py           # Hybrid retrieval and reranking
    ├── pipeline.py            # Main orchestration
    ├── evaluation.py          # IR metrics (MRR, NDCG, P/R/F)
    └── Pipeline.md            # Detailed module documentation
```

##  Configuration

### Model Configuration

Edit `rag_pipeline/config.py` or use environment variables:

- `HF_CHAT_MODEL`: LLM model name (default: `Qwen/Qwen2.5-0.5B-Instruct`)
- `SBERT_MODEL_NAME`: Dense embedding model (default: `all-mpnet-base-v2`)
- `CROSS_ENCODER_NAME`: Reranking model (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `CLIP_MODEL_NAME`: Vision-text model (default: `sentence-transformers/clip-ViT-B-32`)

### Retrieval Parameters

- `CHUNK_SIZE`: Text chunk size (default: 600)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 100)
- `TOP_K_CANDIDATES`: Initial retrieval candidates (default: 20)
- `TOP_K_FINAL`: Final passages after reranking (default: 5)
- `RELEVANCE_THRESHOLD`: Minimum score to generate answer (default: 0.30)
- `USE_RRF`: Enable RRF fusion (default: True)
- `RRF_K`: RRF constant (default: 60)

### Hybrid Weights (Weighted Sum Mode)

- `tfidf`: 0.35 (35%)
- `word2vec`: 0.30 (30%)
- `sbert`: 0.15 (15%)
- `cross_encoder`: 0.20 (20%)

##  Pipeline Workflow

1. **Ingestion** (`ingestion.py`)
   - Extract text from PDFs using LangChain/PyMuPDF
   - Extract tables using Camelot
   - Extract images and run OCR using Tesseract
   - Extract chart metadata

2. **Chunking** (`chunking.py`)
   - Split documents into overlapping chunks
   - Preserve metadata (page, source, type)
   - Create unified chunk structure

3. **Embedding Generation** (`embeddings.py`)
   - **TF-IDF**: Build sparse vectorizer and matrix
   - **Word2Vec**: Generate average word embeddings
   - **SBERT**: Generate dense semantic embeddings
   - **CLIP**: Generate vision embeddings for images
   - Build FAISS index for fast similarity search

4. **Hybrid Retrieval** (`retrieval.py`)
   - Compute similarity scores from all methods
   - Fuse scores using RRF or weighted sum
   - Retrieve top-K candidates

5. **Reranking** (`retrieval.py`)
   - Cross-encoder reranking for precision
   - Cross-modal boosting for image chunks
   - Select top 5 final passages

6. **Answer Generation** (`app.py` / `cli.py`)
   - Format context with citations
   - Generate prompt with instructions
   - Call LLM for answer generation
   - Return answer with source citations

## Streamlit UI Features

### Sidebar Configuration
- **HF chat model**: Select or enter model name
- **Use RRF**: Toggle between RRF and weighted sum
- **Relevance threshold**: Adjust minimum score (0.0-1.0)
- **Rebuild processed data**: Force complete rebuild
- **Build / Load pipeline**: Initialize the system

### Main Interface
- **Chat interface**: ChatGPT-like conversation UI
- **Enter to send**: Submit questions with Enter key
- **Retrieval debug panel**: Expandable section showing:
  - Top retrieved passages with scores
  - Fusion component scores (TF-IDF, W2V, SBERT, CLIP)
  - Cross-encoder rerank scores

## Evaluation Metrics

The system includes evaluation utilities (`evaluation.py`) for measuring retrieval performance:

- **MRR** (Mean Reciprocal Rank): Position of first relevant document
- **NDCG** (Normalized Discounted Cumulative Gain): Ranking quality
- **Precision/Recall/F1**: Classification metrics
- **Precision@K / Recall@K / F1@K**: Top-K metrics

## Advanced Usage

### Custom Model Selection

**In Streamlit UI:**
- Enter model name in sidebar: `Qwen/Qwen2.5-7B-Instruct`

**In CLI:**
```bash
python cli.py "Question" --model "meta-llama/Meta-Llama-3-8B-Instruct"
```

**Environment variable:**
```bash
export HF_CHAT_MODEL="Qwen/Qwen2.5-7B-Instruct"
```

### Rebuilding Pipeline

**When to rebuild:**
- After adding new PDFs to `data/raw/`
- After changing chunking parameters
- If cached data is corrupted
- To regenerate CLIP embeddings

**How to rebuild:**
- Streamlit: Check "Rebuild processed data" checkbox, then click "Build / Load pipeline"
- CLI: Use `--rebuild` flag

### Batch Processing

Use CLI for batch question answering:

```bash
# Process multiple questions from file
cat questions.txt | while read q; do
    python cli.py "$q" >> answers.txt
done
```

## Troubleshooting

### Common Issues

**1. "Hugging Face token not found"**
- Ensure `Hf_token.txt` exists in root directory
- Or set `HF_API_TOKEN` environment variable

**2. "Module not found" errors**
- Run: `pip install -r requirements.txt`
- Or use `run_chatbot.bat` for automatic installation

**3. CUDA out of memory**
- Set `FORCE_CPU=1` environment variable
- Or use smaller model (e.g., Qwen 0.5B instead of 7B)

**4. Slow first query**
- LLM loads on first use
- Pre-load by clicking "Build / Load pipeline" button

**5. "No relevant information found"**
- Lower relevance threshold in sidebar
- Check if documents are properly ingested
- Verify PDFs are in `data/raw/` directory

### Performance Tips

- **GPU acceleration**: Use CUDA-capable GPU for faster embeddings and LLM inference
- **Model size**: Smaller models (0.5B-1B) are faster but less capable
- **Caching**: Pipeline artifacts are cached - rebuild only when needed
- **Batch processing**: Use CLI for multiple questions to avoid UI overhead

##  Contributing

This is a modular system designed for easy extension:

- Add new retrieval methods in `retrieval.py`
- Add new document types in `ingestion.py`
- Customize prompts in `Prompt.json`
- Add evaluation metrics in `evaluation.py`

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review `Pipeline.md` for module details



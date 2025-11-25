# RAG Pipeline Module Documentation

This document provides detailed explanations of each module in the `rag_pipeline` package, describing their functions, classes, and how they work together to build the multi-modal RAG system.

##  Package Overview

The `rag_pipeline` package is organized into modular components that handle different stages of the RAG pipeline:

```
rag_pipeline/
├── __init__.py          # Package initialization
├── config.py            # Configuration and constants
├── ingestion.py         # Document ingestion (PDF, tables, images)
├── chunking.py           # Text chunking and segmentation
├── embeddings.py         # Embedding generation (TF-IDF, W2V, SBERT)
├── retrieval.py         # Hybrid retrieval and reranking
├── pipeline.py          # Main orchestration
└── evaluation.py        # IR evaluation metrics
```

---

##  `__init__.py`

**Purpose**: Package initialization and public API exposure.

**Contents**:
- Exports `config` module for easy access
- Defines `__all__` for controlled imports

**Usage**:
```python
from rag_pipeline import config
```

---

##  `config.py`

**Purpose**: Centralized configuration management for the entire system.

### Key Components

#### **Path Configuration**
- `PROJECT_ROOT`: Root directory of the project
- `DATA_DIR`: Data directory (`data/`)
- `RAW_DOC_DIR`: Input documents directory (`data/raw/`)
- `PROCESSED_DIR`: Output directory for processed data (`data/processed/`)
- File paths for all cached artifacts (TF-IDF, embeddings, indices)

#### **Model Configuration**
- `HF_CHAT_MODEL`: Default LLM model (`Qwen/Qwen2.5-0.5B-Instruct`)
- `SBERT_MODEL_NAME`: Dense embedding model (`all-mpnet-base-v2`)
- `CROSS_ENCODER_NAME`: Reranking model (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `CLIP_MODEL_NAME`: Vision-text model (`sentence-transformers/clip-ViT-B-32`)
- `WORD2VEC_NAME`: Word2Vec model (`word2vec-google-news-300`)

#### **Retrieval Parameters**
- `CHUNK_SIZE`: Text chunk size (600 characters)
- `CHUNK_OVERLAP`: Overlap between chunks (100 characters)
- `TOP_K_CANDIDATES`: Initial retrieval candidates (20)
- `TOP_K_FINAL`: Final passages after reranking (5)
- `RELEVANCE_THRESHOLD`: Minimum rerank score to generate answer (0.30)
- `USE_RRF`: Enable Reciprocal Rank Fusion (True)
- `RRF_K`: RRF constant (60)

#### **Data Classes**

**`HybridWeights`**:
- Defines weights for weighted sum fusion
- `tfidf`: 0.35 (35%)
- `word2vec`: 0.30 (30%)
- `sbert`: 0.15 (15%)
- `cross_encoder`: 0.20 (20%)
- Includes validation to ensure weights sum to 1.0

#### **Helper Functions**

**`load_openai_key()`**:
- Loads OpenAI API key from `Api_key.txt` (optional, for OpenAI models)

**`load_hf_token()`**:
- Loads Hugging Face token from `Hf_token.txt` or environment variable
- Required for accessing gated models

**`list_supported_modalities()`**:
- Returns list of supported document modalities from prompt config

#### **Device Detection**
- Automatically detects CUDA availability
- Falls back to CPU if CUDA unavailable or `FORCE_CPU=1` is set

---

##  `ingestion.py`

**Purpose**: Extract content from multi-modal PDF documents (text, tables, images with OCR).

### Functions

#### **`extract_text_with_langchain(pdf_path: str)`**
- **Purpose**: Extract text from PDF using LangChain's PyMuPDFLoader
- **Returns**: List of dicts with `page`, `text`, `source`
- **Features**: 
  - Robust page-wise extraction
  - Preserves page numbers
  - Handles various PDF formats

#### **`extract_tables_pdf(pdf_path: str)`**
- **Purpose**: Extract tabular data from PDF using Camelot
- **Returns**: List of dicts with `page`, `table_text` (CSV format), `source`
- **Strategy**:
  1. Tries "lattice" flavor (for tables with clear borders)
  2. Falls back to "stream" flavor (for borderless tables)
- **Error handling**: Gracefully handles extraction failures

#### **`extract_images_and_ocr(pdf_path: str, output_dir: Path)`**
- **Purpose**: Extract embedded images and perform OCR
- **Process**:
  1. Opens PDF with PyMuPDF (fitz)
  2. Iterates through all pages
  3. Extracts each embedded image
  4. Saves images to `output_dir/images/`
  5. Runs Tesseract OCR on each image
  6. Returns metadata with image path and OCR text
- **Returns**: List of dicts with `page`, `image_path`, `ocr_text`, `source`
- **Output**: Images saved as `{pdf_name}_p{page}_img{index}.{ext}`

#### **`extract_chart_metadata_from_page_text(text: str)`**
- **Purpose**: Heuristic extraction of chart/figure metadata
- **Strategy**:
  - Searches for lines starting with "Figure", "Fig.", "Chart", "Table"
  - Captures lines containing axis/legend keywords
- **Returns**: Dict with `captions` field

#### **`ingest_documents(raw_dir: Path, processed_dir: Path)`**
- **Purpose**: Main ingestion orchestrator
- **Process**:
  1. Scans `raw_dir` for PDF/DOCX files
  2. For each file:
     - Extracts page text (with fallback to pdfplumber)
     - Extracts tables
     - Extracts images and runs OCR
  3. Aggregates all results
- **Returns**: Dict with keys:
  - `pages`: List of page text records
  - `tables`: List of table records
  - `images`: List of image records with OCR

---

## `chunking.py`

**Purpose**: Split documents into manageable chunks with metadata preservation.

### Functions

#### **`clean_text(text: str | None)`**
- **Purpose**: Normalize whitespace in text
- **Process**: Joins all whitespace sequences into single spaces
- **Returns**: Cleaned text string

#### **`chunk_documents(page_records, table_records, image_records, chunk_size, chunk_overlap)`**
- **Purpose**: Create unified chunks from all document types
- **Process**:
  1. **Preparation**: 
     - Extracts content from each record type (text, table_text, ocr_text)
     - Builds metadata map for each document
  2. **Chunking**:
     - Uses `RecursiveCharacterTextSplitter` from LangChain
     - Splits on: `["\n\n", "\n", ".", " ", ""]` (hierarchical)
     - Applies `chunk_size` and `chunk_overlap` parameters
  3. **Output**:
     - Creates chunks with unique IDs (`chunk_0`, `chunk_1`, ...)
     - Preserves metadata: `type`, `page`, `source`, `image_path` (for images)
- **Returns**: List of chunk dicts with:
  - `id`: Unique chunk identifier
  - `type`: "page_text", "table", or "image_ocr"
  - `page`: Page number
  - `content`: Chunk text content
  - `metadata`: Full metadata dict

**Chunk Structure Example**:
```python
{
    "id": "chunk_42",
    "type": "page_text",
    "page": 5,
    "content": "The main topic discusses...",
    "metadata": {
        "type": "page_text",
        "page": 5,
        "source": "path/to/document.pdf"
    }
}
```

---

## `embeddings.py`

**Purpose**: Generate various types of embeddings for retrieval.

### Functions

#### **`save_pickle(obj, path)` / `load_pickle(path)`**
- **Purpose**: Serialize/deserialize Python objects
- **Usage**: Save/load TF-IDF vectorizers, models, etc.

#### **`build_tfidf(doc_texts: List[str])`**
- **Purpose**: Build TF-IDF vectorizer and matrix
- **Parameters**:
  - `ngram_range=(1, 2)`: Unigrams and bigrams
  - `stop_words="english"`: Remove common words
  - `max_features=50000`: Limit vocabulary size
- **Returns**: Tuple of (TfidfVectorizer, sparse matrix)
- **Output**: Sparse scipy matrix for memory efficiency

#### **`load_word2vec_model(name: str)`**
- **Purpose**: Load pre-trained Word2Vec model from Gensim
- **Default**: `word2vec-google-news-300` (300-dimensional vectors)
- **Returns**: Gensim Word2Vec model or None if unavailable
- **Note**: First load downloads ~1.6GB model

#### **`get_avg_word2vec_embeddings(w2v_model, texts: List[str])`**
- **Purpose**: Generate document embeddings by averaging word vectors
- **Process**:
  1. Tokenize each text
  2. Look up each word in Word2Vec model
  3. Average all word vectors for the document
  4. Handle OOV (out-of-vocabulary) words gracefully
- **Returns**: NumPy array of shape `(n_docs, 300)`
- **Fallback**: Returns zero vectors if model unavailable

#### **`build_sbert_and_faiss(texts, model_name, faiss_index_path)`**
- **Purpose**: Generate dense embeddings and build FAISS index
- **Process**:
  1. Load SentenceTransformer model (SBERT)
  2. Encode all documents to dense vectors
  3. Normalize vectors (L2 normalization for cosine similarity)
  4. Build FAISS IndexFlatIP (Inner Product = cosine similarity)
  5. Save index and embeddings to disk
- **Returns**: Tuple of (SBERT model, FAISS index, embeddings array)
- **FAISS Index**: Enables fast similarity search (O(n) with IndexFlatIP)

#### **`persist_doc_texts(doc_texts: List[str])`**
- **Purpose**: Save document texts to JSON for reference
- **Usage**: Used for debugging and reference

#### **`cosine_sim_matrix(vectorizer, tfidf_matrix, query: str)`**
- **Purpose**: Compute TF-IDF cosine similarity between query and all documents
- **Process**:
  1. Transform query to TF-IDF vector
  2. Compute cosine similarity with document matrix
- **Returns**: NumPy array of similarity scores

---

## `retrieval.py`

**Purpose**: Hybrid retrieval, reranking, and fusion logic.

### Research Background

The retrieval methods implemented in this module (TF-IDF, Word2Vec, SBERT) were selected based on comprehensive preliminary Information Retrieval evaluation studies. These studies, documented in `Information_Retrieval.ipynb` and `ir_evaluation_results.csv`, evaluated 13 different embedding approaches on the target dataset.

**Key Findings from Preliminary Studies:**

1. **TF-IDF + Word2Vec Combination**: Achieved the best performance (MRR: 0.502, NDCG: 0.454, F1: 0.232), outperforming dense transformer models like SBERT (MRR: 0.399, F1: 0.168) and MPNet (MRR: 0.439, F1: 0.178).

2. **Why Sparse Methods Excel on This Dataset**:
   - **High lexical repetition**: Technical terms appear frequently → TF-IDF detects rare tokens perfectly
   - **Technical terminology**: Domain-specific vocabulary → TF-IDF matches pages based on term overlap
   - **OCR distortions**: Scanned documents with character errors → Sparse methods are more robust
   - **Short chunks**: 600-character fragments → Word2Vec's local window co-occurrence works well
   - **Page-based relevance**: Relevance = "same page" (structural, not semantic) → Term overlap is highly effective

3. **Why Dense Models (SBERT) Underperform**:
   - **Semantic similarity collapse**: All chunks appear similar in dense space, reducing discriminative power
   - **OCR noise amplification**: Dense embeddings degrade with character-level errors
   - **Over-smooth semantic space**: Fine-grained distinctions (page boundaries) are lost
   - **Mismatch with relevance**: Dense models optimize for semantic similarity, but relevance here is structural

4. **Why All Three Methods Are Combined**:
   - **Complementary strengths**: TF-IDF (exact matching), Word2Vec (local semantics), SBERT (contextual understanding)
   - **Query diversity**: Different queries benefit from different methods
   - **Robustness**: Reduces failure modes when one method underperforms
   - **Hybrid fusion**: RRF or weighted sum intelligently combines rankings

5. **Cross-Encoder Addition**:
   - Applied after initial retrieval for higher precision
   - Sees query and passage together for accurate relevance scoring
   - Enables fine-grained ranking of top candidates
   - Supports cross-modal reranking (CLIP + cross-encoder) for image chunks
   - Provides normalized scores for relevance threshold filtering (prevents hallucinations)

### Data Classes

#### **`RetrievalArtifacts`**
Container for all retrieval components:
- `chunks`: List of all document chunks
- `tfidf_vectorizer`: TF-IDF vectorizer
- `tfidf_matrix`: TF-IDF sparse matrix
- `w2v_model`: Word2Vec model (optional)
- `w2v_doc_embs`: Word2Vec document embeddings
- `sbert_model`: SentenceTransformer model
- `sbert_doc_embs`: SBERT document embeddings
- `faiss_index`: FAISS index for fast search
- `clip_model`: CLIP model for vision-text (optional)
- `clip_image_embs`: CLIP image embeddings (optional)

### Functions

#### **`_normalize(scores: np.ndarray)`**
- **Purpose**: Min-max normalization to [0, 1] range
- **Handles**: Edge case where all scores are identical
- **Returns**: Normalized scores

#### **`reciprocal_rank_fusion(rankings: List[np.ndarray], k: int)`**
- **Purpose**: Combine multiple ranked lists using RRF
- **Formula**: `score = sum(1 / (k + rank))` for each ranking
- **Parameters**:
  - `rankings`: List of arrays containing document indices in ranked order
  - `k`: RRF constant (default: 60, typical range: 20-100)
- **Returns**: Combined RRF scores for all documents
- **Use case**: When score distributions differ significantly across methods

#### **`hybrid_retrieval(query, artifacts, top_k, use_rrf)`**
- **Purpose**: Main hybrid retrieval function implementing the three-method approach validated in preliminary IR studies
- **Research Basis**: Combines TF-IDF, Word2Vec, and SBERT based on evaluation showing TF-IDF+Word2Vec outperforms SBERT alone, but all three together provide complementary strengths
- **Process**:
  1. **TF-IDF Retrieval**:
     - Compute cosine similarity
     - Get ranking
  2. **Word2Vec Retrieval**:
     - Generate query embedding (average word vectors)
     - Compute cosine similarity with document embeddings
     - Get ranking
  3. **SBERT Retrieval**:
     - Encode query with SBERT
     - Use FAISS for fast similarity search
     - Get ranking
  4. **CLIP Retrieval** (if available):
     - Encode query text with CLIP
     - Compute similarity with image embeddings
     - Get ranking
  5. **Fusion**:
     - **If RRF**: Combine rankings using reciprocal rank fusion
     - **If Weighted Sum**: Normalize scores and compute weighted sum
  6. **Normalization**: Normalize component scores for display
- **Returns**: Tuple of:
  - `top_idxs`: Top-K document indices
  - `top_scores`: Corresponding scores
  - `component_scores`: Dict with individual method scores

#### **`cross_encoder_rerank(query, candidate_texts, top_k, cross_encoder_model, artifacts, candidate_chunk_indices)`**
- **Purpose**: Rerank candidates using cross-encoder with cross-modal support
- **Research Basis**: Cross-encoder reranking was added to improve precision on top candidates after preliminary studies showed initial retrieval could be further refined. The model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is specifically trained for passage reranking.
- **Process**:
  1. **Cross-Encoder Scoring**:
     - Create (query, passage) pairs
     - Score each pair using cross-encoder model
  2. **Cross-Modal Boosting** (if CLIP available):
     - Identify image chunks in candidates
     - Compute CLIP similarity between query and images
     - Boost cross-encoder scores: `0.7 * cross_encoder + 0.3 * clip_sim * 5.0`
  3. **Normalization**:
     - Normalize scores to [0, 1] for threshold comparison
  4. **Selection**:
     - Sort by score (descending)
     - Return top-K
- **Returns**: Tuple of:
  - `order`: Indices of top-K candidates (sorted)
  - `raw_scores`: Raw cross-encoder scores
  - `normalized_scores`: Normalized scores [0, 1]

**Why Cross-Encoder?**
- More accurate than bi-encoder (SBERT) because it sees query and passage together
- Better at understanding query-passage relationship
- Slower but more precise (used only on top candidates)

---

## `pipeline.py`

**Purpose**: Main orchestration - coordinates all pipeline stages.

### Functions

#### **`_save_chunks(chunks: List[dict])`**
- **Purpose**: Save processed chunks to JSON
- **Location**: `data/processed/processed_chunks.json`

#### **`_load_chunks()`**
- **Purpose**: Load saved chunks from JSON
- **Returns**: List of chunks or None if file doesn't exist

#### **`load_existing_artifacts()`**
- **Purpose**: Load all cached pipeline artifacts
- **Checks**: Verifies all required files exist
- **Loads**:
  - Chunks from JSON
  - TF-IDF vectorizer and matrix
  - Word2Vec embeddings (if available)
  - SBERT embeddings and FAISS index
  - CLIP embeddings (if available)
- **Returns**: `RetrievalArtifacts` object or None if files missing

#### **`build_pipeline_and_index(rebuild: bool)`**
- **Purpose**: Main pipeline builder - orchestrates entire process
- **Process**:
  1. **Check Cache** (if `rebuild=False`):
     - Try to load existing artifacts
     - Return early if successful
  2. **Ingestion**:
     - Call `ingest_documents()` to extract content
  3. **Chunking**:
     - Call `chunk_documents()` to create chunks
     - Save chunks to JSON
  4. **TF-IDF**:
     - Build vectorizer and matrix
     - Save to disk
  5. **Word2Vec**:
     - Load model (if available)
     - Generate embeddings
     - Save to disk
  6. **SBERT & FAISS**:
     - Generate dense embeddings
     - Build FAISS index
     - Save to disk
  7. **CLIP** (for images):
     - Load CLIP model
     - Extract image paths from chunks
     - Encode images
     - Create embedding array (zero-padded for non-image chunks)
     - Save to disk
  8. **Return**: `RetrievalArtifacts` with all components

**Caching Strategy**:
- All artifacts saved to `data/processed/`
- Subsequent runs load from cache (fast)
- Rebuild only when needed (new documents, parameter changes)

---

##  `evaluation.py`

**Purpose**: Information Retrieval evaluation metrics.

### Functions

#### **`build_label_matrix(chunks: List[Dict])`**
- **Purpose**: Create binary label matrix for evaluation
- **Process**:
  - Groups chunks by page number
  - Creates binary matrix: `labels[chunk_idx, page_idx] = 1` if chunk belongs to page
- **Returns**: NumPy array of shape `(n_chunks, n_pages)`
- **Use case**: Define "relevant" documents for a query (same page = relevant)

#### **`calculate_ir_metrics_for_query(similarities, query_label_indices, labels_mat, k)`**
- **Purpose**: Compute comprehensive IR metrics for a single query
- **Parameters**:
  - `similarities`: Ranking scores for all documents
  - `query_label_indices`: Indices of query's representative documents
  - `labels_mat`: Binary label matrix
  - `k`: Top-K cutoff (default: 5)
- **Metrics Computed**:
  1. **MRR** (Mean Reciprocal Rank):
     - `1 / rank` of first relevant document in top-K
     - Measures how quickly relevant docs appear
  2. **NDCG@K** (Normalized Discounted Cumulative Gain):
     - Discounted gain based on position
     - Normalized by ideal DCG
     - Measures ranking quality
  3. **NDCG** (Overall):
     - Same as NDCG@K but for top 100
  4. **Precision/Recall/F1**:
     - Overall metrics (cutoff = 2 * |relevant_set|)
  5. **Precision@K / Recall@K / F1@K**:
     - Top-K specific metrics
- **Returns**: Dict with all metric values

**Evaluation Strategy**:
- Uses page-based relevance (chunks from same page are relevant to each other)
- Can be extended to use manual relevance judgments

---

##  Module Interactions

### Data Flow

```
1. config.py
   ↓ (provides paths, constants)
   
2. ingestion.py
   ↓ (extracts: pages, tables, images)
   
3. chunking.py
   ↓ (creates unified chunks)
   
4. embeddings.py
   ↓ (generates: TF-IDF, W2V, SBERT, CLIP)
   
5. retrieval.py
   ↓ (hybrid retrieval + reranking)
   
6. pipeline.py
   ↓ (orchestrates all stages)
   
7. app.py / cli.py
   ↓ (user interface)
```

### Dependencies

- **`config.py`**: Used by all modules (paths, constants)
- **`ingestion.py`**: Independent (only uses external libraries)
- **`chunking.py`**: Depends on `ingestion.py` output
- **`embeddings.py`**: Depends on `chunking.py` output, uses `config.py`
- **`retrieval.py`**: Depends on `embeddings.py` output, uses `config.py`
- **`pipeline.py`**: Orchestrates all modules
- **`evaluation.py`**: Independent utility (can be used separately)

### Key Design Patterns

1. **Modularity**: Each module has single responsibility
2. **Caching**: All expensive operations cached to disk
3. **Error Handling**: Graceful fallbacks (e.g., Word2Vec optional)
4. **Configuration**: Centralized in `config.py`
5. **Type Hints**: Full type annotations for clarity
6. **Data Classes**: Structured data containers (`RetrievalArtifacts`)

---

##  Usage Examples

### Building Pipeline Programmatically

```python
from rag_pipeline.pipeline import build_pipeline_and_index

# Load existing or build new
artifacts = build_pipeline_and_index(rebuild=False)

# Force rebuild
artifacts = build_pipeline_and_index(rebuild=True)
```

### Custom Retrieval

```python
from rag_pipeline.retrieval import hybrid_retrieval

# Use RRF
candidate_idxs, scores, comp_scores = hybrid_retrieval(
    query="Your question",
    artifacts=artifacts,
    use_rrf=True
)

# Use weighted sum
candidate_idxs, scores, comp_scores = hybrid_retrieval(
    query="Your question",
    artifacts=artifacts,
    use_rrf=False
)
```

### Evaluation

```python
from rag_pipeline.evaluation import build_label_matrix, calculate_ir_metrics_for_query
from rag_pipeline.retrieval import hybrid_retrieval

# Build labels
labels = build_label_matrix(artifacts.chunks)

# Get retrieval scores
_, _, comp_scores = hybrid_retrieval(query, artifacts)
similarities = comp_scores["fused"]

# Calculate metrics
query_idx = 0  # Representative chunk index
metrics = calculate_ir_metrics_for_query(
    similarities, 
    [query_idx], 
    labels, 
    k=5
)
print(f"MRR: {metrics['mrr']}, NDCG@5: {metrics['ndcg_at_k']}")
```

---

##  Extension Points

### Adding New Retrieval Method

1. Add embedding generation in `embeddings.py`
2. Add retrieval logic in `retrieval.py` `hybrid_retrieval()`
3. Update `RetrievalArtifacts` dataclass
4. Update `pipeline.py` to build new embeddings
5. Add weight in `config.py` `HybridWeights`

### Adding New Document Type

1. Add extraction function in `ingestion.py`
2. Update `chunking.py` to handle new type
3. Update `ingest_documents()` to call new extractor

### Custom Evaluation

1. Add new metric function in `evaluation.py`
2. Integrate into evaluation workflow
3. Export results to CSV/JSON

---



# rag-retrieval-optimization
A hands-on RAG engineering project exploring retrieval accuracy, latency, and performance optimization across 20-30k pages of complex pdf's and regulatory documents. The project documents the journey from baseline retrieval to performance profiling, candidate recall evaluation, and accuracy–latency trade-offs, with implementation and benchmarking.

## 🚀 Architecture Evolution Summary

| Version | File | Key Innovations & Improvements | Retrieval Strategy |
| :--- | :--- | :--- | :--- |
| **V1** | [`1. baseline_rag.py`](1.%20baseline_rag.py) | • Basic PDF text extraction<br>• Fixed-size word sliding window (500 words, 75 overlap)<br>• Single dense vector search | Pure Dense FAISS (`all-MiniLM-L6-v2`) |
| **V2** | [`2. rag_section_aware_BM25+Hybrid.py`](2.%20rag_section_aware_BM25+Hybrid.py) | • Regex-based heading and clause detection (`is_heading`)<br>• Section-aware chunking preserving regulatory context<br>• Dual indexing: FAISS (Dense) + BM25 (Sparse) | Hybrid Dense + BM25 with Reciprocal Rank Fusion (RRF) |
| **V3** | [`3. rag_section_aware_BM25+Hybrid_latency copy.py`](3.%20rag_section_aware_BM25+Hybrid_latency%20copy.py) | • In-memory index caching (avoids re-reading index files)<br>• Fine-grained `RetrievalProfiler` measuring per-stage micro-latencies (Embedding, Dense, BM25, RRF) | Hybrid Dense + BM25 with Latency Profiling |
| **V4** | [`4. rag_improvedBM25+Reranking.py`](4.%20rag_improvedBM25+Reranking.py) | • **Two-Stage Retrieval Architecture**<br>• `np.argpartition` fast BM25 candidate selection<br>• Cross-Encoder Reranking (`ms-marco-MiniLM-L-6-v2`)<br>• Candidate Recall@20 vs Final Top-5 evaluation<br>• Strict Banking Compliance System Prompt | Hybrid RRF Candidate Generation (Top-10) + Cross-Encoder Reranking (Top-5) |

---


---

## 🔍 Detailed Version-by-Version Improvements

### 1. Baseline RAG (`1. baseline_rag.py`)
- **Chunking**: Naive fixed-size chunking (500 words with 75 words overlap). Disregards document structure and splits regulatory clauses across arbitrary boundaries.
- **Retrieval**: Pure Dense FAISS `IndexFlatIP` retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
- **Limitation**: Suffers from vocabulary mismatch on exact regulatory terms, clause numbers, acronyms, and circular references.

### 2. Section-Aware + BM25 Hybrid (`2. rag_section_aware_BM25+Hybrid.py`)
- **Heading Detection**: Added regular expressions to recognize regulatory structure (`\d+\.\d+`, `Annex`, `Appendix`, `Chapter`, `Section`, `ALL CAPS`).
- **Section Chunks**: Chunks maintain section titles in their metadata and text boundaries.
- **Hybrid Retrieval**: Combines Dense Vector Search (FAISS) with Lexical Search (`rank_bm25.BM25Okapi`).
- **RRF Fusion**: Fuses rankings using weighted Reciprocal Rank Fusion ($w_{\text{dense}}=0.6, w_{\text{bm25}}=0.4, k=60$).

### 3. Latency Profiling & In-Memory Indexing (`3. rag_section_aware_BM25+Hybrid_latency copy.py`)
- **Index Caching**: Reads index and chunk files once into memory at application initialization.
- **Latency Profiler**: Introduces `RetrievalProfiler` to instrument and log:
  - Query Embedding Latency
  - Dense Search Latency
  - BM25 Scoring Latency
  - RRF Fusion Latency
  - Statistical latency breakdown (Min, Avg, P50, P90, P95, P99, Max).

### 4. Cross-Encoder Reranking & BM25 Optimization (`4. rag_improvedBM25+Reranking.py`)
- **BM25 Optimization**: Switched full sorting to `np.argpartition` for candidate selection ($O(N)$ vs $O(N \log N)$), slashing sparse retrieval latency.
- **Cross-Encoder Reranking**: Utilizes `cross-encoder/ms-marco-MiniLM-L-6-v2` to compute deep query-document cross-attention over the top 10 hybrid candidates.
- **Candidate Recall Tracking**: Evaluates candidate pool recall (`candidate_recall_at_20` ≈ 98.95%) before reranking to verify zero retrieval bottleneck.
- **Strict Compliance Prompt**: Specialized banking prompt requiring exact source + page citations and fallback rules to avoid hallucinations.

---

## 📊 Key Evaluation Metrics

| Metric | Baseline V1 | Improved V4 (Hybrid + Reranker) |
| :--- | :--- | :--- |
| **Candidate Recall@20** | N/A (single stage) | **98.95%** |
| **Hit Rate@5** | 93.0% | **93.0%** (with higher relevance quality) |
| **MRR@5** | 0.894 | **0.872** |
| **Correct Page Accuracy** | 73.0% | **78.0% (+5% gain)** |
| **Retrieval Stage** | Single dense vector search | Two-stage Hybrid RRF + Cross-Encoder |
| **Keyword Precision** | Low (semantic only) | **High (BM25 + Cross-Encoder)** |

---

## 🛠️ Quickstart

### 1. Requirements
- Python 3.10+
- Ollama with `llama3.1:8b` installed and running locally
- Dependencies: `sentence-transformers`, `faiss-cpu`, `rank-bm25`, `pypdf`, `numpy`, `tabulate`

### 2. Run the Pipelines
```bash
# 1. Run Baseline RAG
python "1. baseline_rag.py"

# 2. Run Section-Aware Hybrid RAG
python "2. rag_section_aware_BM25+Hybrid.py"

# 3. Run Hybrid RAG with Latency Profiler
python "3. rag_section_aware_BM25+Hybrid_latency copy.py"

# 4. Run Final Optimized Pipeline (Hybrid + Reranker + Profiler)
python "4. rag_improvedBM25+Reranking.py"
```

### 3. Interactive Menu Options
When executing any file, you can choose:
- `1`: **Build / Rebuild Index** (Processes PDFs from `data/pdfs/` and creates `index/`)
- `2`: **Ask Questions (Interactive CLI Mode)**
- `3`: **Run Automated Evaluation** (Tests against `evaluation_questions.json` and writes CSV + JSON summaries into `evaluation/`)

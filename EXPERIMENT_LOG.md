
# RAG Surrogate Optimization — Experiment Log

## Research objective

Study sample-efficient surrogate modeling for ranking and
optimizing RAG configurations using limited actual evaluations.

## Dataset

- Dataset: HotpotQA distractor validation
- Fixed questions: 100
- Sampling seed: 42
- Corpus documents: 997
- Unique supporting titles: 200
- Supporting-title coverage in corpus: 100%
- Chunk size: 256
- Chunk overlap: 32
- Number of chunks: 1039

## Environment

- Python: 3.12
- PyTorch: 2.5.1+cu121
- Transformers: 4.46.3
- Sentence Transformers: 3.3.1
- Hugging Face Hub: 0.26.5
- Datasets: 3.2.0
- GPU: NVIDIA RTX 3050 Laptop GPU
- CUDA available: True

## Experiment 1: BM25 retrieval

- Questions: 100
- Metric: Supporting-title Recall@5
- Result: 0.72

## Experiment 2: Dense retrieval

- Embedding model: all-MiniLM-L6-v2
- Vector index: FAISS
- Questions: 100
- Metric: Supporting-title Recall@5
- Result: 0.79

## Experiment 3: Dense + Cross-Encoder reranking

- Initial retriever: Dense
- Candidate pool: 20
- Final results: 5
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- Questions: 100
- Dense candidate Recall@20: 0.955
- Dense Recall@5: 0.79
- Reranked Recall@5: 0.88
- Mean reranking latency: 0.0728 seconds

## Important limitations

- Current corpus uses candidate contexts from the selected HotpotQA questions,
  not the full Wikipedia collection.
- Retrieval metrics measure supporting-title coverage, not exact sentence
  retrieval or generated answer correctness.
- Reranking latency is not total end-to-end RAG latency.
- Results are from the current pilot experiment and need further validation.

## Pending experiments

- BM25 + Cross-Encoder reranking
- LLM answer generation
- Answer quality evaluation
- Token and end-to-end latency measurements
- SciFact integration
- Configuration-space experiments
- Surrogate model comparison
- Configuration optimization
- BM25 Recall@5: 0.72
- BM25 Candidate Recall@20: 0.95
- BM25 + reranker Recall@5: 0.865
- Average reranking latency: 0.0668 seconds
Also add the four-way comparison:
Configuration	Recall@5
BM25	0.720
Dense	0.790
BM25 + reranker	0.865
Dense + reranker	0.880

from src.rag.chunking import chunk_documents
from src.rag.bm25_retriever import BM25Retriever
from src.rag.dense_retriever import DenseRetriever
from src.rag.reranker import CrossEncoderReranker


class RAGPipeline:

    def __init__(self, config, documents):

        self.config = config
        self.documents = documents

        required_keys = [
            "chunk_size",
            "chunk_overlap",
            "retriever",
            "top_k",
            "reranker",
            "similarity_threshold",
            "context_budget"
        ]

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing configuration key: {key}")

        if config["chunk_size"] <= 0:
            raise ValueError("chunk_size must be positive")

        if not 0 <= config["chunk_overlap"] < config["chunk_size"]:
            raise ValueError("Invalid chunk_overlap")

        if config["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        if config["retriever"] not in ["bm25", "dense"]:
            raise ValueError("Retriever must be 'bm25' or 'dense'")

        if not 0.0 <= config["similarity_threshold"] <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if config["context_budget"] <= 0:
            raise ValueError("context_budget must be positive")

        self.use_reranker = config["reranker"]

        print("Creating document chunks...")

        self.chunks = chunk_documents(
            documents,
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"]
        )

        if not self.chunks:
            raise ValueError("No chunks were generated.")

        print(f"Created {len(self.chunks)} chunks.")

        print(f"Initializing {config['retriever']} retriever...")

        if config["retriever"] == "bm25":
            self.retriever = BM25Retriever(self.chunks)
        else:
            self.retriever = DenseRetriever(self.chunks)

        self.reranker = None

        if self.use_reranker:
            self.reranker = CrossEncoderReranker()

        self.last_retrieval_stats = {}

        print("RAG pipeline initialized successfully.")

    def normalize_scores(self, candidates):

        if not candidates:
            return []

        scores = [float(doc["score"]) for doc in candidates]

        if self.config["retriever"] == "dense":
            # Dense scores are cosine similarities in [-1, 1].
            normalized = [
                max(0.0, min(1.0, (score + 1.0) / 2.0))
                for score in scores
            ]

        else:
            # BM25 scores are normalized within this query's
            # retrieved candidate set.
            min_score = min(scores)
            max_score = max(scores)

            if max_score - min_score < 1e-12:
                normalized = [1.0] * len(scores)
            else:
                normalized = [
                    (score - min_score) / (max_score - min_score)
                    for score in scores
                ]

        results = []

        for document, raw_score, norm_score in zip(
            candidates, scores, normalized
        ):
            result = document.copy()
            result["raw_retrieval_score"] = raw_score
            result["normalized_retrieval_score"] = norm_score
            results.append(result)

        return results

    def retrieve(self, question):

        retrieval_k = self.config.get(
            "retrieval_k",
            max(20, self.config["top_k"])
        )

        candidates = self.retriever.retrieve(
            question,
            top_k=retrieval_k
        )

        scored_candidates = self.normalize_scores(candidates)

        threshold = self.config["similarity_threshold"]

        filtered_candidates = [
            doc for doc in scored_candidates
            if doc["normalized_retrieval_score"] >= threshold
        ]

        if self.reranker is not None:
            retrieved_documents = self.reranker.rerank(
                question,
                filtered_candidates,
                top_k=self.config["top_k"]
            )
        else:
            retrieved_documents = filtered_candidates[:self.config["top_k"]]

        self.last_retrieval_stats = {
            "candidate_count": len(candidates),
            "filtered_count": len(filtered_candidates),
            "returned_count": len(retrieved_documents),
            "similarity_threshold": threshold
        }

        return retrieved_documents

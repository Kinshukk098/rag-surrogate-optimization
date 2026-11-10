
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
            "reranker"
        ]

        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing configuration key: {key}")

        if config["chunk_size"] <= 0:
            raise ValueError("chunk_size must be positive")

        if config["chunk_overlap"] < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if config["chunk_overlap"] >= config["chunk_size"]:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        if config["top_k"] <= 0:
            raise ValueError("top_k must be positive")

        if config["retriever"] not in ["bm25", "dense"]:
            raise ValueError("Retriever must be 'bm25' or 'dense'")

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

        print("RAG pipeline initialized successfully.")

    def retrieve(self, question):

        if self.use_reranker:
            retrieval_k = self.config.get("retrieval_k", 20)
        else:
            retrieval_k = self.config["top_k"]

        candidates = self.retriever.retrieve(
            question,
            top_k=retrieval_k
        )

        if self.reranker is not None:
            return self.reranker.rerank(
                question,
                candidates,
                top_k=self.config["top_k"]
            )

        return candidates[:self.config["top_k"]]

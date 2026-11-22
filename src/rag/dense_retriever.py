import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(
        self,
        documents,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.documents = documents

                  # Automatically select GPU when available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding model on: {self.device}")

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

        print("Embedding model device:", self.model.device)

        print("Generating document embeddings...")
        texts = [doc["text"] for doc in documents]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

        print(f"Indexed {len(documents)} documents.")

    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            doc = self.documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

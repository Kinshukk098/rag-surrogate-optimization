
import torch
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading reranker on: {self.device}")

        self.model = CrossEncoder(
            model_name,
            device=self.device,
            max_length=512
        )

        print(f"Reranker device: {self.device}")

    def rerank(self, query, documents, top_k=5):

        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        ranked_results = []

        for document, score in zip(documents, scores):
            result = document.copy()
            result["rerank_score"] = float(score)
            ranked_results.append(result)

        ranked_results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return ranked_results[:top_k]

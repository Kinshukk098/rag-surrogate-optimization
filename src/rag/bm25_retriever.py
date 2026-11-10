
import re
from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, chunks):
        self.chunks = chunks

        self.tokenized_chunks = [
            self.tokenize(chunk["text"])
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_chunks)

    @staticmethod
    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    def retrieve(self, query, top_k=5):

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        indices = scores.argsort()[::-1][:top_k]

        return [
            {
                **self.chunks[i],
                "score": float(scores[i])
            }
            for i in indices
        ]

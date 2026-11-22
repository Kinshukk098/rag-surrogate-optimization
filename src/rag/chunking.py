
from transformers import AutoTokenizer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load tokenizer once, not every time chunk_documents() is called.
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    model_max_length=100000
)


def chunk_documents(documents, chunk_size=256, chunk_overlap=32):

    if chunk_overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    if chunk_size <= 0 or chunk_overlap < 0:
        raise ValueError("Invalid chunk size or overlap")

    chunks = []
    step = chunk_size - chunk_overlap

    for doc_id, document in enumerate(documents):

        # Tokenize without adding special tokens.
        token_ids = tokenizer.encode(
            document["text"],
            add_special_tokens=False
        )

        for start in range(0, len(token_ids), step):

            token_chunk = token_ids[start:start + chunk_size]

            if not token_chunk:
                break

            chunk_text = tokenizer.decode(
                token_chunk,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            chunks.append({
                "chunk_id": len(chunks),
                "doc_id": doc_id,
                "title": document["title"],
                "text": chunk_text
            })

            if start + chunk_size >= len(token_ids):
                break

    return chunks

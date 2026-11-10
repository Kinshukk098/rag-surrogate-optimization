
import json

from src.rag.chunking import chunk_documents, tokenizer


with open("data/processed/hotpotqa_corpus.json", encoding="utf-8") as f:
    documents = json.load(f)

chunks = chunk_documents(
    documents,
    chunk_size=256,
    chunk_overlap=32
)

print("Original documents:", len(documents))
print("Generated chunks:", len(chunks))

print("\nCHUNK DETAILS")

for chunk in chunks[:5]:
    token_count = len(
        tokenizer.encode(
            chunk["text"],
            add_special_tokens=False
        )
    )

    print("\nTitle:", chunk["title"])
    print("Chunk ID:", chunk["chunk_id"])
    print("Token count:", token_count)
    print("Text:", chunk["text"][:150])

print("\nChunking test completed.")

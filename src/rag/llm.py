
import os
import time

from dotenv import load_dotenv
from groq import Groq

from src.rag.chunking import tokenizer


load_dotenv()

MODEL_NAME = "qwen/qwen3.8-27b"


class GroqAnswerGenerator:

    def __init__(
        self,
        model_name=MODEL_NAME,
        temperature=0,
        max_completion_tokens=256,
        seed=42
    ):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env")

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.seed = seed

    def build_context(self, documents, context_budget):

        if context_budget <= 0:
            raise ValueError("context_budget must be positive")

        selected_blocks = []
        selected_titles = []

        for document in documents:

            title = document.get("title", "")
            text = document.get("text", "")

            if not text:
                continue

            prefix = (
                f"Document {len(selected_blocks) + 1}\n"
                f"Title: {title}\n"
                "Content: "
            )

            text_ids = tokenizer.encode(
                text,
                add_special_tokens=False
            )

            if not text_ids:
                continue

            # Find the largest text prefix that keeps the COMPLETE
            # assembled context within the configured token budget.
            low = 0
            high = len(text_ids)
            best_block = None

            while low <= high:

                mid = (low + high) // 2

                selected_text = tokenizer.decode(
                    text_ids[:mid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                candidate_block = prefix + selected_text

                candidate_context = "\n\n".join(
                    selected_blocks + [candidate_block]
                )

                candidate_tokens = len(
                    tokenizer.encode(
                        candidate_context,
                        add_special_tokens=False
                    )
                )

                if candidate_tokens <= context_budget:
                    best_block = candidate_block
                    low = mid + 1
                else:
                    high = mid - 1

            if best_block is None:
                break

            selected_blocks.append(best_block)
            selected_titles.append(title)

            final_context = "\n\n".join(selected_blocks)

            final_token_count = len(
                tokenizer.encode(
                    final_context,
                    add_special_tokens=False
                )
            )

            if final_token_count >= context_budget:
                break

            # If this document was truncated, later documents
            # cannot be added without changing the intended order.
            selected_text_ids = tokenizer.encode(
                best_block[len(prefix):],
                add_special_tokens=False
            )

            if len(selected_text_ids) < len(text_ids):
                break

        context = "\n\n".join(selected_blocks)

        context_tokens = len(
            tokenizer.encode(
                context,
                add_special_tokens=False
            )
        )

        return (
            context,
            context_tokens,
            len(selected_blocks),
            selected_titles
        )

    def generate(self, question, documents, context_budget):

        if not documents:
            raise ValueError("No retrieved documents provided.")

        # 1. Measure context construction separately.
        context_start_time = time.perf_counter()

        (
            context,
            context_tokens,
            documents_used,
            context_document_titles
        ) = self.build_context(
            documents,
            context_budget
        )

        context_construction_latency = (
            time.perf_counter() - context_start_time
        )

        if not context.strip():
            raise ValueError(
                "No context fits within the configured budget."
            )

        # 2. Prepare the LLM messages.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise question-answering system. "
                    "Answer the question using the supplied context only. "
                    "Return only the shortest possible answer, usually "
                    "a name, date, number, location, or short phrase. "
                    "Do not provide explanations, reasoning, or introductory "
                    "phrases. Do not repeat the question. "
                    "If the answer genuinely cannot be determined from "
                    "the context, respond exactly with: "
                    "Insufficient information."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    "Answer:"
                )
            }
        ]

        # 3. Measure Groq API generation latency.
        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
            seed=self.seed
        )

        generation_latency = time.perf_counter() - start_time

        usage = response.usage

        # 4. Return answer and experiment measurements.
        return {
            "answer": response.choices[0].message.content or "",
            "model": self.model_name,

            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,

            "context_tokens": context_tokens,
            "context_documents_used": documents_used,
            "context_document_titles": context_document_titles,
            "context_text": context,

            "context_construction_latency_seconds": (
                context_construction_latency
            ),

            "generation_latency_seconds": generation_latency
        }

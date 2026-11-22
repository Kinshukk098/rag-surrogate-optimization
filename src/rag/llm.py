
import time

from openai import OpenAI

from src.rag.chunking import tokenizer


MODEL_NAME = "qwen2.5:3b"


class GroqAnswerGenerator:

    def __init__(
        self,
        model_name=MODEL_NAME,
        temperature=0,
        max_completion_tokens=256,
        seed=42
    ):
        # Ollama's local OpenAI-compatible API.
        # The API key is a placeholder; no real key is required.
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            timeout=300.0
        )

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

            current_context = "\n\n".join(selected_blocks)
            separator = "\n\n" if selected_blocks else ""

            low = 1
            high = len(text_ids)
            best_text = ""

            while low <= high:

                mid = (low + high) // 2

                selected_text = tokenizer.decode(
                    text_ids[:mid],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                candidate_context = (
                    current_context
                    + separator
                    + prefix
                    + selected_text
                )

                candidate_tokens = len(
                    tokenizer.encode(
                        candidate_context,
                        add_special_tokens=False
                    )
                )

                if candidate_tokens <= context_budget:
                    best_text = selected_text
                    low = mid + 1
                else:
                    high = mid - 1

            if not best_text:
                continue

            selected_block = prefix + best_text

            selected_blocks.append(selected_block)
            selected_titles.append(title)

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

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_completion_tokens,
            seed=self.seed
        )

        generation_latency = time.perf_counter() - start_time

        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = usage.total_tokens if usage else (
            input_tokens + output_tokens
        )

        return {
            "answer": response.choices[0].message.content or "",
            "model": self.model_name,

            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,

            "context_tokens": context_tokens,
            "context_documents_used": documents_used,
            "context_document_titles": context_document_titles,
            "context_text": context,

            "context_construction_latency_seconds": (
                context_construction_latency
            ),

            "generation_latency_seconds": generation_latency
        }

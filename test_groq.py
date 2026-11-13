
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=api_key)

MODEL = "qwen/qwen3.8-27b"

print("Testing Groq API...")
print("Model:", MODEL)

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "system",
            "content": (
                "Answer questions accurately and concisely. "
                "Do not invent facts."
            )
        },
        {
            "role": "user",
            "content": (
                "Who developed the theory of general relativity? "
                "Answer in one sentence."
            )
        }
    ],
    temperature=0,
    max_completion_tokens=100,
    seed=42
)

answer = response.choices[0].message.content

print("\nGenerated answer:")
print(answer)

print("\nToken usage:")
print("Input tokens:", response.usage.prompt_tokens)
print("Output tokens:", response.usage.completion_tokens)
print("Total tokens:", response.usage.total_tokens)

print("\nAPI test successful!")

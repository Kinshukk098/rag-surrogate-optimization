
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

print("Fetching models available to your Groq account...\n")

models = client.models.list()

for model in models.data:
    print(model.id, "| Owner:", model.owned_by)

print(f"\nTotal models returned: {len(models.data)}")

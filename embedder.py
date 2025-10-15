import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load chunks from previous step
with open("output_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

embedded_chunks = []

# Loop and embed each chunk
for i, text in enumerate(chunks):
    print(f"🔄 Embedding chunk {i+1} of {len(chunks)}...")

    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )

    embedding = response.data[0].embedding

    embedded_chunks.append({
        "id": f"chunk-{i}",
        "text": text,
        "embedding": embedding
    })

# Save embeddings
with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(embedded_chunks, f, indent=2)

print(f"✅ Embeddings generated and saved for {len(embedded_chunks)} chunks.")



import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
region = os.getenv("PINECONE_ENVIRONMENT")  # should be "us-east-1"

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Define index name and dimensions
index_name = "vuedatavectordb"
embedding_dimension = 1536  # for text-embedding-ada-002

# Delete the index if it already exists (to avoid dimension mismatch)
if index_name in pc.list_indexes().names():
    print(f"🗑️ Deleting old index: {index_name} (wrong dimension)...")
    pc.delete_index(index_name)

# Create the new index with correct dimension
print(f"🛠️ Creating index '{index_name}' with dimension {embedding_dimension}...")
pc.create_index(
    name=index_name,
    dimension=embedding_dimension,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region=region  # your .env should have us-east-1
    )
)

# Connect to the new index
index = pc.Index(index_name)

# Load the embeddings from file
with open("embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Format and upload data to Pinecone
vectors = [
    (item["id"], item["embedding"], {"text": item["text"]})
    for item in data
]

print(f"🔼 Uploading {len(vectors)} vectors to Pinecone...")
index.upsert(vectors=vectors)
print("✅ Upload complete.")

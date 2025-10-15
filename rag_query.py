import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
index_name = os.getenv("INDEX_NAME")

# Set up clients
openai_client = OpenAI(api_key=openai_api_key)
pinecone_client = Pinecone(api_key=pinecone_api_key)
index = pinecone_client.Index(index_name)

# Initialize conversation history
messages = [{"role": "system", "content": "You are a helpful assistant answering questions about VueData, an IT services company."}]

print("🔁 Ask me anything about VueData! Type 'exit' to quit.\n")

while True:
    # Step 1: User input
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Conversation ended.")
        break

    # Step 2: Embed user query
    query_embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding

    # Step 3: Search Pinecone
    search_response = index.query(
        vector=query_embedding,
        top_k=8,
        include_metadata=True
    )

    # Step 4: Extract context from results
    relevant_texts = [match["metadata"]["text"] for match in search_response["matches"]]
    context = "\n\n---\n\n".join(relevant_texts)

    # Step 5: Update messages and get assistant response
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"})

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.3
    )

    assistant_reply = response.choices[0].message.content.strip()
    messages.append({"role": "assistant", "content": assistant_reply})

    # Step 6: Print assistant answer
    print(f"\n🧠 VueData Assistant: {assistant_reply}\n")

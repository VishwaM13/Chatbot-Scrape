import json
import tiktoken

# Load the text content from your renamed file
with open("FinalVueDataWEB.txt", "r", encoding="utf-8") as file:
    full_text = file.read()

# Use OpenAI's tokenizer for ada-002 embeddings
tokenizer = tiktoken.get_encoding("cl100k_base")

# Preprocess: split into paragraphs or sections
paragraphs = [p.strip() for p in full_text.split("\n") if p.strip()]

# Define chunking parameters
MAX_TOKENS = 500
OVERLAP = 50

chunks = []
current_chunk = []
current_length = 0

for para in paragraphs:
    tokens = tokenizer.encode(para)

    # If adding this paragraph keeps us under the token limit
    if current_length + len(tokens) <= MAX_TOKENS:
        current_chunk.append(para)
        current_length += len(tokens)
    else:
        # Finalize the current chunk
        chunk_text = "\n".join(current_chunk)
        chunks.append(chunk_text)

        # Create overlap by taking last OVERLAP tokens
        overlap_tokens = tokenizer.encode(chunk_text)[-OVERLAP:]
        overlap_text = tokenizer.decode(overlap_tokens)

        # Start new chunk with overlap + new paragraph
        current_chunk = [overlap_text, para]
        current_length = len(tokenizer.encode(overlap_text + para))

# Add the last chunk if any
if current_chunk:
    chunks.append("\n".join(current_chunk))

# Save to JSON file
with open("output_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"✅ Chunking complete: {len(chunks)} chunks saved to output_chunks.json")

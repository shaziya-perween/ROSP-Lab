
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from pinecone import Pinecone, ServerlessSpec
import time

# -------------------- Load environment variables --------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY","AIzaSyADX1vF7KyQHS52ygqSmqqyKp5n6U5BEL4")

# -------------------- Check API keys --------------------
print("PINECONE_API_KEY loaded:", bool(PINECONE_API_KEY))
print(" OPENAI_API_KEY loaded:", bool(OPENAI_API_KEY))
print(" GOOGLE_API_KEY loaded:", bool(GOOGLE_API_KEY))

# -------------------- Initialize Pinecone --------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "context-rag"

if "context-rag" in [idx.name for idx in pc.list_indexes()]:
    pc.delete_index("context-rag")
    print("Deleted old index 'context-rag'")


if index_name not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Created Pinecone index: {index_name}")
else:
    print(f" Using existing Pinecone index: {index_name}")

index = pc.Index(index_name)
print(" Connected to Pinecone index")

# -------------------- Initialize models --------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",  # Google embedding model
    google_api_key=GOOGLE_API_KEY
    
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# -------------------- Load sample text --------------------
file_path = "data/sample_docs/example.txt"

if not os.path.exists(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(
            "Contextual search helps improve search accuracy by understanding the meaning of a query. "
            "Machine learning models analyze context to return relevant results."
        )
        print(f" Created sample text file: {file_path}")

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read().strip()

if not text:
    raise ValueError("The text file is empty! Please add content to proceed.")

print(" Loaded text:\n", text)

# -------------------- Split text into clean chunks --------------------
chunks = [chunk.strip() for chunk in text.split(". ") if chunk.strip()]
if not chunks:
    raise ValueError("No valid chunks found in your text.")

print(f" Split text into {len(chunks)} chunks:")
for i, c in enumerate(chunks):
    print(f"  Chunk {i+1}: {c}")

# -------------------- Safe embedding function --------------------
def safe_embed_documents(chunks, retries=5):
    for i in range(retries):
        try:
            return embeddings.embed_documents(chunks)  # batch embedding
        except Exception as e:
            wait_time = 2 ** i
            print(f" Embedding failed (attempt {i+1}/{retries}): {e}")
            time.sleep(wait_time)
    raise Exception("Embedding failed after retries.")

# -------------------- Generate embeddings --------------------
print(" Generating embeddings...")
try:
    chunk_vectors = safe_embed_documents(chunks)
except Exception as e:
    print(" Failed to generate embeddings:", e)
    exit(1)

print(f" Generated {len(chunk_vectors)} embeddings")

# -------------------- Upsert data into Pinecone --------------------
print(" Uploading embeddings to Pinecone...")
for i, (chunk, vec) in enumerate(zip(chunks, chunk_vectors)):
    index.upsert(vectors=[{"id": f"chunk_{i}", "values": vec, "metadata": {"text": chunk}}])
print(f" Uploaded {len(chunks)} chunks to Pinecone.")

# -------------------- Query Pinecone --------------------
query = "What is contextual search?"
query_vector = embeddings.embed_query(query)
results = index.query(vector=query_vector, top_k=3, include_metadata=True)

print("\n Pinecone Search Results:")
if results.get("matches"):
    for match in results["matches"]:
        print(f"→ Score: {match['score']:.3f} | Text: {match['metadata']['text']}")
else:
    print("No matches found.")

# -------------------- Use Gemini to summarize --------------------
context = " ".join([m["metadata"]["text"] for m in results.get("matches", [])])
if context:
    prompt = f"Based on the following context, explain what contextual search means:\n\n{context}"
    response = llm.invoke(prompt)
    print("\n Gemini Response:\n", response.content)
else:
    print("No context available for Gemini to summarize.")

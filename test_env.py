import os
from dotenv import load_dotenv

load_dotenv()

print("PINECONE_API_KEY:", os.getenv("PINECONE_API_KEY"))
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))

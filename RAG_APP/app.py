
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import os

# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL = os.getenv("MODEL")

# --------------------------------------------------
# 2. Connect to Ollama
# --------------------------------------------------

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# --------------------------------------------------
# 3. Read PDF
# --------------------------------------------------

pdf_path = "D:\\Downloaded\\Test.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text() + "\n"

print("PDF loaded successfully.")
print("Characters:", len(text))

# --------------------------------------------------
# 4. Split document into chunks
# --------------------------------------------------

chunk_size = 500

chunks = []

for i in range(0, len(text), chunk_size):
    chunk = text[i:i + chunk_size]
    chunks.append(chunk)

print("Chunks created:", len(chunks))

# --------------------------------------------------
# 5. Create embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# --------------------------------------------------
# 6. Create ChromaDB
# --------------------------------------------------

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)

# --------------------------------------------------
# 7. Create embeddings and store them
# --------------------------------------------------

embeddings = embedding_model.encode(chunks)

for i, chunk in enumerate(chunks):

    collection.upsert(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embeddings[i].tolist()]
    )

print("Document stored in vector database.")

# --------------------------------------------------
# 8. Ask user a question
# --------------------------------------------------

question = input("\nAsk a question about the document: ")

# --------------------------------------------------
# 9. Convert question into embedding
# --------------------------------------------------

question_embedding = embedding_model.encode(
    question
).tolist()

# --------------------------------------------------
# 10. Search vector database
# --------------------------------------------------

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

relevant_chunks = results["documents"][0]

context = "\n\n".join(relevant_chunks)

# --------------------------------------------------
# 11. Send context + question to Ollama
# --------------------------------------------------

prompt = f"""
Answer the question using only the information
provided in the context.

Context:
{context}

Question:
{question}

If the answer is not available in the context,
say "I don't know based on this document."
"""

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# --------------------------------------------------
# 12. Display answer
# --------------------------------------------------

answer = response.choices[0].message.content

print("\nAnswer:")
print(answer)


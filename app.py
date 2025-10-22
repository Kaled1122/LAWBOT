import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PyPDF2 import PdfReader
from openai import OpenAI
import faiss
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------
app = Flask(__name__)
CORS(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# LOAD & INDEX PDF CONTENT
# -----------------------------
def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=800):
    """Split large text into overlapping chunks for vector search."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

print("📘 Loading and indexing LABOR LAW...")
pdf_text = load_pdf_text("LABOR LAW.pdf")
chunks = chunk_text(pdf_text)

# Create embeddings
embeddings = []
for chunk in chunks:
    emb = client.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding
    embeddings.append(emb)

embeddings = np.array(embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(embeddings)

print(f"✅ Indexed {len(chunks)} chunks from LABOR LAW.pdf")

# -----------------------------
# CHAT ENDPOINT
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    user_input = request.json.get("question", "")
    if not user_input:
        return jsonify({"answer": "Please enter a valid question."})

    # Create query embedding
    query_emb = client.embeddings.create(input=user_input, model="text-embedding-3-small").data[0].embedding
    query_emb = np.array(query_emb).astype("float32").reshape(1, -1)

    # Search for top 3 relevant chunks
    distances, indices = index.search(query_emb, k=3)
    retrieved_text = "\n".join([chunks[i] for i in indices[0]])

    # Generate a grounded answer
    prompt = f"""
You are a Saudi Labor Law expert. Answer ONLY using the information below.
If the answer is not found, respond with "The document does not contain this information."

Context:
{retrieved_text}

Question: {user_input}
Answer:
    """

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a legal chatbot limited to the Saudi Labor Law PDF."},
                  {"role": "user", "content": prompt}],
        temperature=0
    )

    answer = completion.choices[0].message.content.strip()
    return jsonify({"answer": answer})

# -----------------------------
# FRONTEND ROUTE
# -----------------------------
@app.route("/")
def serve_frontend():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

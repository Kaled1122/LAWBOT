import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
from openai import OpenAI
import faiss
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# LOAD AND INDEX PDF (with caching)
# -----------------------------
def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=800):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Load cached index or create a new one
if os.path.exists("law_index.faiss") and os.path.exists("law_chunks.pkl"):
    print("⚡ Loading cached index...")
    index = faiss.read_index("law_index.faiss")
    with open("law_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
else:
    print("📘 Indexing LABOR LAW.pdf...")
    pdf_text = load_pdf_text("LABOR LAW.pdf")
    chunks = chunk_text(pdf_text)

    embeddings = []
    for chunk in chunks:
        emb = client.embeddings.create(
            input=chunk, model="text-embedding-3-small"
        ).data[0].embedding
        embeddings.append(emb)

    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)

    faiss.write_index(index, "law_index.faiss")
    with open("law_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Indexed and cached {len(chunks)} chunks.")

# -----------------------------
# ASK ENDPOINT (Adaptive Mode)
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question", "")
    if not user_question.strip():
        return jsonify({"answer": "Please enter a valid question."}), 400

    # Adaptive mode: adjust temperature automatically
    word_count = len(user_question.split())
    if word_count < 8:
        temperature = 0.2
    elif word_count < 20:
        temperature = 0.4
    else:
        temperature = 0.7

    # Search relevant chunks
    query_emb = client.embeddings.create(
        input=user_question, model="text-embedding-3-small"
    ).data[0].embedding
    query_emb = np.array(query_emb).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_emb, k=5)
    context = "\n".join([chunks[i] for i in indices[0]])

    prompt = f"""
You are a Saudi Labor Law expert. Use only the official Saudi Labor Law context below to answer clearly and accurately.

CONTEXT:
{context}

QUESTION:
{user_question}

INSTRUCTIONS:
- Answer directly, in plain language.
- Summarize long clauses if needed.
- If unclear in context, say "Not specified clearly in the law".
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a legal chatbot restricted to Saudi Labor Law."},
            {"role": "user", "content": prompt},
        ],
    )

    answer = completion.choices[0].message.content.strip()
    return jsonify({"answer": answer})

# -----------------------------
# MAIN ENTRY
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

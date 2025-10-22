import os
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
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins (for Vercel)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# LOAD AND INDEX PDF
# -----------------------------
def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=800):
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
# ASK ENDPOINT
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question", "")
    if not user_question.strip():
        return jsonify({"answer": "Please enter a valid question."}), 400

    # Search top chunks
    query_emb = client.embeddings.create(
        input=user_question,
        model="text-embedding-3-small"
    ).data[0].embedding
    query_emb = np.array(query_emb).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_emb, k=5)
    context = "\n".join([chunks[i] for i in indices[0]])

    # Generate legal answer (improved flexible prompt)
    prompt = f"""
You are a Saudi Labor Law expert specializing in explaining and summarizing regulations 
based strictly on the official Saudi Labor Law (provided below as context). 

Your role is to:
- Use the context to form accurate, clear, and practical answers.
- You may paraphrase or summarize the relevant clauses to make them easier to understand.
- If the question is broad, give a general explanation based on the closest related sections.
- If the context clearly lacks an answer, politely say so, and suggest which part of the law may apply.

CONTEXT (from the official Saudi Labor Law):
{context}

QUESTION:
{user_question}

Please provide your answer in a clear, helpful tone:
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a legal chatbot restricted to Saudi Labor Law."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = completion.choices[0].message.content.strip()
    return jsonify({"answer": answer})

# -----------------------------
# MAIN ENTRY
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # ✅ Handles Railway dynamic ports
    app.run(host="0.0.0.0", port=port)

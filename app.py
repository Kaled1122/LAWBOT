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
# LOAD & INDEX PDF (with caching)
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

if os.path.exists("law_index.faiss") and os.path.exists("law_chunks.pkl"):
    print("⚡ Loading cached index...")
    index = faiss.read_index("law_index.faiss")
    with open("law_chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
else:
    print("📘 Indexing LABOR LAW.pdf...")
    pdf_text = load_pdf_text("LABOR LAW.pdf")
    chunks = chunk_text(pdf_text)

    embeddings = [
        client.embeddings.create(input=c, model="text-embedding-3-small").data[0].embedding
        for c in chunks
    ]
    emb_array = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(emb_array)
    faiss.write_index(index, "law_index.faiss")
    with open("law_chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ Indexed and cached {len(chunks)} chunks.")

# -----------------------------
# ASK ENDPOINT (Adaptive + Clean Output)
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"answer": "Please enter a valid question."}), 400

    # Guardrail: refuse questions unrelated to Saudi Labor Law
    off_topic_keywords = [
        "python", "html", "javascript", "football", "movie", "recipe", "ai", "openai",
        "weather", "love", "politics", "game", "health", "music", "device", "chatgpt"
    ]
    if any(k.lower() in user_question.lower() for k in off_topic_keywords):
        return jsonify({"answer": "Sorry, I can only answer questions related to the Saudi Labor Law."})

    # Adaptive mode
    wc = len(user_question.split())
    temperature = 0.2 if wc < 8 else 0.4 if wc < 20 else 0.7

    # Retrieve top chunks
    query_emb = client.embeddings.create(input=user_question, model="text-embedding-3-small").data[0].embedding
    query_emb = np.array(query_emb).astype("float32").reshape(1, -1)
    _, idx = index.search(query_emb, k=5)
    context = "\n".join([chunks[i] for i in idx[0]])

    # Prompt with strict formatting rules
    prompt = f"""
You are a Saudi Labor Law expert.
Answer strictly using the context below.

Rules:
- Only discuss Saudi Labor Law.
- Never use Markdown or asterisks (**).
- Use simple plain text with bullet points or short paragraphs.
- Avoid any decorative characters like *, #, or underscores.
- If the context does not contain an answer, reply with:
  "This topic is not specified clearly in the Saudi Labor Law."

CONTEXT:
{context}

QUESTION:
{user_question}

Provide the answer below:
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You are a factual assistant restricted to the Saudi Labor Law."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = res.choices[0].message.content.strip()

    # Safety clean-up (remove any leftover markdown)
    clean_answer = (
        answer.replace("*", "")
              .replace("_", "")
              .replace("#", "")
              .replace("**", "")
              .replace("•", "-")
    )

    return jsonify({"answer": clean_answer})

# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})

# -----------------------------
# MAIN ENTRY
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

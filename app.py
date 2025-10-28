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
# ASK ENDPOINT (Adaptive + Bulleted Output)
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    user_question = data.get("question", "").strip()
    if not user_question:
        return jsonify({"answer": "Please enter a valid question."}), 400

    # Refuse unrelated topics
    off_topic = [
        "python","html","javascript","ai","football","recipe","music","love",
        "openai","weather","device","politics","movie","game","chatgpt"
    ]
    if any(word in user_question.lower() for word in off_topic):
        return jsonify({"answer": "Sorry, I can only answer questions related to the Saudi Labor Law."})

    # Adaptive response depth
    wc = len(user_question.split())
    temperature = 0.2 if wc < 8 else 0.4 if wc < 20 else 0.7

    # Search top chunks
    query_emb = client.embeddings.create(
        input=user_question, model="text-embedding-3-small"
    ).data[0].embedding
    query_emb = np.array(query_emb).astype("float32").reshape(1, -1)
    _, idx = index.search(query_emb, k=5)
    context = "\n".join([chunks[i] for i in idx[0]])

    # Prompt for plain bullet answers
    prompt = f"""
You are a Saudi Labor Law expert.

Rules:
- Only discuss Saudi Labor Law.
- Write each key point on a new line, starting with a dash (-).
- Do NOT use asterisks, markdown, or numbering.
- Keep text factual and clear.
- If the context lacks an answer, reply:
  "This topic is not specified clearly in the Saudi Labor Law."

CONTEXT:
{context}

QUESTION:
{user_question}

Provide the answer in plain bullet points:
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

    # Clean up output and ensure dash bullets
    cleaned = (
        answer.replace("•", "-")
              .replace("*", "")
              .replace("_", "")
              .replace("#", "")
              .replace("**", "")
              .strip()
    )

    # Ensure each dash bullet starts on a new line
    cleaned = cleaned.replace(". -", ".\n-")

    # Split into lines and rebuild with proper newlines
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    bullet_text = "\n".join(f"- {line.lstrip('-').strip()}" for line in lines)

    return jsonify({"answer": bullet_text})

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

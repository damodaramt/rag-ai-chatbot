from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import os
import json

# LOAD MODEL
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

VECTOR_DB_PATH = "vector_store/db.json"
os.makedirs("vector_store", exist_ok=True)


# EXTRACT TEXT
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    texts = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)

    return texts


# SPLIT TEXT
def split_text(texts, chunk_size=500):
    chunks = []
    for text in texts:
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i+chunk_size])
    return chunks


# CREATE VECTOR DB
def create_vector_db(pdf_files):
    all_data = []

    for pdf in pdf_files:
        texts = extract_text_from_pdf(pdf)
        chunks = split_text(texts)

        for chunk in chunks:
            embedding = model.encode(chunk).tolist()

            all_data.append({
                "text": chunk,
                "embedding": embedding,
                "source": os.path.basename(pdf)
            })

    with open(VECTOR_DB_PATH, "w") as f:
        json.dump(all_data, f)

    print("✅ Vector DB created")


# SEARCH
def search_docs(query, top_k=3):
    if not os.path.exists(VECTOR_DB_PATH):
        return []

    with open(VECTOR_DB_PATH, "r") as f:
        data = json.load(f)

    query_embedding = model.encode(query)

    results = []

    for item in data:
        score = util.cos_sim(query_embedding, item["embedding"]).item()
        results.append((score, item))

    results.sort(reverse=True, key=lambda x: x[0])

    return [r[1] for r in results[:top_k]]

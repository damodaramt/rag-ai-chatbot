from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, util
import os
import json

# ==============================
# LOAD MODEL
# ==============================
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

VECTOR_DB_PATH = "vector_store/db.json"
os.makedirs("vector_store", exist_ok=True)


# ==============================
# EXTRACT TEXT FROM PDF
# ==============================
def extract_text_from_pdf(file_path):
    texts = []

    try:
        reader = PdfReader(file_path)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                texts.append({
                    "page": page_num + 1,
                    "text": text
                })

    except Exception as e:
        print("❌ PDF read error:", str(e))

    return texts


# ==============================
# SPLIT TEXT INTO CHUNKS
# ==============================
def split_text(pages, chunk_size=500, overlap=100):
    chunks = []

    for page in pages:
        text = page["text"]
        page_num = page["page"]

        start = 0
        while start < len(text):
            chunk = text[start:start + chunk_size]

            chunks.append({
                "text": chunk,
                "page": page_num
            })

            start += (chunk_size - overlap)

    return chunks


# ==============================
# CREATE / UPDATE VECTOR DB
# ==============================
def create_vector_db(pdf_files):

    all_data = []

    for pdf in pdf_files:
        print(f"📄 Processing PDF: {pdf}")

        pages = extract_text_from_pdf(pdf)
        chunks = split_text(pages)

        for chunk in chunks:
            embedding = model.encode(chunk["text"]).tolist()

            all_data.append({
                "text": chunk["text"],
                "embedding": embedding,
                "source": os.path.basename(pdf),
                "page": chunk["page"]
            })

    # ✅ FIX: Append instead of overwrite
    if os.path.exists(VECTOR_DB_PATH):
        try:
            with open(VECTOR_DB_PATH, "r") as f:
                existing_data = json.load(f)
        except:
            existing_data = []
    else:
        existing_data = []

    # OPTIONAL: Remove duplicates (same text + source)
    existing_texts = set((d["text"], d["source"]) for d in existing_data)

    new_data = []
    for item in all_data:
        key = (item["text"], item["source"])
        if key not in existing_texts:
            new_data.append(item)

    final_data = existing_data + new_data

    with open(VECTOR_DB_PATH, "w") as f:
        json.dump(final_data, f)

    print(f"✅ Vector DB updated. Total chunks: {len(final_data)}")


# ==============================
# SEARCH DOCUMENTS
# ==============================
def search_docs(query, top_k=3):

    if not os.path.exists(VECTOR_DB_PATH):
        print("⚠️ No vector DB found")
        return []

    with open(VECTOR_DB_PATH, "r") as f:
        data = json.load(f)

    if not data:
        return []

    query_embedding = model.encode(query)

    results = []

    for item in data:
        score = util.cos_sim(query_embedding, item["embedding"]).item()
        results.append((score, item))

    # Sort by similarity
    results.sort(key=lambda x: x[0], reverse=True)

    top_results = [r[1] for r in results[:top_k]]

    return top_results

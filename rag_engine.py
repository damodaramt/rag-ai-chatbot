from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import os

# 🔥 EMBEDDINGS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 🔥 RERANK MODEL (VERY IMPORTANT)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# 🚀 CREATE VECTOR DB
def create_vector_db(pdf_files, save_path="vector_store"):
    all_docs = []

    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = os.path.basename(pdf)

        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("❌ No text found in PDF")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(all_docs)

    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(save_path)

    print("✅ Vector DB created & saved")
    return db


# 🚀 LOAD DB
def load_vector_db(path="vector_store"):
    if not os.path.exists(path):
        return None

    db = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("✅ Vector DB loaded")
    return db


# 🔥 RERANK FUNCTION
def rerank(query, docs):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)

    ranked = list(zip(docs, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in ranked]


# 🔍 SEARCH + RERANK
def search_docs(vector_db, query, k=10, final_k=5):

    if vector_db is None:
        return []

    # Step 1: retrieve more docs
    docs = vector_db.similarity_search(query, k=k)

    # Step 2: rerank
    docs = rerank(query, docs)

    # Step 3: take top final_k
    docs = docs[:final_k]

    results = []

    for doc in docs:
        results.append({
            "content": doc.page_content,
            "page": doc.metadata.get("page", "N/A"),
            "source": doc.metadata.get("source", "Unknown")
        })

    return results

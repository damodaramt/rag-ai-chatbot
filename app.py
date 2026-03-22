from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag_advanced import create_vector_db, search_docs, load_vector_db
import requests
import json
import shutil
import os
import time

# 🔥 RAG IMPORT
from rag_advanced import create_vector_db, search_docs, load_vector_db

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 OLLAMA
OLLAMA_URL = "http://localhost:11434/api/generate"

# 📂 PATHS
UPLOAD_DIR = "uploads"
VECTOR_DB_DIR = "vector_store"
FILES_JSON = "files.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔥 MEMORY (multi-user)
user_sessions = {}

# 🔥 LOAD FILE LIST
def load_files():
    if os.path.exists(FILES_JSON):
        with open(FILES_JSON, "r") as f:
            return json.load(f)
    return []

def save_files(files):
    with open(FILES_JSON, "w") as f:
        json.dump(files, f)

uploaded_files = load_files()

# 🔥 LOAD VECTOR DB
vector_db = None
if os.path.exists(VECTOR_DB_DIR):
    try:
        vector_db = load_vector_db(VECTOR_DB_DIR)
        print("✅ Vector DB loaded from disk")
    except:
        print("⚠️ Failed to load vector DB")

# ✅ Request Model
class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "default_user"

@app.get("/")
def home():
    return {"message": "LLM API running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

# 🔥 USER MEMORY
def get_user_history(user_id):
    if user_id not in user_sessions:
        user_sessions[user_id] = []
    return user_sessions[user_id]

# 🚀 PDF UPLOAD API
@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    global vector_db, uploaded_files

    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files allowed"}

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    uploaded_files.append(file_path)
    save_files(uploaded_files)

    print("📂 Uploaded:", uploaded_files)

    # 🔥 Build + Save DB
    vector_db = create_vector_db(uploaded_files, save_path=VECTOR_DB_DIR)

    return {"message": f"{file.filename} uploaded and processed"}

# 🚀 CHAT API
@app.post("/chat")
def chat(req: ChatRequest):
    global vector_db

    start_time = time.time()

    prompt = req.prompt.strip()
    print("\n📩 USER:", prompt)

    # 🔥 Check Ollama
    try:
        requests.get("http://localhost:11434")
    except:
        return {"error": "⚠️ Ollama not running"}

    # 🔥 USER MEMORY
    chat_history = get_user_history(req.user_id)

    if len(prompt.split()) < 4:
        chat_history.clear()

    chat_history.append({"role": "user", "content": prompt})
    chat_history[:] = chat_history[-6:]

    full_prompt = ""
    context = ""

    # 🔥 RAG
    if vector_db:
        result = search_docs(vector_db, prompt, k=5)

        print("🔍 RETRIEVED DOCS:", result)

        context = ""
        sources = []

        for item in result:
            context += item["content"] + "\n\n"
            sources.append(f"Page {item['page']}")

        context = context[:3000]

        print("📚 CONTEXT LENGTH:", len(context))
        print("📚 SAMPLE:", context[:200])

    # 🔥 PROMPT
    if context.strip():
        full_prompt = f"""
You are an AI assistant.

- Answer ONLY using the context
- Be clear and concise

Context:
{context}

Question:
{prompt}

Answer:
"""
    else:
        full_prompt = "You are a helpful AI assistant.\n"
        for msg in chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            full_prompt += f"{role}: {msg['content']}\n"

    print("🧠 PROMPT READY")

    # 🚀 STREAMING
    def generate():
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": "llama3:8b",
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "num_predict": 300,
                        "temperature": 0.4
                    }
                },
                stream=True,
                timeout=120
            )

            if response.status_code != 200:
                yield "⚠️ Model error"
                return

            ai_response = ""

            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        token = chunk.get("response", "")
                        ai_response += token
                        yield token
                    except:
                        continue

            chat_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            # 🔥 ADD SOURCES HERE
            unique_sources = list(set(sources))

            sources_text = "\n\nSources:\n"
            for s in unique_sources:
                sources_text += f"• {s}\n"

            yield sources_text

            end_time = time.time()
            print(f"⏱️ Response Time: {end_time - start_time:.2f}s")

        except Exception as e:
            print("ERROR:", str(e))
            yield "⚠️ Error: Unable to connect to AI model."

    return StreamingResponse(generate(), media_type="text/plain")

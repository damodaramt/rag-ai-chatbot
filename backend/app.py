from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psycopg2
import requests
import os
import shutil
import time

from rag_engine import create_vector_db, search_docs

app = FastAPI()

# ---------------- CORS ---------------- #
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DB CONNECTION ---------------- #
def get_conn():
    while True:
        try:
            conn = psycopg2.connect(
                dbname="rag_db",
                user="rag_user",
                password="rag123",
                host=os.getenv("DB_HOST", "postgres"),
                port="5432"
            )
            return conn
        except Exception as e:
            print("❌ DB not ready, retrying...", e)
            time.sleep(2)

# ---------------- OLLAMA ---------------- #
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://host.docker.internal:11434/api/generate"
)

# ---------------- PATHS ---------------- #
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- REQUEST MODEL ---------------- #
class ChatRequest(BaseModel):
    prompt: str
    user_id: int


# =========================================================
# CHAT MANAGEMENT
# =========================================================

@app.post("/api/new_chat")
def new_chat():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO conversations (title) VALUES (%s) RETURNING id",
        ("New Chat",)
    )
    chat_id = cur.fetchone()[0]
    conn.commit()

    cur.close()
    conn.close()

    return {"id": chat_id}


@app.get("/api/conversations")
def get_conversations():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id, title FROM conversations ORDER BY id DESC")
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [{"id": r[0], "title": r[1]} for r in rows]


@app.get("/api/messages/{chat_id}")
def get_messages(chat_id: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "SELECT role, content FROM messages WHERE conversation_id=%s ORDER BY id ASC",
        (chat_id,)
    )
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return [{"role": r[0], "content": r[1]} for r in rows]


# =========================================================
# PDF UPLOAD
# =========================================================

@app.post("/api/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    try:
        path = os.path.join(UPLOAD_DIR, file.filename)

        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        create_vector_db([path])

        return {"message": "PDF processed successfully"}

    except Exception as e:
        print("❌ PDF upload error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})


# =========================================================
# CHAT
# =========================================================

@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        conn = get_conn()
        cur = conn.cursor()

        prompt = req.prompt
        chat_id = req.user_id

        # ---------------- RAG ---------------- #
        docs = search_docs(prompt)
        context = "\n".join([d["text"] for d in docs]) if docs else ""
        sources = list(set([d["source"] for d in docs])) if docs else []

        full_prompt = f"""
Answer based on context.

Context:
{context}

Question:
{prompt}
"""

        # ---------------- SAVE USER ---------------- #
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (chat_id, "user", prompt)
        )
        conn.commit()

        # ---------------- CALL OLLAMA ---------------- #
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "llama3:latest",
                "prompt": full_prompt,
                "stream": False
            },
            timeout=300  # 🔥 important (model heavy)
        )

        data = response.json()

        # 🔥 SAFETY CHECK
        if "response" not in data:
            raise Exception(f"Ollama error: {data}")

        answer = data["response"]

        final_answer = answer + "\n\nSources:\n" + "\n".join(sources)

        # ---------------- SAVE BOT ---------------- #
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (chat_id, "assistant", final_answer)
        )
        conn.commit()

        cur.close()
        conn.close()

        return {"response": final_answer}

    except Exception as e:
        print("❌ CHAT ERROR:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()


class DocumentRequest(BaseModel):
    user_name: str
    content: str


client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://127.0.0.1:11434",
)

ollama_client = ollama.Client(host="http://127.0.0.1:11434")

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)


@app.get("/")
def home():
    return {"message": "Server is running, use /docs to test the API"}


@app.post("/documents")
def add_document(doc: DocumentRequest):
    chunks = [c.strip() for c in doc.content.split("\n\n") if c.strip()]
    ids = [f"{doc.user_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"user_name": doc.user_name} for _ in chunks]

    # delete the old chunks of this user first
    collection.delete(where={"user_name": doc.user_name})

    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return {"user": doc.user_name, "chunks_added": len(chunks)}


@app.get("/ask")
def ask(question: str, user: str | None = None):
    where = {"user_name": user} if user else None

    results = collection.query(
        query_texts=[question],
        n_results=2,
        where=where,
    )

    context = "\n\n".join(results["documents"][0])

    augmented_prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.
Answer in the same language as the question.

Context:
{context}

Question: {question}"""

    response = ollama_client.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
    }
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

with open("profile.txt","r") as f:
    text=f.read()

chunks=[chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
print(f"loaded {len(chunks)} chunks from profile.txt")



client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    url="http://localhost:11434",
    model_name="nomic-embed-text",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)

collection.add(
    ids=[f"chunk{i}" for i in range(len(chunks))],
    documents=chunks,
    metadatas=[{"source":"profile","chunk_index":i } for i in range(len(chunks))],
)

print(f"added {len(chunks)} chunks to the 'personal_profile' collection.")

print("knowledge base built successfully")
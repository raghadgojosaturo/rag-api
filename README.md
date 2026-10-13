# Personal AI Directory (RAG API)

A small API that answers questions about people, using only the info you give it. Everything runs on your own machine. No cloud, no API keys.

I built this while learning how RAG works. I wanted something I could actually run locally instead of paying for an API, and I wanted it to handle more than one person without mixing their details together.

## What it does

- You send a profile or resume for a person (`POST /documents`).
- The text gets split into chunks, embedded, and stored in ChromaDB. Every chunk is tagged with that person's `user_name`.
- You ask a question (`GET /ask`). The API finds the most relevant chunks, hands them to a local LLM, and returns the answer plus the chunks it used, so you can check where the answer came from.
- If you pass `user`, the search is limited to that person's chunks only.

## Stack

- FastAPI
- ChromaDB (local vector database)
- Ollama, running `qwen2.5:3b` for answers and `nomic-embed-text` for embeddings

## Setup

You need Python 3.10 or newer and [Ollama](https://ollama.com) installed.

1. Get the code:

```
git clone https://github.com/YOUR_USERNAME/rag-api.git
cd rag-api
```

2. Download the two models (this can take a while):

```
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

3. Create a virtual environment and install the dependencies:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On Mac/Linux, activate with `source venv/bin/activate` instead.

4. Start the server:

```
uvicorn main:app --reload
```

## Using it

Open `http://127.0.0.1:8000/docs` in your browser. You can test everything from there.

**Add a profile** with `POST /documents`:

```json
{
  "user_name": "Jordan",
  "content": "Jordan is a data analyst who loves SQL.\n\nJordan's email is jordan@example.com"
}
```

Two things that trip people up:

- The `content` has to stay on one line. Use `\n\n` to separate paragraphs, not the Enter key. Each paragraph becomes one chunk.
- If a paragraph contains a double quote, write it as `\"`.

**Ask a question** with `GET /ask`:

```
http://127.0.0.1:8000/ask?question=what is my name&user=Jordan
```

Leave out `user` and it searches everyone's chunks.

## Good to know

- `user` names are case-sensitive. `Jordan` and `jordan` are treated as two different people.
- Posting a profile again with the same `user_name` replaces the old one.
- Small models make mistakes. If the answers feel weak, change the model name in `main.py` to something bigger.
- It works best when you ask in English. Other languages work, but the small models get confused more often.
- Your data lives in a local `chroma_db/` folder. It is not part of this repo.
- There's no authentication. Don't expose this to the internet without adding some.


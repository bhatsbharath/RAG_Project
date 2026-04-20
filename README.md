# RAG Service

A Retrieval-Augmented Generation (RAG) application that lets you ask questions about your own documents. It combines semantic search (FAISS + sentence-transformers) with a local LLM to produce answers grounded in your data.

The stack runs entirely inside Docker — clone the repo, build the image, and the app is ready with the included sample document.

---

## Project Structure

```
RAG_Project/
├── main.py              # FastAPI backend (port 8000)
├── ui.py                # Streamlit frontend (port 8501)
├── start.sh             # Entrypoint: starts both services
├── Dockerfile
├── requirements.txt
├── app/
│   ├── rag_pipeline.py  # Document ingestion, chunking, FAISS index
│   ├── llm_engine.py    # HuggingFace text-generation wrapper
│   └── models.py        # Pydantic request/response schemas
├── data/                # Drop your documents here (.txt .md .pdf)
│   └── US Constitution.pdf   # Sample document (included)
└── models/              # Optional: pre-downloaded HuggingFace models
    └── README.md        # Instructions for downloading local models
```

---

## Model Choices & Reasoning

### Embedding Model — `all-MiniLM-L6-v2`
- **Size**: ~80 MB  
- **Why**: Fast, lightweight, and produces high-quality sentence embeddings for semantic search. Ideal for CPU environments. Automatically downloaded by `sentence-transformers` on first run.

### LLM — default `HuggingFaceH4/zephyr-7b-beta`
- **Size**: ~14 GB  
- **Why**: Fine-tuned on instruction-following and chat tasks; produces coherent, grounded answers. Best run on a GPU.

### LLM — lightweight alternative `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- **Size**: ~2 GB  
- **Why**: Runs comfortably on CPU with ~4 GB RAM. Noticeably less capable than Zephyr but good for development and low-resource machines.

**Auto-selection logic**: On startup the app scans the `models/` directory. If any model directory is present there, it is used automatically. Otherwise it downloads the default Zephyr model from HuggingFace.

| Model | Params | RAM (CPU) | Quality | Recommended for |
|-------|--------|-----------|---------|-----------------|
| `TinyLlama-1.1B-Chat-v1.0` | 1.1 B | ~4 GB | Good | CPU / dev / Docker |
| `zephyr-7b-beta` | 7 B | ~16 GB | Excellent | GPU |
| `Mistral-7B-Instruct-v0.2` | 7 B | ~16 GB | Excellent | GPU |

---

## Quick Start — Docker (recommended)

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed and running
- ~4 GB free RAM (for TinyLlama) or ~16 GB (for Zephyr)

### 1. Clone the repository

```bash
git clone <repo-url>
cd RAG_Project
```

### 2. (Optional) Pre-download a lightweight model

This avoids a large download at container startup. Skip this step to use the default Zephyr model (downloaded automatically, requires ~16 GB RAM).

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                  local_dir='models/TinyLlama-1.1B-Chat-v1.0')
"
```

### 3. Build the Docker image

```bash
docker build -t rag-app .
```

This copies everything — the `data/` folder (with the US Constitution sample), the `models/` folder, and all source code — into the image. No extra setup steps required after cloning.

### 4. Run the container

```bash
docker run -p 8000:8000 rag-app
```

Wait for the startup logs to show:

```
RAG Service is ready!
```

---

## Using the App

### Streamlit GUI (recommended)

Open **http://localhost:8000** in your browser.

1. Type your question in the text area (e.g. *"What are the first 10 amendments?"*)
2. Use the **Top sources** slider in the sidebar to control how many document chunks are retrieved
3. Click **Search** — the answer and the source passages appear below

The sidebar **API URL** field defaults to `http://localhost:8080` (the internal FastAPI port) and should not need changing when running via Docker.

### REST API

The FastAPI backend runs internally on port 8080 inside the container and is accessible directly when running locally:

```bash
# Ask a question
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the First Amendment say?", "top_k": 3}'
```

Interactive Swagger docs (local dev only): **http://localhost:8080/docs**

---

## Adding Your Own Documents

1. Copy `.txt`, `.md`, or `.pdf` files into the `data/` folder
2. Rebuild the Docker image so the new files are included:

```bash
docker build -t rag-app .
docker run -p 8000:8000 -p 8501:8501 rag-app
```

Documents are automatically loaded, chunked, and indexed at container startup.

---

## Using a Local Model

Download a model into `models/` (see [models/README.md](models/README.md)), then rebuild:

```bash
docker build -t rag-app .
docker run -p 8000:8000 rag-app
```

The app picks up the first directory found inside `models/` automatically.

---

## Local Development Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt

# Run the API
python main.py

# In a second terminal, run the UI
streamlit run ui.py
```

API at `http://localhost:8080` · UI at `http://localhost:8000`

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Readiness check |
| `POST` | `/query` | Submit a question |
| `GET` | `/docs` | Swagger UI |

### Query request

```json
{
  "question": "What is the Second Amendment?",
  "top_k": 3
}
```

### Query response

```json
{
  "answer": "The Second Amendment states...",
  "sources": [
    {"text": "...relevant passage...", "distance": 0.12}
  ],
  "model": {
    "model_name": "models/TinyLlama-1.1B-Chat-v1.0",
    "device": "cpu",
    "family": "TinyLlama"
  },
  "status": "success"
}
```


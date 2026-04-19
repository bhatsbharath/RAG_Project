# RAG Service

Retrieval Augmented Generation (RAG) service using semantic search + LLM.

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

API at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

## Usage

Add documents to `data/` folder (.txt, .pdf, or .md files).

Query example:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "top_k": 3}'
```

## Streamlit UI (Optional)

```bash
streamlit run ui.py
```

## Docker

```bash
docker build -t rag-app .
docker run -p 8000:8000 rag-app
```

## Components

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (in-memory)
- **LLM**: Zephyr 7B (HuggingFace)
- **API**: FastAPI

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info |
| GET | `/health` | Service status |
| POST | `/query` | Submit question |
| GET | `/docs` | Swagger UI |

## Query Response Format

```json
{
  "answer": "...",
  "sources": [
    {"text": "...", "distance": 0.123}
  ],
  "model": {
    "model_name": "HuggingFaceH4/zephyr-7b-beta",
    "device": "cpu",
    "family": "Zephyr"
  },
  "status": "success"
}
```

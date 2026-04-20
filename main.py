"""
RAG API - main FastAPI server

Service Overview:
1. Load embedding model (sentence-transformers)
2. Ingest documents from data/ directory
3. Create document index (chunks + embeddings)
4. Load LLM for answer generation
5. Serve /query endpoint for RAG conversations
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.rag_pipeline import RAGPipeline
from app.llm_engine import LLMEngine
from app.models import QueryRequest, QueryResponse, HealthResponse

# Will be initialized on startup
rag_pipeline = None
llm_engine = None


def startup_event():
    """Service initialization on startup
    
    Steps:
    1. Initialize RAG pipeline with embedding model
    2. Load and ingest documents from data/
    3. Initialize LLM for answer generation
    """
    global rag_pipeline, llm_engine
    
    print("\n" + "=" * 60)
    print("Initializing RAG Service")
    print("=" * 60)

    # Step 1: Initialize embedding pipeline
    print("\n[Step 1] Loading embedding model...")
    rag_pipeline = RAGPipeline(embedding_model="all-MiniLM-L6-v2")

    # Step 2: Load documents
    print("[Step 2] Loading documents...")
    data_dir = "data"
    if os.path.exists(data_dir):
        docs = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith((".txt", ".md", ".pdf"))
        ]
        if docs:
            print(f"         Found {len(docs)} document(s)")
            rag_pipeline.ingest_documents(docs)
        else:
            print("         No documents found")
    else:
        print(f"         Data directory not found")

    # Step 3: Initialize LLM
    print("[Step 3] Loading language model...")
    models_dir = "models"
    local_models = []
    if os.path.exists(models_dir):
        local_models = [
            os.path.join(models_dir, d)
            for d in sorted(os.listdir(models_dir))
            if os.path.isdir(os.path.join(models_dir, d))
        ]

    if local_models:
        model_path = local_models[0]
        print(f"         Using local model: {model_path}")
    else:
        model_path = "HuggingFaceH4/zephyr-7b-beta"
        print(f"         No local model found in {models_dir}/, downloading: {model_path}")

    llm_engine = LLMEngine(model_name=model_path)

    print("\n" + "=" * 60)
    print("RAG Service is ready!")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 50 + "\n")


# Create the FastAPI app
app = FastAPI(
    title="RAG Service",
    description="Retrieval Augmented Generation API",
    version="1.0",
)

# Add CORS so UI can call from different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Run startup when app starts
@app.on_event("startup")
async def on_startup():
    startup_event()


@app.get("/")
async def root():
    """Say hello"""
    return {
        "service": "RAG Service",
        "version": "1.0",
        "endpoints": {
            "health": "GET /health",
            "query": "POST /query",
            "docs": "/docs",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Check if service is ready"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    stats = rag_pipeline.get_index_stats()
    return HealthResponse(
        status="ready",
        documents_indexed=stats["total_chunks"],
        embedding_model=stats["embedding_model"],
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process question using RAG (Retrieval Augmented Generation)
    
    RAG Algorithm:
    1. RETRIEVE: Find top-k relevant document chunks using semantic similarity
    2. BUILD CONTEXT: Concatenate retrieved chunks as context
    3. PROMPT: Create prompt = context + question
    4. GENERATE: Use LLM to generate answer grounded in context
    5. RETURN: Answer + sources + model metadata
    """
    if rag_pipeline is None or llm_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    try:
        print(f"\n>>> Query: {question}")
        
        # Step 1: RETRIEVE - Get relevant chunks
        print("   [1] Retrieving relevant chunks...")
        chunks = rag_pipeline.retrieve(question, k=request.top_k)

        if not chunks:
            answer = "I couldn't find relevant information in the documents."
            sources = []
        else:
            # Step 2: BUILD CONTEXT - Assemble chunks
            context_text = "\n\n".join(
                [f"[Source {i+1}]\n{c['chunk']}" for i, c in enumerate(chunks)]
            )

            # Step 3: PROMPT - Create LLM prompt
            prompt = f"""Answer this question based on the provided context.
Only use information from the context. If you cannot answer, say so.

Context:
{context_text}

Question: {question}

Answer:"""

            # Step 4: GENERATE - Get LLM response
            print("   [2] Generating answer with LLM...")
            answer = llm_engine.generate(prompt, max_length=256, temperature=0.7)

            # Step 5: RETURN - Format response
            sources = [
                {
                    "text": c["chunk"],
                    "distance": c["distance"],
                }
                for c in chunks
            ]

        model_info = llm_engine.get_model_info()

        return QueryResponse(
            answer=answer,
            sources=sources,
            model=model_info,
            status="success",
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

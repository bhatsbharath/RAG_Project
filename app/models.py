"""
Request and Response models for API
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    model: Dict[str, str]
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    embedding_model: str

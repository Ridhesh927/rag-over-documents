from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    page_number: int
    text: str
    relevance_score: Optional[float] = None

class Citation(BaseModel):
    doc_name: str
    page_number: int
    chunk_id: str
    snippet: str
    relevance_score: float

class IngestResponse(BaseModel):
    success: bool
    doc_id: str
    filename: str
    file_type: str
    total_pages: int
    chunks_created: int
    processing_time_seconds: float

class QueryRequest(BaseModel):
    question: str = Field(description="The user question to ask about the ingested documents")
    top_k: Optional[int] = Field(default=4, ge=1, le=20, description="Number of top passages to retrieve")
    similarity_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    doc_ids: Optional[List[str]] = Field(default=None, description="Optional filter to restrict search to specific documents")

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    retrieved_chunks_count: int
    model_used: str
    latency_seconds: float

class DocumentMeta(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    total_pages: int
    chunk_count: int
    ingested_at: str

class RAGHealthResponse(BaseModel):
    status: str
    version: str
    vector_store_type: str
    total_documents: int
    total_chunks: int
    embedding_model: str
    llm_configured: bool

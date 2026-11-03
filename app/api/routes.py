from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import List

from app.config import settings
from app.schemas.rag_schemas import (
    IngestResponse,
    QueryRequest,
    QueryResponse,
    DocumentMeta,
    RAGHealthResponse,
)
from app.services.rag_engine import rag_engine
from app.services.vector_store import vector_store

router = APIRouter(prefix="/api/v1/rag")

@router.get("/health", response_model=RAGHealthResponse)
async def health_check():
    stats = vector_store.get_stats()
    return RAGHealthResponse(
        status="healthy",
        version=settings.VERSION,
        vector_store_type=stats["type"],
        total_documents=stats["total_documents"],
        total_chunks=stats["total_chunks"],
        embedding_model=settings.EMBEDDING_MODEL,
        llm_configured=bool(settings.GEMINI_API_KEY),
    )

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_document"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}"
        )

    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.MAX_FILE_SIZE_MB}MB"
        )

    try:
        response = rag_engine.ingest_document(filename, contents)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse)
async def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question string cannot be empty."
        )

    try:
        response = rag_engine.query(
            question=req.question,
            top_k=req.top_k or settings.DEFAULT_TOP_K,
            similarity_threshold=req.similarity_threshold or 0.0,
            doc_ids=req.doc_ids
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query execution failed: {str(e)}"
        )

@router.get("/documents", response_model=List[DocumentMeta])
async def list_documents():
    return vector_store.list_documents()

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    success = vector_store.delete_document(doc_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{doc_id}' not found."
        )
    return {"success": True, "message": f"Document '{doc_id}' and all associated chunks deleted."}

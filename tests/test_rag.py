import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE_DOC_TEXT = (
    "Artificial Intelligence and Document Processing.\n"
    "Document intelligence enables automated parsing of semi-structured invoices, receipts, and forms. "
    "By combining optical character recognition (OCR) with large language models, systems achieve high accuracy. "
    "LangChain and pgvector allow efficient vector retrieval with exact page-level source citations. "
    "Author Ridhesh Anil Mahajan created this pipeline to accelerate enterprise document workflows."
)

def test_rag_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["author"] == "Ridhesh Anil Mahajan"
    assert "version" in data

def test_rag_health():
    response = client.get("/api/v1/rag/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store_type" in data
    assert "total_documents" in data

def test_rag_ingest_and_query_flow():
    # 1. Ingest document
    files = {"file": ("ai_guide.txt", SAMPLE_DOC_TEXT.encode("utf-8"), "text/plain")}
    ingest_res = client.post("/api/v1/rag/ingest", files=files)
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["success"] is True
    assert ingest_data["chunks_created"] >= 1
    doc_id = ingest_data["doc_id"]

    # 2. List documents
    list_res = client.get("/api/v1/rag/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert any(d["doc_id"] == doc_id for d in docs)

    # 3. Query document
    query_payload = {
        "question": "Who created this pipeline?",
        "top_k": 2,
        "similarity_threshold": 0.0
    }
    query_res = client.post("/api/v1/rag/query", json=query_payload)
    assert query_res.status_code == 200
    query_data = query_res.json()
    assert query_data["question"] == "Who created this pipeline?"
    assert len(query_data["citations"]) > 0
    assert query_data["citations"][0]["doc_name"] == "ai_guide.txt"
    assert query_data["retrieved_chunks_count"] > 0
    assert len(query_data["answer"]) > 0

    # 4. Delete document
    del_res = client.delete(f"/api/v1/rag/documents/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 5. Verify deletion
    list_res_after = client.get("/api/v1/rag/documents")
    assert not any(d["doc_id"] == doc_id for d in list_res_after.json())

def test_rag_empty_query():
    res = client.post("/api/v1/rag/query", json={"question": "   "})
    assert res.status_code == 400
    assert "Question string cannot be empty" in res.json()["detail"]

def test_rag_delete_nonexistent():
    res = client.delete("/api/v1/rag/documents/doc_nonexistent_999")
    assert res.status_code == 404

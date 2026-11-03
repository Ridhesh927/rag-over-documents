import datetime
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from app.schemas.rag_schemas import DocumentChunk, DocumentMeta
from app.config import settings

logger = logging.getLogger(__name__)

class InMemoryVectorStore:
    def __init__(self):
        # In-memory index
        self.chunks: Dict[str, Dict[str, Any]] = {}  # chunk_id -> metadata & text
        self.embeddings: Dict[str, np.ndarray] = {}  # chunk_id -> 1D numpy vector
        self.documents: Dict[str, Dict[str, Any]] = {}  # doc_id -> metadata

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], doc_meta: Dict[str, Any]):
        doc_id = doc_meta["doc_id"]
        self.documents[doc_id] = {
            **doc_meta,
            "chunk_count": len(chunks),
            "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = chunk["chunk_id"]
            vec = np.array(emb, dtype=np.float32)
            # normalize if not already normalized
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.embeddings[chunk_id] = vec
            self.chunks[chunk_id] = chunk

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        threshold: float = 0.0,
        doc_ids: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        if not self.embeddings:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = []
        for chunk_id, emb in self.embeddings.items():
            chunk = self.chunks[chunk_id]
            if doc_ids and chunk["doc_id"] not in doc_ids:
                continue

            # Cosine similarity between two unit vectors is dot product
            sim = float(np.dot(q_vec, emb))
            if sim >= threshold:
                scores.append((sim, chunk))

        # Sort descending by similarity score
        scores.sort(key=lambda x: x[0], reverse=True)
        top_matches = scores[:top_k]

        return [
            DocumentChunk(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                doc_name=chunk["doc_name"],
                page_number=chunk["page_number"],
                text=chunk["text"],
                relevance_score=round(score, 4)
            )
            for score, chunk in top_matches
        ]

    def list_documents(self) -> List[DocumentMeta]:
        return [
            DocumentMeta(
                doc_id=meta["doc_id"],
                filename=meta.get("filename", "unknown"),
                file_type=meta.get("file_type", "pdf"),
                total_pages=meta.get("total_pages", 1),
                chunk_count=meta.get("chunk_count", 0),
                ingested_at=meta.get("ingested_at", "")
            )
            for meta in self.documents.values()
        ]

    def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self.documents:
            return False

        del self.documents[doc_id]
        chunks_to_remove = [c_id for c_id, chunk in self.chunks.items() if chunk["doc_id"] == doc_id]
        for c_id in chunks_to_remove:
            self.chunks.pop(c_id, None)
            self.embeddings.pop(c_id, None)
        return True

    def get_stats(self) -> dict:
        return {
            "type": "in-memory-cosine" if not settings.DATABASE_URL else "pgvector-hybrid",
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
        }

vector_store = InMemoryVectorStore()

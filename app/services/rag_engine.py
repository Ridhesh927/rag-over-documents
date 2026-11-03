import io
import time
import uuid
import logging
from typing import List, Dict, Any, Optional

import pypdf
from app.config import settings
from app.schemas.rag_schemas import IngestResponse, QueryResponse, Citation
from app.services.chunker import RecursiveChunker
from app.services.embeddings import embedding_engine
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        self.chunker = RecursiveChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.llm_model = settings.GEMINI_MODEL
        self._gemini_client = None

        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self._gemini_client = genai.GenerativeModel(self.llm_model)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini model for RAG: {e}")

    def extract_pages(self, filename: str, file_bytes: bytes) -> List[Dict[str, Any]]:
        ext = filename.lower().split(".")[-1] if "." in filename else ""
        pages = []

        if ext == "pdf":
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    for i, page in enumerate(pdf.pages):
                        txt = page.extract_text() or ""
                        if txt.strip():
                            pages.append({"page_number": i + 1, "text": txt})
            except Exception:
                try:
                    reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                    for i, page in enumerate(reader.pages):
                        txt = page.extract_text() or ""
                        if txt.strip():
                            pages.append({"page_number": i + 1, "text": txt})
                except Exception:
                    # Fallback for plain text or mock
                    txt = file_bytes.decode("utf-8", errors="ignore")
                    pages.append({"page_number": 1, "text": txt})
        else:
            # Text / Markdown / CSV
            try:
                txt = file_bytes.decode("utf-8", errors="ignore")
                pages.append({"page_number": 1, "text": txt})
            except Exception:
                pages.append({"page_number": 1, "text": str(file_bytes)})

        return pages if pages else [{"page_number": 1, "text": "Empty document"}]

    def ingest_document(self, filename: str, file_bytes: bytes) -> IngestResponse:
        start_time = time.time()
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        ext = filename.lower().split(".")[-1] if "." in filename else "bin"

        # 1. Extract pages
        pages = self.extract_pages(filename, file_bytes)

        # 2. Chunk text
        chunks = self.chunker.chunk_document_pages(
            doc_id=doc_id,
            doc_name=filename,
            pages_text=pages
        )

        if not chunks:
            # Ensure at least 1 chunk
            chunks = [{
                "chunk_id": f"{doc_id}_p1_c0",
                "doc_id": doc_id,
                "doc_name": filename,
                "page_number": 1,
                "text": "Empty document contents"
            }]

        # 3. Compute embeddings
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embedding_engine.embed_batch(chunk_texts)

        # 4. Store in vector database
        doc_meta = {
            "doc_id": doc_id,
            "filename": filename,
            "file_type": ext,
            "total_pages": len(pages),
        }
        vector_store.add_chunks(chunks, embeddings, doc_meta)

        elapsed = round(time.time() - start_time, 3)
        return IngestResponse(
            success=True,
            doc_id=doc_id,
            filename=filename,
            file_type=ext,
            total_pages=len(pages),
            chunks_created=len(chunks),
            processing_time_seconds=elapsed,
        )

    def query(
        self,
        question: str,
        top_k: int = 4,
        similarity_threshold: float = 0.0,
        doc_ids: Optional[List[str]] = None
    ) -> QueryResponse:
        start_time = time.time()

        # 1. Embed query
        q_vector = embedding_engine.embed_text(question)

        # 2. Retrieve top-k chunks
        matched_chunks = vector_store.similarity_search(
            query_vector=q_vector,
            top_k=top_k,
            threshold=similarity_threshold,
            doc_ids=doc_ids
        )

        # 3. Formulate Citations
        citations = [
            Citation(
                doc_name=chunk.doc_name,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                snippet=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
                relevance_score=chunk.relevance_score or 0.0
            )
            for chunk in matched_chunks
        ]

        # 4. Synthesize Answer
        if not matched_chunks:
            return QueryResponse(
                question=question,
                answer="No relevant information found in the indexed documents for this query.",
                citations=[],
                retrieved_chunks_count=0,
                model_used="none",
                latency_seconds=round(time.time() - start_time, 3),
            )

        context_blocks = "\n\n".join([
            f"[Passage {i+1} | Source: {c.doc_name}, Page: {c.page_number}]\n{c.text}"
            for i, c in enumerate(matched_chunks)
        ])

        system_prompt = (
            "You are an expert Document Intelligence Assistant. Answer the user's question accurately and concisely, "
            "using ONLY the provided reference passages. Cite sources for every claim using the format [DocumentName, Page X]. "
            "If the answer cannot be determined from the passages, state clearly that the documents do not contain the answer."
        )

        user_content = f"Reference Passages:\n{context_blocks}\n\nQuestion: {question}"

        if self._gemini_client:
            try:
                response = self._gemini_client.generate_content(f"{system_prompt}\n\n{user_content}")
                answer_text = response.text.strip()
                model_used = self.llm_model
            except Exception as e:
                logger.warning(f"LLM synthesis call failed: {e}. Using extractive summary fallback.")
                answer_text = self._synthesize_extractive_fallback(matched_chunks, question)
                model_used = "extractive-fallback"
        else:
            answer_text = self._synthesize_extractive_fallback(matched_chunks, question)
            model_used = "extractive-fallback"

        elapsed = round(time.time() - start_time, 3)

        return QueryResponse(
            question=question,
            answer=answer_text,
            citations=citations,
            retrieved_chunks_count=len(matched_chunks),
            model_used=model_used,
            latency_seconds=elapsed,
        )

    def _synthesize_extractive_fallback(self, chunks: List[Any], question: str) -> str:
        """
        Extractive synthesis for offline / test environments.
        """
        top_chunk = chunks[0]
        return (
            f"Based on [{top_chunk.doc_name}, Page {top_chunk.page_number}]: "
            f"\"{top_chunk.text.strip()}\""
        )

rag_engine = RAGEngine()

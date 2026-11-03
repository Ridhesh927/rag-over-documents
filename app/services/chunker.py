import uuid
from typing import List, Dict, Any

class RecursiveChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    def split_text(self, text: str) -> List[str]:
        """
        Recursively splits text into chunks of maximum size with overlap.
        """
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            
            # If not at the end of the text, try to find a natural break separator
            if end < text_len:
                split_at = -1
                for sep in self.separators:
                    pos = text.rfind(sep, start, end)
                    if pos != -1 and pos > start + (self.chunk_size // 3):
                        split_at = pos + len(sep)
                        break
                if split_at != -1:
                    end = split_at

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move forward taking overlap into account
            if end >= text_len:
                break
            start = max(start + 1, end - self.chunk_overlap)

        return chunks

    def chunk_document_pages(
        self, doc_id: str, doc_name: str, pages_text: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        pages_text: list of {"page_number": int, "text": str}
        Returns list of structured chunk dicts.
        """
        chunks = []
        for page_info in pages_text:
            page_num = page_info.get("page_number", 1)
            raw_text = page_info.get("text", "")
            if not raw_text.strip():
                continue

            page_chunks = self.split_text(raw_text)
            for idx, chunk_text in enumerate(page_chunks):
                chunk_id = f"{doc_id}_p{page_num}_c{idx}_{uuid.uuid4().hex[:6]}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_number": page_num,
                    "text": chunk_text
                })
        return chunks

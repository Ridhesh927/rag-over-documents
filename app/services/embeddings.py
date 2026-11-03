import math
import logging
from typing import List
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.EMBEDDING_MODEL
        self._genai_configured = False
        self.embedding_dim = 768

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai_configured = True
            except Exception as e:
                logger.warning(f"Could not configure Gemini embeddings: {e}")

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self._genai_configured:
            try:
                import google.generativeai as genai
                embeddings = []
                for t in texts:
                    res = genai.embed_content(
                        model=self.model_name,
                        content=t,
                        task_type="retrieval_document"
                    )
                    embeddings.append(res['embedding'])
                return embeddings
            except Exception as e:
                logger.warning(f"Gemini API embed call failed: {e}. Falling back to local vector generation.")

        # Local deterministic normalized embedding generator (for offline / test usage)
        return [self._local_fallback_embed(t) for t in texts]

    def _local_fallback_embed(self, text: str) -> List[float]:
        """
        Generates a 768-dimensional normalized deterministic vector from text tokens.
        """
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            return vec.tolist()

        for idx, word in enumerate(words):
            h = hash(word)
            pos = abs(h) % self.embedding_dim
            weight = 1.0 / (1.0 + math.log(1.0 + idx))
            vec[pos] += weight

            # Bigram hashing for local semantic capture
            if idx > 0:
                bi_h = hash(f"{words[idx-1]}_{word}")
                bi_pos = abs(bi_h) % self.embedding_dim
                vec[bi_pos] += weight * 1.5

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

embedding_engine = EmbeddingEngine()

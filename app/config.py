import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional, Union

class RAGSettings(BaseSettings):
    PROJECT_NAME: str = "RAG Over Documents Microservice"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8001
    
    # LLM & Embedding Settings
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-1.5-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    
    # Optional Postgres pgvector
    DATABASE_URL: Optional[str] = None

    # Ingestion & Retrieval configs
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    DEFAULT_TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.2
    
    # File limits
    MAX_FILE_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: list[str] = [".pdf", ".txt", ".md", ".csv"]

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, v: Union[str, bool, int]) -> bool:
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on", "dev", "development")
        return bool(v)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = RAGSettings()

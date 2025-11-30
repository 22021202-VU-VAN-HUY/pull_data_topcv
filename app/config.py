# app/config.py

import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


class Settings:
    # db
    POSTGRES_USER = os.getenv("POSTGRES_USER", "topcv_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "topcv_password")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "topcv_db")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5440"))

    # crawl - app/topcv
    TOPCV_SITEMAP_ROOT: str = os.getenv(
        "TOPCV_SITEMAP_ROOT",
        "https://www.topcv.vn/sitemap.xml",
    )
    SITEMAP_MAX_JOBS: int = int(os.getenv("SITEMAP_MAX_JOBS", "2000"))
    JOB_MAX_RETRY: int = int(os.getenv("JOB_MAX_RETRY", "3"))
    CRAWL_SLEEP_SECONDS: float = float(os.getenv("CRAWL_SLEEP_SECONDS", "5.0"))

    # RAG - app/api/rag
    RAG_EMBEDDING_MODEL_NAME: str = os.getenv(
        "RAG_EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    RAG_EMBEDDING_BATCH_SIZE: int = int(os.getenv("RAG_EMBEDDING_BATCH_SIZE", "64"))
    RAG_CHUNK_MAX_CHARS: int = int(os.getenv("RAG_CHUNK_MAX_CHARS", "800"))
    RAG_DEFAULT_TOP_K: int = int(os.getenv("RAG_DEFAULT_TOP_K", "8"))
    RAG_MAX_CONTEXT_DOCS: int = int(os.getenv("RAG_MAX_CONTEXT_DOCS", "20"))
    RAG_MAX_HISTORY_TURNS: int = int(os.getenv("RAG_MAX_HISTORY_TURNS", "10"))

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.15"))
    GEMINI_MAX_OUTPUT_TOKENS: int = int(
        os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "2048")
    )

settings = Settings()

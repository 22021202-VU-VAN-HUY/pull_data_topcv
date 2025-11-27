# app/api/rag/retriever.py

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from app.db import get_connection
from app.config import settings

logger = logging.getLogger(__name__)

# =========================
#  EMBEDDING MODEL
# =========================

_embedding_model: Optional[SentenceTransformer] = None


def get_query_embedding_model() -> SentenceTransformer:
    """
    Model dùng cho query (phải trùng với model lúc index).
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(
            "Loading query embedding model: %s",
            settings.RAG_EMBEDDING_MODEL_NAME,
        )
        model = SentenceTransformer(settings.RAG_EMBEDDING_MODEL_NAME)
        # hạn chế độ dài cho chắc
        model.max_seq_length = 512
        _embedding_model = model
    return _embedding_model


def embed_query(text: str) -> List[float]:
    model = get_query_embedding_model()
    vec = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
    return vec.tolist()


# =========================
#  RETRIEVE
# =========================

def retrieve_jobs(
    query: str,
    top_k: Optional[int] = None,
    only_active: bool = True,
) -> List[Dict[str, Any]]:
    """
    Truy vấn rag_job_documents theo embedding + ưu tiên job còn hạn.

    Trả về list doc:
    [
      {
        "doc_id": ...,
        "job_id": ...,
        "chunk_index": ...,
        "chunk_text": "...",   # lấy từ cột content
        "score": 0.87,
        "metadata": {...},     # snapshot full job
      },
      ...
    ]
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = top_k or settings.RAG_DEFAULT_TOP_K

    # 1. Tạo embedding cho câu hỏi
    query_vec = embed_query(query)

    # Bảng rag_job_documents hiện có:
    #   - id BIGSERIAL
    #   - job_id BIGINT
    #   - doc_type VARCHAR
    #   - section_type VARCHAR
    #   - chunk_index INT
    #   - content TEXT
    #   - metadata JSONB
    #   - embedding_vec vector(384)
    sql = """
        WITH q AS (
            SELECT %s::vector AS embedding_vec
        )
        SELECT
            d.id          AS doc_id,
            d.job_id      AS job_id,
            d.chunk_index AS chunk_index,
            d.content     AS content,
            d.metadata    AS metadata,
            1 - (d.embedding_vec <=> q.embedding_vec) AS score
        FROM rag_job_documents d, q
        WHERE
            (%s = FALSE
             OR (d.metadata->>'deadline') IS NULL
             OR (d.metadata->>'deadline')::timestamptz >= NOW())
        ORDER BY d.embedding_vec <-> q.embedding_vec
        LIMIT %s;
    """

    docs: List[Dict[str, Any]] = []

    with get_connection() as conn:
        # get_connection dùng RealDictCursor → row là dict
        with conn.cursor() as cur:
            cur.execute(sql, [query_vec, only_active, top_k])
            rows = cur.fetchall()

    for row in rows:
        # row: RealDictCursor → dict với key đúng theo alias trong SELECT
        #   row["doc_id"], row["job_id"], row["chunk_index"], row["content"], row["metadata"], row["score"]
        metadata_raw = row.get("metadata")

        if isinstance(metadata_raw, str):
            try:
                metadata = json.loads(metadata_raw)
            except Exception:
                metadata = {"raw": metadata_raw}
        else:
            metadata = metadata_raw or {}

        score_val = row.get("score")
        try:
            score_float = float(score_val) if score_val is not None else None
        except (TypeError, ValueError):
            # phòng hờ trường hợp score bị kiểu lạ
            logger.warning("Không convert được score='%s' sang float", score_val)
            score_float = None

        docs.append(
            {
                "doc_id": row.get("doc_id"),
                "job_id": row.get("job_id"),
                "chunk_index": row.get("chunk_index"),
                "chunk_text": row.get("content"),
                "metadata": metadata,
                "score": score_float,
            }
        )

    logger.info(
        "retrieve_jobs: query='%s', top_k=%s, only_active=%s, got %d docs",
        query,
        top_k,
        only_active,
        len(docs),
    )

    return docs

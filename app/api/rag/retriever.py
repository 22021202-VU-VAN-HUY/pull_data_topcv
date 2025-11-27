# app/api/rag/retriever.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from app.config import settings
from app.db import get_connection
from app.api.rag.embeddings import embed_texts

logger = logging.getLogger(__name__)


def _vector_to_literal(vec: List[float]) -> str:
    """
    Chuyển list float -> chuỗi literal Postgres vector: '[0.1,0.2,...]'.
    """
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def retrieve_relevant_chunks(
    query: str,
    current_job_id: Optional[int] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Lấy các chunk liên quan nhất đến câu hỏi.
    Ưu tiên:
      - job_id = current_job_id (nếu có)
      - job đang còn hạn (metadata->>'is_active' = true)
      - khoảng cách vector nhỏ (embedding_vec <-> query_vec)
    """
    query = (query or "").strip()
    if not query:
        return []

    k = top_k or settings.RAG_DEFAULT_TOP_K

    logger.info("RAG retriever: query='%s', current_job_id=%s, top_k=%s", query, current_job_id, k)

    q_vec = embed_texts([query])[0]
    q_vec_lit = _vector_to_literal(q_vec)

    with get_connection() as conn:
        with conn.cursor() as cur:
            base_sql = """
                SELECT
                    id,
                    job_id,
                    doc_type,
                    section_type,
                    chunk_index,
                    content,
                    metadata,
                    (embedding_vec <-> %(q_vec)s::vector) AS dist,
                    (metadata->>'is_active')::boolean AS is_active
                FROM rag_job_documents
            """

            if current_job_id is not None:
                sql = base_sql + """
                    ORDER BY
                        (CASE WHEN job_id = %(current_job_id)s THEN 0 ELSE 1 END),
                        (CASE WHEN (metadata->>'is_active')::boolean THEN 0 ELSE 1 END),
                        dist
                    LIMIT %(limit)s
                """
                params = {
                    "q_vec": q_vec_lit,
                    "current_job_id": current_job_id,
                    "limit": k,
                }
            else:
                sql = base_sql + """
                    ORDER BY
                        (CASE WHEN (metadata->>'is_active')::boolean THEN 0 ELSE 1 END),
                        dist
                    LIMIT %(limit)s
                """
                params = {
                    "q_vec": q_vec_lit,
                    "limit": k,
                }

            cur.execute(sql, params)
            rows = cur.fetchall() or []

    results: List[Dict[str, Any]] = []
    for r in rows:
        results.append(
            {
                "id": r["id"],
                "job_id": r["job_id"],
                "doc_type": r["doc_type"],
                "section_type": r["section_type"],
                "chunk_index": r["chunk_index"],
                "content": r["content"],
                "metadata": r["metadata"],
                "distance": float(r["dist"]),
                "is_active": bool(r["is_active"]),
            }
        )

    return results

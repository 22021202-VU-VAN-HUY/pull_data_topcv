# app/api/rag/retriever.py

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from datetime import datetime, timezone

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
#  FILTER HELPERS
# =========================


def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def _location_pass(
    meta: Dict[str, Any],
    filter_locations: List[str],
    chunk_text: str,
) -> bool:
    """
    Lọc theo địa điểm:
    - Nếu không có filter_locations -> luôn pass
    - Nếu có -> check trong meta["locations"] và text.
    """
    if not filter_locations:
        return True

    meta_locs: List[str] = meta.get("locations") or []
    meta_locs_norm = " ".join(_normalize_text(x) for x in meta_locs)
    text_norm = _normalize_text(chunk_text)

    for loc in filter_locations:
        loc_norm = _normalize_text(loc)
        if loc_norm and (loc_norm in meta_locs_norm or loc_norm in text_norm):
            return True
    return False


def _salary_pass(
    meta: Dict[str, Any],
    f_min: Optional[int],
    f_max: Optional[int],
) -> bool:
    """
    Lọc theo khoảng lương.
    - f_min, f_max: VND, có thể None.
    - meta["salary"] có thể có min/max, cũng là VND.
    """
    salary = meta.get("salary") or {}
    s_min = salary.get("min")
    s_max = salary.get("max")

    # nếu không có filter lương -> pass
    if f_min is None and f_max is None:
        return True

    # nếu job không có thông tin lương rõ ràng -> vẫn giữ (đừng loại quá tay)
    if s_min is None and s_max is None:
        return True

    # kiểm tra giao nhau của khoảng [s_min, s_max] và [f_min, f_max]
    low = s_min if s_min is not None else s_max
    high = s_max if s_max is not None else s_min

    if low is None and high is None:
        return True

    if f_min is not None and high is not None and high < f_min:
        return False
    if f_max is not None and low is not None and low > f_max:
        return False

    return True


def _skills_pass(
    meta: Dict[str, Any],
    filter_skills: List[str],
    chunk_text: str,
) -> bool:
    """
    Lọc theo kỹ năng: check trong mô tả / yêu cầu / chunk_text.
    """
    if not filter_skills:
        return True

    detail_sections = meta.get("detail_sections") or {}
    mo_ta = (detail_sections.get("mo_ta_cong_viec") or {}).get("text") or ""
    yeu_cau = (detail_sections.get("yeu_cau_ung_vien") or {}).get("text") or ""

    haystack = " ".join(
        [
            _normalize_text(mo_ta),
            _normalize_text(yeu_cau),
            _normalize_text(chunk_text),
        ]
    )

    for skill in filter_skills:
        s_norm = _normalize_text(skill)
        if s_norm and s_norm in haystack:
            return True
    return False


# =========================
#  RETRIEVE
# =========================


def retrieve_jobs(
    query: str,
    top_k: Optional[int] = None,
    only_active: bool = True,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Truy vấn rag_job_documents theo embedding + lọc hybrid (địa điểm, lương, kỹ năng).

    Trả về list doc dạng:
    {
      "doc_id": .,
      "job_id": .,
      "chunk_index": .,
      "chunk_text": ".",
      "metadata": {.},
      "score": 0.87,
    }
    """
    query = (query or "").strip()
    if not query:
        return []

    top_k = top_k or settings.RAG_DEFAULT_TOP_K
    filters = filters or {}
    f_locations: List[str] = filters.get("locations") or []
    f_min_salary: Optional[int] = filters.get("min_salary_vnd")
    f_max_salary: Optional[int] = filters.get("max_salary_vnd")
    f_skills: List[str] = filters.get("skills") or []

    # ------------ 1. embedding cho query ------------
    query_vec = embed_query(query)

    # Lấy pool lớn hơn top_k để còn lọc
    candidate_k = max(top_k * 5, 30)

    sql = """
        WITH q AS (
            SELECT %s::vector AS embedding_vec
        )
        SELECT
            d.id          AS doc_id,
            d.job_id      AS job_id,
            d.chunk_index AS chunk_index,
            d.content     AS chunk_text,
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

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [query_vec, only_active, candidate_k])
            rows = cur.fetchall()

    raw_results: List[Dict[str, Any]] = []

    for row in rows:
        # với RealDictCursor, row là dict
        if isinstance(row, dict):
            doc_id = row.get("doc_id")
            job_id = row.get("job_id")
            chunk_index = row.get("chunk_index")
            chunk_text = row.get("chunk_text")
            metadata_raw = row.get("metadata")
            score = row.get("score")
        else:
            # fallback nếu không dùng RealDictCursor
            doc_id, job_id, chunk_index, chunk_text, metadata_raw, score = row

        if isinstance(metadata_raw, str):
            try:
                metadata = json.loads(metadata_raw)
            except Exception:
                metadata = {"raw": metadata_raw}
        else:
            metadata = metadata_raw or {}

        raw_results.append(
            {
                "doc_id": doc_id,
                "job_id": job_id,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "metadata": metadata,
                "score": float(score) if score is not None else None,
            }
        )

    logger.info(
        "retrieve_jobs raw: query=%r, candidate_k=%s, only_active=%s, got %d docs",
        query,
        candidate_k,
        only_active,
        len(raw_results),
    )

    # ------------ 2. Lọc theo filters (hybrid) ------------
    filtered: List[Dict[str, Any]] = []
    for d in raw_results:
        meta = d.get("metadata") or {}
        chunk_text = d.get("chunk_text") or ""

        if not _location_pass(meta, f_locations, chunk_text):
            continue
        if not _salary_pass(meta, f_min_salary, f_max_salary):
            continue
        if not _skills_pass(meta, f_skills, chunk_text):
            continue

        filtered.append(d)

    # Nếu lọc xong trống → fallback: dùng list gốc
    final_docs = filtered if filtered else raw_results

    # Sort lại theo score giảm dần, lấy top_k
    final_docs = sorted(
        final_docs,
        key=lambda x: (x.get("score") is not None, x.get("score")),
        reverse=True,
    )[:top_k]

    logger.info(
        "retrieve_jobs final: query=%r, top_k=%s, filters=%s, return %d docs (filtered=%d)",
        query,
        top_k,
        filters,
        len(final_docs),
        len(filtered),
    )

    return final_docs

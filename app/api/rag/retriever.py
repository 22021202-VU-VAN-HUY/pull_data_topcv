from __future__ import annotations
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
from app.db import get_connection
from app.config import settings

logger = logging.getLogger(__name__)

#  EMBEDDING MODEL
_embedding_model: Optional[SentenceTransformer] = None

#  Model dùng cho query (phải trùng với model lúc index).
def get_query_embedding_model() -> SentenceTransformer:
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

#  FILTER HELPERS

def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()

#     Ghép thêm từ khoá / địa điểm đã parse vào câu truy vấn để tăng độ khớp embedding.
def _augment_query_with_filters(query: str, filters: Dict[str, Any]) -> str:
    query = (query or "").strip()
    parts: List[str] = [query] if query else []

    job_keywords = filters.get("job_keywords") or []
    locations = filters.get("locations") or []
    skills = filters.get("skills") or []
    min_salary = filters.get("min_salary_vnd")
    max_salary = filters.get("max_salary_vnd")

    if job_keywords:
        parts.append("Từ khoá: " + ", ".join(job_keywords))
    if locations:
        parts.append("Địa điểm: " + ", ".join(locations))
    if skills:
        parts.append("Kỹ năng: " + ", ".join(skills))
    if min_salary is not None or max_salary is not None:
        parts.append(
            "Lương: "
            + (f">= {min_salary:,} VND" if min_salary is not None else "")
            + (" – " if min_salary is not None and max_salary is not None else "")
            + (f"<= {max_salary:,} VND" if max_salary is not None else "")
        )
    return " | ".join(parts)

# Lọc theo địa điểm:
#    - Nếu không có filter_locations -> luôn pass
#    - Nếu có -> check trong meta["locations"] và text.
def _location_pass(
    meta: Dict[str, Any],
    filter_locations: List[str],
    chunk_text: str,
) -> bool:
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

# Lọc theo khoảng lương.
#    - f_min, f_max: VND, có thể None.
#    - meta["salary"] có thể có min/max, cũng là VND.
def _salary_pass(
    meta: Dict[str, Any],
    f_min: Optional[int],
    f_max: Optional[int],
) -> bool:
    salary = meta.get("salary") or {}
    s_min = salary.get("min")
    s_max = salary.get("max")

    # nếu không có filter lương -> pass
    if f_min is None and f_max is None:
        return True

    # nếu job không có thông tin lương rõ ràng -> vẫn giữ 
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

# Lọc theo kỹ năng: check trong mô tả / yêu cầu / chunk_text.
def _skills_pass(
    meta: Dict[str, Any],
    filter_skills: List[str],
    chunk_text: str,
) -> bool:
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

# Lọc theo chức danh / từ khoá nghề nghiệp để tránh drift sang ngành khác.
def _keyword_pass(meta: Dict[str, Any], job_keywords: List[str], chunk_text: str) -> bool:
    if not job_keywords:
        return True

    title = _normalize_text(meta.get("title"))
    company = _normalize_text(_get_company(meta))
    haystack = " ".join(
        [
            title,
            company,
            _normalize_text(chunk_text),
        ]
    )

    for kw in job_keywords:
        k_norm = _normalize_text(kw)
        if k_norm and k_norm in haystack:
            return True
    return False


def _get_company(meta: Dict[str, Any]) -> str:
    company = meta.get("company")
    if isinstance(company, dict):
        return company.get("name") or ""
    if isinstance(company, str):
        return company
    return ""


#  RETRIEVE
#     Lấy nhanh các chunk thuộc 1 job cụ thể (ưu tiên job_overview, sau đó section)."""

def _fetch_job_docs(job_id: int, limit: int = 6) -> List[Dict[str, Any]]:
    sql = """
        SELECT
            d.id          AS doc_id,
            d.job_id      AS job_id,
            d.chunk_index AS chunk_index,
            d.content     AS chunk_text,
            d.metadata    AS metadata,
            1.0           AS score
        FROM rag_job_documents d
        WHERE d.job_id = %s
        ORDER BY (d.doc_type = 'job_overview') DESC, d.section_type NULLS LAST, d.chunk_index ASC
        LIMIT %s;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, [job_id, limit])
            rows = cur.fetchall() or []
    results: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            doc_id = row.get("doc_id")
            job_id = row.get("job_id")
            chunk_index = row.get("chunk_index")
            chunk_text = row.get("chunk_text")
            metadata_raw = row.get("metadata")
            score = row.get("score")
        else:
            doc_id, job_id, chunk_index, chunk_text, metadata_raw, score = row
        if isinstance(metadata_raw, str):
            try:
                metadata = json.loads(metadata_raw)
            except Exception:
                metadata = {"raw": metadata_raw}
        else:
            metadata = metadata_raw or {}
        results.append(
            {
                "doc_id": doc_id,
                "job_id": job_id,
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
                "metadata": metadata,
                "score": float(score) if score is not None else None,
            }
        )
    return results

# Lấy toàn bộ thông tin 1 job (metadata + các section text) để đưa vào prompt.
# Trả về doc có cấu trúc tương tự kết quả retrieve, phục vụ cho job hiện tại.
def fetch_full_job_detail(job_id: int) -> Optional[Dict[str, Any]]:
    sql_job = """
        SELECT
            j.id AS job_id,
            j.title,
            j.url,
            j.salary_min,
            j.salary_max,
            j.salary_currency,
            j.salary_interval,
            j.salary_raw_text,
            j.experience_raw_text,
            j.cap_bac,
            j.hinh_thuc_lam_viec,
            j.hinh_thuc_lam_viec_raw,
            j.hoc_van,
            j.so_luong_tuyen,
            j.so_luong_tuyen_raw,
            j.deadline,
            c.name AS company_name,
            c.url AS company_url
        FROM jobs j
        LEFT JOIN companies c ON j.company_id = c.id
        WHERE j.id = %s
        LIMIT 1;
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql_job, [job_id])
            row = cur.fetchone()
            if not row:
                return None

            cur.execute(
                """
                SELECT location_text
                FROM job_locations
                WHERE job_id = %s
                ORDER BY is_primary DESC, sort_order, id
                """,
                [job_id],
            )
            loc_rows = cur.fetchall() or []
            locations = [r.get("location_text") for r in loc_rows if r.get("location_text")]

            cur.execute(
                """
                SELECT section_type, text_content, html_content
                FROM job_sections
                WHERE job_id = %s
                ORDER BY id
                """,
                [job_id],
            )
            sec_rows = cur.fetchall() or []
            detail_sections: Dict[str, Any] = {}
            for sr in sec_rows:
                stype = sr.get("section_type")
                text = sr.get("text_content")
                html = sr.get("html_content")
                if stype and (text or html):
                    detail_sections[stype] = {"text": text, "html": html}

    meta = {
        "id": row.get("job_id"),
        "title": row.get("title"),
        "company": {"name": row.get("company_name"), "url": row.get("company_url")},
        "url": row.get("url"),
        "locations": locations,
        "salary": {
            "min": row.get("salary_min"),
            "max": row.get("salary_max"),
            "currency": row.get("salary_currency"),
            "interval": row.get("salary_interval"),
            "raw_text": row.get("salary_raw_text"),
        },
        "general_info": {
            "cap_bac": row.get("cap_bac"),
            "hinh_thuc_lam_viec": row.get("hinh_thuc_lam_viec")
            or row.get("hinh_thuc_lam_viec_raw"),
            "hoc_van": row.get("hoc_van"),
            "so_luong_tuyen": row.get("so_luong_tuyen") or row.get("so_luong_tuyen_raw"),
            "experience": row.get("experience_raw_text"),
            "deadline": row.get("deadline"),
        },
        "detail_sections": detail_sections,
    }

    return {
        "doc_id": f"job-{job_id}-full",
        "job_id": job_id,
        "chunk_index": 0,
        "chunk_text": "",  # nội dung đã nằm trong detail_sections
        "metadata": meta,
        "score": 1.0,
    }


def retrieve_jobs(
    query: str,
    top_k: Optional[int] = None,
    only_active: bool = True,
    filters: Optional[Dict[str, Any]] = None,
    *,
    current_job_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Truy vấn rag_job_documents theo embedding + lọc hybrid (địa điểm, lương, kỹ năng),
    đồng thời ghim job hiện tại (nếu truyền current_job_id).

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
    f_job_keywords: List[str] = filters.get("job_keywords") or []
    pinned_docs: List[Dict[str, Any]] = []

    if current_job_id:
        try:
            pinned_docs = _fetch_job_docs(current_job_id, limit=max(6, top_k or 0))
        except Exception as e:
            logger.warning("Không lấy được doc cho job hiện tại %s: %s", current_job_id, e)

    # ------------ 1. embedding cho query ------------
    augmented_query = _augment_query_with_filters(query, filters)
    query_vec = embed_query(augmented_query)

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

    #  2. Lọc theo filters (hybrid) 
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
        if not _keyword_pass(meta, f_job_keywords, chunk_text):
            continue

        filtered.append(d)

    # Lọc xong trống → fallback: dùng list gốc
    final_docs = filtered if filtered else raw_results
    # Sort lại, lấy top_k
    final_docs = sorted(
        final_docs,
        key=lambda x: (x.get("score") is not None, x.get("score")),
        reverse=True,
    )[:top_k]

    # Luôn ghim doc của job hiện tại (nếu có) lên đầu, tránh trùng doc_id
    if pinned_docs:
        seen_ids = {d.get("doc_id") for d in pinned_docs}
        dedup_tail = [d for d in final_docs if d.get("doc_id") not in seen_ids]
        final_docs = pinned_docs + dedup_tail

    logger.info(
        "retrieve_jobs final: query=%r, augmented=%r, top_k=%s, filters=%s, current_job_id=%s, return %d docs (filtered=%d, pinned=%d)",
        query,
        augmented_query,
        top_k,
        filters,
        current_job_id,
        len(final_docs),
        len(filtered),
        len(pinned_docs),
    )

    return final_docs

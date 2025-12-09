from __future__ import annotations
import json
import logging
import os  
import re
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

from app.config import settings
from app.db import get_connection
from app.api.jobs import SECTION_LABELS  # dùng lại label section từ jobs.py

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = settings.RAG_EMBEDDING_MODEL_NAME
EMBEDDING_BATCH_SIZE = settings.RAG_EMBEDDING_BATCH_SIZE
CHUNK_MAX_CHARS = settings.RAG_CHUNK_MAX_CHARS

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

# Encode list text -> list vector (list[float]), dùng cho cả doc & query.
def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]

# Chuyển list float -> chuỗi literal Postgres vector: '[0.1,0.2,...]'.
# Dùng với embedding_vec::vector.
def _vector_to_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def _to_jsonable(obj):
    if isinstance(obj, Decimal):
        if obj == obj.to_integral_value():
            return int(obj)
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj

#  kết nối db -> lấy job cho snapshot
def _fetch_job_full(cur, job_id: int) -> Dict[str, Any]:
    cur.execute(
        """
        SELECT
            j.id AS job_id,
            j.url,
            j.title,
            j.salary_min,
            j.salary_max,
            j.salary_currency,
            j.salary_interval,
            j.salary_raw_text,
            j.experience_months,
            j.experience_raw_text,
            j.deadline,
            j.cap_bac,
            j.hoc_van,
            j.so_luong_tuyen,
            j.hinh_thuc_lam_viec,
            j.hinh_thuc_lam_viec_raw,
            j.so_luong_tuyen_raw,
            j.crawled_at,

            COALESCE(c.name, '')         AS company_name,
            c.url                        AS company_url,
            c.logo                       AS company_logo,
            c.size                       AS company_size,
            c.industry                   AS company_industry,
            c.address                    AS company_address
        FROM jobs j
        LEFT JOIN companies c ON j.company_id = c.id
        WHERE j.id = %(job_id)s
        """,
        {"job_id": job_id},
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Job {job_id} không tồn tại")
    return row

def _fetch_job_locations(cur, job_id: int) -> List[str]:
    cur.execute(
        """
        SELECT location_text
        FROM job_locations
        WHERE job_id = %(job_id)s
        ORDER BY is_primary DESC, sort_order, id
        """,
        {"job_id": job_id},
    )
    rows = cur.fetchall() or []
    return [r["location_text"] for r in rows]

# Trả về dict: section_type -> {html, text}
def _fetch_job_sections_raw(cur, job_id: int) -> Dict[str, Dict[str, Any]]:
    cur.execute(
        """
        SELECT section_type, text_content, html_content, id
        FROM job_sections
        WHERE job_id = %(job_id)s
        ORDER BY id
        """,
        {"job_id": job_id},
    )
    rows = cur.fetchall() or []
    result: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        html = r.get("html_content")
        text = r.get("text_content")
        if not html and not text:
            continue
        result[r["section_type"]] = {
            "html": html,
            "text": text,
        }
    return result

#  snapshot job / meta
# Chuẩn hoá thông tin lương theo các dạng trên web TopCV.
def _build_salary(job_row: Dict[str, Any]) -> Dict[str, Any]:
    salary_min = job_row.get("salary_min")
    salary_max = job_row.get("salary_max")
    currency = job_row.get("salary_currency") or "VND"
    interval = job_row.get("salary_interval") or "MONTH"
    raw_text = job_row.get("salary_raw_text")
    salary = {
        "min": salary_min,
        "max": salary_max,
        "currency": currency,
        "interval": interval,
        "raw_text": raw_text,
    }

    return salary

def _build_experience(job_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "months": job_row.get("experience_months"),
        "raw_text": job_row.get("experience_raw_text"),
    }

def _build_company(job_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": job_row.get("company_name"),
        "url": job_row.get("company_url"),
        "logo": job_row.get("company_logo"),
        "size": job_row.get("company_size"),
        "industry": job_row.get("company_industry"),
        "address": job_row.get("company_address"),
    }

def _build_general_info(job_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cap_bac": job_row.get("cap_bac"),
        "hoc_van": job_row.get("hoc_van"),
        "so_luong_tuyen": job_row.get("so_luong_tuyen"),
        "hinh_thuc_lam_viec": job_row.get("hinh_thuc_lam_viec"),
        "hinh_thuc_lam_viec_raw": job_row.get("hinh_thuc_lam_viec_raw"),
        "so_luong_tuyen_raw": job_row.get("so_luong_tuyen_raw"),
    }

# form chung cho mọi chunk của 1 job
def build_job_meta(job_row: Dict[str, Any], locations: List[str]) -> Dict[str, Any]:
    deadline = job_row.get("deadline")
    crawled_at = job_row.get("crawled_at")

    now_utc = datetime.now(timezone.utc)
    is_active = bool(deadline and deadline >= now_utc)
    meta = {
        "id": job_row["job_id"],
        "url": job_row["url"],
        "title": job_row["title"],
        "salary": _build_salary(job_row),
        "locations": locations,
        "experience": _build_experience(job_row),
        "company": _build_company(job_row),
        "general_info": _build_general_info(job_row),
        "deadline": deadline.isoformat() if deadline else None,
        "crawled_at": crawled_at.isoformat() if crawled_at else None,
        "is_active": is_active,
        "source": "topcv",
    }
    return meta

#  format text từ meta
def _format_currency_amount(value: Optional[float | int], currency: Optional[str]) -> str:
    if value is None:
        return ""
    cur = currency or "VND"
    try:
        v_int = int(value)
        return f"{v_int:,} {cur}".replace(",", ".")
    except Exception:
        return f"{value} {cur}"

# mô tả lương
def _format_salary_line(salary: Dict[str, Any]) -> Optional[str]:
    raw_text = salary.get("raw_text")
    if raw_text:
        return f"Thu nhập: {raw_text}"
    min_v = salary.get("min")
    max_v = salary.get("max")
    currency = salary.get("currency") or "VND"
    interval = salary.get("interval") or "MONTH"
    if min_v is None and max_v is None:
        return None
    suffix = ""
    if interval.upper() == "MONTH":
        suffix = "/tháng"
    elif interval.upper() == "YEAR":
        suffix = "/năm"
    if min_v is not None and max_v is not None:
        min_str = _format_currency_amount(min_v, currency)
        max_str = _format_currency_amount(max_v, currency)
        return f"Thu nhập: từ {min_str} đến {max_str} {suffix}".strip()
    if min_v is not None:
        min_str = _format_currency_amount(min_v, currency)
        return f"Thu nhập: từ {min_str} {suffix}".strip()
    if max_v is not None:
        max_str = _format_currency_amount(max_v, currency)
        return f"Thu nhập: đến {max_str} {suffix}".strip()
    return None

def overview_meta_to_text(meta: Dict[str, Any]) -> str:
    lines: List[str] = []
    title = meta.get("title") or ""
    lines.append(f"Tiêu đề: {title}")
    company = meta.get("company") or {}
    if company.get("name"):
        lines.append(f"Công ty: {company['name']}")

    locations = meta.get("locations") or []
    if locations:
        lines.append("Địa điểm: " + " | ".join(locations))

    salary_line = _format_salary_line(meta.get("salary") or {})
    if salary_line:
        lines.append(salary_line)

    exp = meta.get("experience") or {}
    if exp.get("raw_text"):
        lines.append(f"Kinh nghiệm: {exp['raw_text']}")
    elif exp.get("months") is not None:
        lines.append(f"Kinh nghiệm: từ {exp['months']} tháng trở lên")

    gi = meta.get("general_info") or {}
    if gi.get("cap_bac"):
        lines.append(f"Cấp bậc: {gi['cap_bac']}")
    if gi.get("hoc_van"):
        lines.append(f"Học vấn: {gi['hoc_van']}")
    if gi.get("so_luong_tuyen"):
        lines.append(f"Số lượng tuyển: {gi['so_luong_tuyen']}")
    if gi.get("hinh_thuc_lam_viec"):
        lines.append(f"Hình thức làm việc: {gi['hinh_thuc_lam_viec']}")

    if meta.get("deadline"):
        lines.append(f"Hạn nộp: {meta['deadline']}")
    return "\n".join(lines)

# text cho 1 chunk section
def section_meta_to_text(meta: Dict[str, Any], section_type: str, chunk_text: str) -> str:
    lines: List[str] = []
    title = meta.get("title") or ""
    lines.append(f"Công việc: {title}")
    company = meta.get("company") or {}
    if company.get("name"):
        lines.append(f"Công ty: {company['name']}")

    locations = meta.get("locations") or []
    if locations:
        lines.append("Địa điểm: " + " | ".join(locations))

    salary_line = _format_salary_line(meta.get("salary") or {})
    if salary_line:
        lines.append(salary_line)

    label = SECTION_LABELS.get(section_type, section_type.replace("_", " ").title())
    lines.append(f"Mục: {label}")

    if meta.get("deadline"):
        lines.append(f"Hạn nộp: {meta['deadline']}")
    lines.append(f"Nội dung: {chunk_text}")
    return "\n".join(lines)

#  chunk text
def split_text_into_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    clean = (text or "").strip()
    if not clean:
        return []

    if len(clean) <= max_chars:
        return [clean]
    sentences = re.split(r"(?<=[\.!?])\s+", clean)
    chunks: List[str] = []
    current = ""

    for sent in sentences:
        if not sent:
            continue
        if len(current) + 1 + len(sent) <= max_chars:
            if current:
                current += " " + sent
            else:
                current = sent
        else:
            if current:
                chunks.append(current)
            current = sent

    if current:
        chunks.append(current)

    return chunks

#  index 1 job thành N docs
def upsert_rag_doc_for_job(job_id: int) -> int:
    """
    Index 1 job thành nhiều document:
      - 1 doc_type = 'job_overview', chunk_index = 0
      - N doc_type = 'job_section' cho từng section/chunk
    Trả về số document đã insert.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            job_row = _fetch_job_full(cur, job_id)
            locations = _fetch_job_locations(cur, job_id)
            sections_raw = _fetch_job_sections_raw(cur, job_id)

            job_meta = build_job_meta(job_row, locations)

            # Xoá toàn bộ doc cũ của job này để insert lại sạch sẽ
            cur.execute(
                "DELETE FROM rag_job_documents WHERE job_id = %(job_id)s",
                {"job_id": job_id},
            )

            docs_count = 0

            # 1) OVERVIEW DOC
            overview_meta = dict(job_meta)  # shallow copy
            overview_meta_json = _to_jsonable(overview_meta)
            overview_content = overview_meta_to_text(overview_meta_json)
            overview_vec = embed_texts([overview_content])[0]

            cur.execute(
                """
                INSERT INTO rag_job_documents (
                    job_id,
                    doc_type,
                    section_type,
                    chunk_index,
                    content,
                    metadata,
                    embedding_vec
                )
                VALUES (
                    %(job_id)s,
                    %(doc_type)s,
                    %(section_type)s,
                    %(chunk_index)s,
                    %(content)s,
                    %(metadata)s::jsonb,
                    %(embedding_vec)s::vector
                )
                """,
                {
                    "job_id": job_id,
                    "doc_type": "job_overview",
                    "section_type": None,
                    "chunk_index": 0,
                    "content": overview_content,
                    "metadata": json.dumps(overview_meta_json, ensure_ascii=False),
                    "embedding_vec": _vector_to_literal(overview_vec),
                },
            )
            docs_count += 1

            # 2) SECTION DOCS
            for section_type, sec in sections_raw.items():
                full_text = (sec or {}).get("text") or ""
                full_text = full_text.strip()
                if not full_text:
                    continue
                chunks = split_text_into_chunks(full_text, max_chars=CHUNK_MAX_CHARS)
                html = (sec or {}).get("html")

                for idx, chunk_text in enumerate(chunks):
                    section_meta = dict(job_meta)
                    label = SECTION_LABELS.get(section_type, section_type.replace("_", " ").title())
                    section_meta["section"] = {
                        "type": section_type,
                        "label": label,
                        "html": html,
                        "text": full_text,
                        "chunk_index": idx,
                        "chunk_text": chunk_text,
                    }
                    section_meta_json = _to_jsonable(section_meta)
                    section_content = section_meta_to_text(section_meta_json, section_type, chunk_text)
                    section_vec = embed_texts([section_content])[0]

                    cur.execute(
                        """
                        INSERT INTO rag_job_documents (
                            job_id,
                            doc_type,
                            section_type,
                            chunk_index,
                            content,
                            metadata,
                            embedding_vec
                        )
                        VALUES (
                            %(job_id)s,
                            %(doc_type)s,
                            %(section_type)s,
                            %(chunk_index)s,
                            %(content)s,
                            %(metadata)s::jsonb,
                            %(embedding_vec)s::vector
                        )
                        """,
                        {
                            "job_id": job_id,
                            "doc_type": "job_section",
                            "section_type": section_type,
                            "chunk_index": idx,
                            "content": section_content,
                            "metadata": json.dumps(section_meta_json, ensure_ascii=False),
                            "embedding_vec": _vector_to_literal(section_vec),
                        },
                    )
                    docs_count += 1
        conn.commit()
    logger.info("Indexed job %s (%s docs)", job_id, docs_count)
    return docs_count

#  BATCH INDEX NHIỀU JOB
def select_job_ids_to_index(cur, limit: Optional[int] = None, reindex: bool = False) -> List[int]:
    """
    - reindex=False: chỉ lấy job chưa có job_overview trong rag_job_documents
    - reindex=True: lấy tất cả jobs còn hạn
    """

    now_utc = datetime.now(timezone.utc)

    if reindex:
        sql = """
            WITH last_index AS (
                SELECT job_id,
                       metadata ->> 'is_active' AS is_active
                FROM rag_job_documents
                WHERE doc_type = 'job_overview'
                  AND chunk_index = 0
            )
            SELECT j.id
            FROM jobs j
            LEFT JOIN last_index d ON d.job_id = j.id
            WHERE (j.deadline IS NULL OR j.deadline >= %(now)s)
               OR (j.deadline < %(now)s AND d.is_active = 'true')
            ORDER BY j.id
            LIMIT %(limit)s
        """
    else:
        sql = """
            SELECT j.id
            FROM jobs j
            WHERE (
                (j.deadline IS NULL OR j.deadline >= %(now)s)
                AND NOT EXISTS (
                    SELECT 1
                    FROM rag_job_documents d
                    WHERE d.job_id = j.id
                      AND d.doc_type = 'job_overview'
                      AND d.chunk_index = 0
                )
            )
               OR (
                   j.deadline < %(now)s
                   AND EXISTS (
                       SELECT 1
                       FROM rag_job_documents d
                       WHERE d.job_id = j.id
                         AND d.doc_type = 'job_overview'
                         AND d.chunk_index = 0
                         AND d.metadata ->> 'is_active' = 'true'
                   )
               )
            ORDER BY j.id
            LIMIT %(limit)s
        """

    cur.execute(sql, {"limit": limit or 10_000_000, "now": now_utc})
    rows = cur.fetchall() or []
    return [r["id"] for r in rows]


def index_all_jobs(limit: Optional[int] = None, reindex: bool = False) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            job_ids = select_job_ids_to_index(cur, limit=limit, reindex=reindex)

    logger.info("Bắt đầu index %d jobs (reindex=%s)", len(job_ids), reindex)

    for job_id in job_ids:
        try:
            upsert_rag_doc_for_job(job_id)
        except Exception as e:
            logger.exception("Lỗi index job %s: %s", job_id, e)

# CLI
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Build RAG index cho jobs → rag_job_documents")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số job cần index")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Nếu set, sẽ index lại tất cả jobs",
    )
    args = parser.parse_args()

    index_all_jobs(limit=args.limit, reindex=args.reindex)

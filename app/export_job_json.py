# app/export_job_json.py
# 1) Xuất job mới nhất: python -m app.export_job_json
# 2) Xuất theo job_id: python -m app.export_job_json --job-id 1
# 3) Xuất theo URL: python -m app.export_job_json --url "https://www.topcv.vn/viec-lam/..."


import argparse
import json
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional, List

import psycopg2
from psycopg2.extras import RealDictCursor

from .config import settings

def get_connection():
    """
    Kết nối Postgres, dùng thông tin từ .env (config settings).
    """
    conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
    )
    return conn


# --------------------- LẤY JOB CHÍNH ---------------------


def fetch_job_row(
    conn,
    job_id: Optional[int] = None,
    url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Lấy 1 job + company (join) theo job_id hoặc url.
    Nếu không truyền gì -> lấy job mới nhất (theo crawled_at).
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        base_select = """
            SELECT
                j.*,
                c.id   AS company_id,
                c.name AS company_name,
                c.url  AS company_url,
                c.logo AS company_logo,
                c.size AS company_size,
                c.industry AS company_industry,
                c.address  AS company_address
            FROM jobs j
            LEFT JOIN companies c ON j.company_id = c.id
        """

        if job_id is not None:
            cur.execute(base_select + " WHERE j.id = %s", (job_id,))
        elif url is not None:
            cur.execute(base_select + " WHERE j.url = %s", (url,))
        else:
            cur.execute(
                base_select
                + " ORDER BY j.crawled_at DESC NULLS LAST, j.id DESC LIMIT 1"
            )

        row = cur.fetchone()
        return dict(row) if row else None


# --------------------- LẤY ĐỊA ĐIỂM ---------------------


def fetch_locations(conn, job_id: int) -> List[str]:
    """
    Lấy list địa điểm từ bảng job_locations.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT location_text
            FROM job_locations
            WHERE job_id = %s
            ORDER BY is_primary DESC, sort_order, id
            """,
            (job_id,),
        )
        return [r[0] for r in cur.fetchall()]


# --------------------- LẤY SECTIONS ---------------------


def fetch_sections(conn, job_id: int) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Lấy các section (Mô tả, Yêu cầu, Thu nhập, ...) từ job_sections
    và trả về dạng dict:

        {
          "mo_ta_cong_viec": {"html": ..., "text": ...},
          "yeu_cau_ung_vien": {...},
          ...
        }

    Dựa trên schema:

        job_sections(
            id BIGSERIAL,
            job_id BIGINT,
            section_type VARCHAR(50),
            text_content TEXT,
            html_content TEXT,
            crawled_at TIMESTAMPTZ
        )
    """
    sections: Dict[str, Dict[str, Optional[str]]] = {}

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT section_type, text_content, html_content, id
            FROM job_sections
            WHERE job_id = %s
            ORDER BY id
            """,
            (job_id,),
        )
        for row in cur.fetchall():
            key = row["section_type"]  # ví dụ: 'mo_ta_cong_viec'
            sections[key] = {
                "html": row.get("html_content"),
                "text": row.get("text_content"),
            }

    return sections


# --------------------- UTIL ---------------------


def to_iso(dt: Optional[datetime]) -> Optional[str]:
    """
    Chuyển datetime -> ISO string, nếu None thì trả None.
    """
    if dt is None:
        return None
    return dt.isoformat()


def to_int_if_decimal(x: Any) -> Optional[int]:
    """
    Nếu x là Decimal -> convert sang int.
    Nếu x là None -> None.
    Ngược lại trả x (không đụng).
    """
    if x is None:
        return None
    if isinstance(x, Decimal):
        # vì schema dùng NUMERIC(18,0) nên có thể cast sang int an toàn
        return int(x)
    return x


# --------------------- BUILD JSON ---------------------


def build_job_json(row: Dict[str, Any], locations, sections) -> Dict[str, Any]:
    """
    Gộp tất cả dữ liệu (job row + locations + sections) thành JSON structure.
    Dựa chuẩn JSON mà crawler của bạn đã dùng trước đây.
    """
    job_id = row["id"]

    # Salary (convert Decimal -> int nếu có)
    salary_min = to_int_if_decimal(row.get("salary_min"))
    salary_max = to_int_if_decimal(row.get("salary_max"))

    salary = {
        "min": salary_min,
        "max": salary_max,
        "currency": row.get("salary_currency"),
        "interval": row.get("salary_interval"),
        "raw_text": row.get("salary_raw_text"),
    }

    # Experience
    experience = {
        "months": row.get("experience_months"),
        "raw_text": row.get("experience_raw_text"),
    }

    # Company
    company = {
        "name": row.get("company_name"),
        "url": row.get("company_url"),
        "logo": row.get("company_logo"),
        "size": row.get("company_size"),
        "industry": row.get("company_industry"),
        "address": row.get("company_address"),
    }

    # General info
    general_info = {
        "cap_bac": row.get("cap_bac"),
        "hoc_van": row.get("hoc_van"),
        "so_luong_tuyen": row.get("so_luong_tuyen"),
        "hinh_thuc_lam_viec": row.get("hinh_thuc_lam_viec"),
        "hinh_thuc_lam_viec_raw": row.get("hinh_thuc_lam_viec_raw"),
        "so_luong_tuyen_raw": row.get("so_luong_tuyen_raw"),
    }

    job_json = {
        "id": job_id,
        "url": row.get("url"),
        "title": row.get("title"),
        "salary": salary,
        "locations": locations,
        "experience": experience,
        "detail_sections": sections,
        "deadline": to_iso(row.get("deadline")),
        "company": company,
        "general_info": general_info,
        "crawled_at": to_iso(row.get("crawled_at")),
    }

    return job_json


# --------------------- HÀM CHÍNH EXPORT ---------------------


def export_job(job_id: Optional[int] = None, url: Optional[str] = None) -> Dict[str, Any]:
    """
    Hàm chính: lấy 1 job từ DB, dựng JSON, và ghi file.
    """
    conn = get_connection()
    try:
        row = fetch_job_row(conn, job_id=job_id, url=url)
        if not row:
            raise ValueError("Không tìm thấy job phù hợp (job_id hoặc url).")

        job_id = row["id"]
        locations = fetch_locations(conn, job_id)
        sections = fetch_sections(conn, job_id)
        job_json = build_job_json(row, locations, sections)

        # Ghi ra file
        filename = f"export_job_{job_id}.json"
        with open(filename, "w", encoding="utf-8-sig") as f:
            json.dump(job_json, f, ensure_ascii=False, indent=2, default=str)

        print(f"Đã export job_id={job_id} -> {filename}")
        print()
        print(json.dumps(job_json, ensure_ascii=False, indent=2, default=str))

        return job_json

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Export 1 job từ DB ra JSON.")
    parser.add_argument(
        "--job-id",
        type=int,
        help="ID của job trong DB (ưu tiên nếu truyền vào).",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="URL job trong DB.",
    )
    args = parser.parse_args()

    export_job(job_id=args.job_id, url=args.url)


if __name__ == "__main__":
    main()

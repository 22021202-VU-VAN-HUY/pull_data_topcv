# app/db.py

import psycopg2
from psycopg2.extras import RealDictCursor
from .config import settings


def get_connection():
    conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        cursor_factory=RealDictCursor,
    )
    return conn


# ================== UPSERT HELPERS ==================


def upsert_company(cur, company_row: dict) -> int:
    """
    Insert hoặc update company theo url.
    Trả về company_id.
    company_row phải có: name, url, logo, size, industry, address
    """
    cur.execute(
        """
        INSERT INTO companies (
            name, url, logo, size, industry, address
        )
        VALUES (
            %(name)s,
            %(url)s,
            %(logo)s,
            %(size)s,
            %(industry)s,
            %(address)s
        )
        ON CONFLICT (url) DO UPDATE
        SET
            name       = EXCLUDED.name,
            logo       = EXCLUDED.logo,
            size       = EXCLUDED.size,
            industry   = EXCLUDED.industry,
            address    = EXCLUDED.address,
            updated_at = NOW()
        RETURNING id;
        """,
        company_row,
    )
    row = cur.fetchone()
    return row["id"]


def upsert_job(cur, job_row: dict) -> int:
    """
    Insert hoặc update job theo url.
    Trả về job_id.
    job_row phải có:
        company_id, url, title,
        salary_min, salary_max, salary_currency, salary_interval, salary_raw_text,
        experience_months, experience_raw_text,
        deadline,
        cap_bac, hoc_van,
        so_luong_tuyen, hinh_thuc_lam_viec,
        hinh_thuc_lam_viec_raw, so_luong_tuyen_raw,
        crawled_at
    """
    cur.execute(
        """
        INSERT INTO jobs (
            company_id,
            url,
            title,
            salary_min,
            salary_max,
            salary_currency,
            salary_interval,
            salary_raw_text,
            experience_months,
            experience_raw_text,
            deadline,
            cap_bac,
            hoc_van,
            so_luong_tuyen,
            hinh_thuc_lam_viec,
            hinh_thuc_lam_viec_raw,
            so_luong_tuyen_raw,
            crawled_at
        )
        VALUES (
            %(company_id)s,
            %(url)s,
            %(title)s,
            %(salary_min)s,
            %(salary_max)s,
            %(salary_currency)s,
            %(salary_interval)s,
            %(salary_raw_text)s,
            %(experience_months)s,
            %(experience_raw_text)s,
            %(deadline)s,
            %(cap_bac)s,
            %(hoc_van)s,
            %(so_luong_tuyen)s,
            %(hinh_thuc_lam_viec)s,
            %(hinh_thuc_lam_viec_raw)s,
            %(so_luong_tuyen_raw)s,
            %(crawled_at)s
        )
        ON CONFLICT (url) DO UPDATE
        SET
            company_id             = EXCLUDED.company_id,
            title                  = EXCLUDED.title,
            salary_min             = EXCLUDED.salary_min,
            salary_max             = EXCLUDED.salary_max,
            salary_currency        = EXCLUDED.salary_currency,
            salary_interval        = EXCLUDED.salary_interval,
            salary_raw_text        = EXCLUDED.salary_raw_text,
            experience_months      = EXCLUDED.experience_months,
            experience_raw_text    = EXCLUDED.experience_raw_text,
            deadline               = EXCLUDED.deadline,
            cap_bac                = EXCLUDED.cap_bac,
            hoc_van                = EXCLUDED.hoc_van,
            so_luong_tuyen         = EXCLUDED.so_luong_tuyen,
            hinh_thuc_lam_viec     = EXCLUDED.hinh_thuc_lam_viec,
            hinh_thuc_lam_viec_raw = EXCLUDED.hinh_thuc_lam_viec_raw,
            so_luong_tuyen_raw     = EXCLUDED.so_luong_tuyen_raw,
            crawled_at             = EXCLUDED.crawled_at,
            updated_at             = NOW()
        RETURNING id;
        """,
        job_row,
    )
    row = cur.fetchone()
    return row["id"]

def get_cursor(conn):
    """
    Trả về cursor kiểu RealDictCursor (row là dict thay vì tuple).
    """
    return conn.cursor(cursor_factory=RealDictCursor)

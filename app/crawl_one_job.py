# app/crawl_one_job.py
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from .config import settings 
from .db import get_connection, get_cursor
from .topcv_parser import parse_job

# thêm hoặc update thông tin công ty
def upsert_company(conn, cur, company_data: Dict[str, Any]) -> int:
    cur.execute(
        """
        INSERT INTO companies (name, url, logo, size, industry, address)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO UPDATE
        SET
            name       = EXCLUDED.name,
            logo       = EXCLUDED.logo,
            size       = EXCLUDED.size,
            industry   = EXCLUDED.industry,
            address    = EXCLUDED.address,
            updated_at = NOW()
        RETURNING id
        """,
        (
            company_data.get("name"),
            company_data.get("url"),
            company_data.get("logo"),
            company_data.get("size"),
            company_data.get("industry"),
            company_data.get("address"),
        ),
    )
    row = cur.fetchone()
    return row["id"]

# thêm hoặc update job
def upsert_job(conn, cur, job_data: Dict[str, Any], company_id: int, crawled_at) -> int:
    g = job_data["general_info"]
    salary = job_data["salary"]
    exp = job_data["experience"]

    cur.execute(
        """
        INSERT INTO jobs (
          company_id, url, title,
          salary_min, salary_max, salary_currency, salary_interval, salary_raw_text,
          experience_months, experience_raw_text,
          deadline,
          cap_bac, hoc_van, so_luong_tuyen,
          hinh_thuc_lam_viec, hinh_thuc_lam_viec_raw,
          so_luong_tuyen_raw,
          crawled_at
        )
        VALUES (
          %s, %s, %s,
          %s, %s, %s, %s, %s,
          %s, %s,
          %s,
          %s, %s, %s,
          %s, %s,
          %s,
          %s
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
        RETURNING id
        """,
        (
            company_id,
            job_data["url"],
            job_data["title"],
            salary.get("min"),
            salary.get("max"),
            salary.get("currency"),
            salary.get("interval"),
            salary.get("raw_text"),
            exp.get("months"),
            exp.get("raw_text"),
            job_data.get("deadline"),
            g.get("cap_bac"),
            g.get("hoc_van"),
            g.get("so_luong_tuyen"),
            g.get("hinh_thuc_lam_viec"),
            g.get("hinh_thuc_lam_viec_raw"),
            g.get("so_luong_tuyen_raw"),
            crawled_at,
        ),
    )
    job_id = cur.fetchone()["id"]
    return job_id

# cập nhật location mới
def insert_locations(conn, cur, job_id: int, locations):
    cur.execute(
        "DELETE FROM job_locations WHERE job_id = %s",
        (job_id,),
    )
    for idx, loc in enumerate(locations):
        cur.execute(
            """
            INSERT INTO job_locations (job_id, location_text, is_primary, sort_order)
            VALUES (%s, %s, %s, %s)
            """,
            (job_id, loc, idx == 0, idx),
        )

# cập nhật section mới
def insert_sections(conn, cur, job_id: int, detail_sections: Dict[str, Any], crawled_at):
    cur.execute(
        "DELETE FROM job_sections WHERE job_id = %s",
        (job_id,),
    )
    for section_type, sec in detail_sections.items():
        if not sec:
            continue
        cur.execute(
            """
            INSERT INTO job_sections (job_id, section_type, text_content, html_content, crawled_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                job_id,
                section_type,
                sec.get("text"),
                sec.get("html"),
                crawled_at,
            ),
        )

# crawl. lưu 1 job
def crawl_and_save_one_job(job_url: str, seq: Optional[int] = None):
    job_data = parse_job(job_url)
    crawled_at = datetime.now(timezone.utc)

    conn = get_connection()
    cur = get_cursor(conn)
    try:
        company_id = upsert_company(conn, cur, job_data["company"])
        job_id = upsert_job(conn, cur, job_data, company_id, crawled_at)
        insert_locations(conn, cur, job_id, job_data.get("locations", []))
        insert_sections(conn, cur, job_id, job_data.get("detail_sections", {}), crawled_at)
        conn.commit()

        # console
        if seq is not None:
            print(f"Đã lưu job {seq} - id {job_id} từ url={job_url}")
        else:
            print(f"Đã lưu job - id {job_id} từ url={job_url}")

    except Exception as e:
        conn.rollback()
        print(f"Lỗi crawl {job_url}: {e}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    #test_url = "https://www.topcv.vn/viec-lam/nhan-vien-kinh-doanh-sale-mang-game-ca-chieu-13h45-23h-tu-thu-2-thu-6-thu-nhap-tu-14-17-trieu/1713005.html"
    test_url = "https://www.topcv.vn/viec-lam/data-engineer/1921346.html"
    crawl_and_save_one_job(test_url)

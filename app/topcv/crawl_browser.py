
import argparse
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

from app.config import settings
from app.db import get_connection, get_cursor
from app.topcv.crawl_one_job import (
    insert_locations,
    insert_sections,
    upsert_company,
    upsert_job,
)
from app.topcv.topcv_parser import parse_job_from_html

# Tải HTML job bằng Playwright (headless)
async def fetch_job_html(job_url: str) -> str:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=settings.PLAYWRIGHT_HEADLESS)
        page = await browser.new_page()

        nav_timeout = settings.PLAYWRIGHT_NAV_TIMEOUT_MS
        page.set_default_timeout(nav_timeout)

        await page.goto(
            job_url,
            wait_until="domcontentloaded",
            timeout=nav_timeout,
        )

        try:
            await page.wait_for_load_state(
                "networkidle",
                timeout=min(5000, nav_timeout),
            )
        except PlaywrightTimeoutError:
            print("[WARN] Network idle chua dat, tiep tuc sau DOMContentLoaded")

        if settings.TOPCV_BROWSER_WAIT_SELECTOR:
            try:
                await page.wait_for_selector(
                    settings.TOPCV_BROWSER_WAIT_SELECTOR,
                    timeout=nav_timeout,
                )
            except PlaywrightTimeoutError:
                # Selector không bắt buộc, chỉ log để debug khi thiếu dữ liệu.
                print(
                    "[WARN] Không thấy selector",
                    settings.TOPCV_BROWSER_WAIT_SELECTOR,
                    "trước khi đọc HTML",
                )

        if settings.PLAYWRIGHT_EXTRA_WAIT_MS > 0:
            await page.wait_for_timeout(settings.PLAYWRIGHT_EXTRA_WAIT_MS)

        content = await page.content()
        await browser.close()
        return content


def _normalize_job_fields(job_data: Dict[str, Any]) -> Dict[str, Any]:

    job_data.setdefault("salary", {})
    job_data.setdefault("experience", {})
    job_data.setdefault("general_info", {})
    job_data.setdefault("company", {})
    job_data.setdefault("detail_sections", {})
    job_data.setdefault("locations", [])

    salary = job_data["salary"]
    salary.setdefault("min", None)
    salary.setdefault("max", None)
    salary.setdefault("currency", None)
    salary.setdefault("interval", None)
    salary.setdefault("raw_text", None)

    exp = job_data["experience"]
    exp.setdefault("months", None)
    exp.setdefault("raw_text", None)

    gen = job_data["general_info"]
    gen.setdefault("cap_bac", None)
    gen.setdefault("hoc_van", None)
    gen.setdefault("so_luong_tuyen", None)
    gen.setdefault("hinh_thuc_lam_viec", None)
    gen.setdefault("hinh_thuc_lam_viec_raw", None)
    gen.setdefault("so_luong_tuyen_raw", None)

    company = job_data["company"]
    company.setdefault("name", None)
    company.setdefault("url", None)
    company.setdefault("logo", None)
    company.setdefault("size", None)
    company.setdefault("industry", None)
    company.setdefault("address", None)

    return job_data


def save_job_to_db(job_data: Dict[str, Any], job_url: str, seq: Optional[int] = None):
    job_data = _normalize_job_fields(job_data)
    crawled_at = datetime.now(timezone.utc)

    conn = get_connection()
    cur = get_cursor(conn)
    try:
        company_id = upsert_company(conn, cur, job_data["company"])
        job_id = upsert_job(conn, cur, job_data, company_id, crawled_at)
        insert_locations(conn, cur, job_id, job_data.get("locations", []))
        insert_sections(
            conn,
            cur,
            job_id,
            job_data.get("detail_sections", {}),
            crawled_at,
        )
        conn.commit()

        if seq is not None:
            print(f"Đã lưu job {seq} - id {job_id} từ url={job_url}")
        else:
            print(f"Đã lưu job - id {job_id} từ url={job_url}")

    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


async def crawl_job_with_browser(job_url: str, seq: Optional[int] = None):
    html = await fetch_job_html(job_url)
    job_data = parse_job_from_html(html, job_url)
    # Bỏ mục Thu nhập khi crawl bằng headless browser (set null)
    detail_sections = job_data.get("detail_sections") or {}
    if "thu_nhap" in detail_sections:
        detail_sections["thu_nhap"] = {"html": None, "text": None}
    save_job_to_db(job_data, job_url, seq=seq)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Crawl 1 job bằng Playwright")
    parser.add_argument("--url", help="URL job cần crawl", default=None)
    args = parser.parse_args(argv)

    job_url = args.url or settings.TOPCV_BROWSER_JOB_URL
    if not job_url:
        print("[ERROR] Hãy truyền --url hoặc cấu hình TOPCV_BROWSER_JOB_URL trong .env")
        return 1

    try:
        asyncio.run(crawl_job_with_browser(job_url))
    except Exception as exc:  # pragma: no cover - log để debug nhanh
        print(f"[ERROR] Crawl headless lỗi: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

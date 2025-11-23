# app/crawl_batch_jobs.py
import time
import random
from typing import List, Set
import requests
from xml.etree import ElementTree as ET

from .config import settings
from .crawl_one_job import crawl_and_save_one_job

SITEMAP_ROOT_URL = settings.TOPCV_SITEMAP_ROOT
SITEMAP_MAX_JOBS = settings.SITEMAP_MAX_JOBS
JOB_MAX_RETRY = settings.JOB_MAX_RETRY
CRAWL_SLEEP_SECONDS = settings.CRAWL_SLEEP_SECONDS

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# hàm đọc sitemap 
def fetch_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text

# Parse (sitemap.xml) để lấy list link các sitemap con.
def parse_sitemap_index(xml_text: str) -> List[str]:
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(xml_text)
    locs: List[str] = []
    for sm in root.findall("sm:sitemap", ns):
        loc_el = sm.find("sm:loc", ns)
        if loc_el is not None and loc_el.text:
            locs.append(loc_el.text.strip())
    return locs

# Parse 1 sitemap con để lấy url job.
def parse_sitemap_urls(xml_text: str) -> List[str]:
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(xml_text)
    urls: List[str] = []
    for u in root.findall("sm:url", ns):
        loc_el = u.find("sm:loc", ns)
        if loc_el is not None and loc_el.text:
            url = loc_el.text.strip()
            if "/viec-lam/" in url:
                urls.append(url)
    return urls

# Đọc sitemap gốc,  từ đó đọc sitemap con, thu thập các URL job. Trả về list URL.
def collect_job_urls(limit: int) -> List[str]:
    print(f"[SITEMAP] Load root: {SITEMAP_ROOT_URL}")
    root_xml = fetch_text(SITEMAP_ROOT_URL)
    children = parse_sitemap_index(root_xml)
    print(f"[SITEMAP] found {len(children)} sitemap children")

    job_urls: List[str] = []
    seen: Set[str] = set()
    priority_keywords = [
        "featured_job_list",
        "job_predefined_titles",
        "jobs_0",
        "jobs_1",
        "jobs_2",
        "jobs_3",
    ]

    def sort_key(u: str) -> int:
        for i, kw in enumerate(priority_keywords):
            if kw in u:
                return i
        return len(priority_keywords)

    children_sorted = sorted(children, key=sort_key)
    for sm_url in children_sorted:
        # chỉ lấy các sitemap liên quan đến job
        if not any(
            kw in sm_url
            for kw in (
                "featured_job_list",
                "job_predefined_titles",
                "jobs_",
            )
        ):
            continue

        print(f"[SITEMAP] read: {sm_url}")
        try:
            xml_text = fetch_text(sm_url)
            urls = parse_sitemap_urls(xml_text)
        except Exception as e:
            print(f"[SITEMAP] ERROR reading {sm_url}: {e}")
            continue
        for u in urls:
            if u not in seen:
                seen.add(u)
                job_urls.append(u)
                if len(job_urls) >= limit:
                    print(f"Lấy được {len(job_urls)} job urls từ sitemap")
                    return job_urls

    print(f"Lấy được {len(job_urls)} job urls từ sitemap")
    return job_urls

# Crawl jobs
def crawl_many_jobs_from_sitemap():
    job_urls = collect_job_urls(SITEMAP_MAX_JOBS)
    if not job_urls:
        print("lỗi crawl - Không tìm thấy job nào từ sitemap.")
        return

    random.shuffle(job_urls)
    total = len(job_urls)
    print(f"Tổng job URLs sẽ crawl: {total},   mỗi job retry tối đa: {JOB_MAX_RETRY}")

    for i, url in enumerate(job_urls, start=1):
        print(f"\n[job {i}/{total}] {url}")
        attempt = 0
        while attempt < JOB_MAX_RETRY:
            attempt += 1
            print(f"Lần {attempt}/{JOB_MAX_RETRY}")
            try:
                crawl_and_save_one_job(url, seq=i)
                break
            except Exception as e:
                # In lỗi
                print(
                    f"  [ERROR] Crawl lỗi (lần {attempt}): {e}\n"
                )
                if attempt < JOB_MAX_RETRY:
                    sleep_s = CRAWL_SLEEP_SECONDS
                    print(
                        f"  -> Thử lại lần {attempt} sau {sleep_s:.1f}s"
                    )
                    try:
                        time.sleep(sleep_s)
                    except KeyboardInterrupt:
                        print("  -> Bị ngắt")
                        return
                else:
                    print("  -> Số lần thử tối đa, crawl fail")
        try:
            time.sleep(CRAWL_SLEEP_SECONDS)
        except KeyboardInterrupt:
            print("Thoát.")
            return


def main():
    crawl_many_jobs_from_sitemap()


if __name__ == "__main__":
    main()

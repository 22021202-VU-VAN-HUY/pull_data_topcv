# app/crawl_batch_jobs.py
import time
import random
from typing import List, Set
import requests
from xml.etree import ElementTree as ET

from .config import settings
from .crawl_one_job import crawl_and_save_one_job


# ================== CẤU HÌNH TỪ .env / config ==================

# URL sitemap gốc của TopCV
SITEMAP_ROOT_URL = getattr(
    settings,
    "TOPCV_SITEMAP_ROOT",
    "https://www.topcv.vn/sitemap.xml",
)

# Số job tối đa sẽ crawl mỗi lần (có thể set trong .env: SITEMAP_MAX_JOBS=5000)
SITEMAP_MAX_JOBS = int(getattr(settings, "SITEMAP_MAX_JOBS", 2000))

# Số lần retry nếu crawl 1 job bị lỗi (HTTPError, DB lỗi, v.v.)
JOB_MAX_RETRY = int(getattr(settings, "JOB_MAX_RETRY", 3))

# Thời gian nghỉ giữa các job (giây), để tránh bị 429 Too Many Requests
CRAWL_SLEEP_SECONDS = float(getattr(settings, "CRAWL_SLEEP_SECONDS", 5.0))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ================== HÀM ĐỌC SITEMAP ==================


def fetch_text(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_sitemap_index(xml_text: str) -> List[str]:
    """
    Parse sitemap index (sitemap.xml) để lấy list link tới các sitemap con.
    """
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(xml_text)
    locs: List[str] = []
    for sm in root.findall("sm:sitemap", ns):
        loc_el = sm.find("sm:loc", ns)
        if loc_el is not None and loc_el.text:
            locs.append(loc_el.text.strip())
    return locs


def parse_sitemap_urls(xml_text: str) -> List[str]:
    """
    Parse 1 sitemap con (jobs_x.xml) để lấy các url job.
    """
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(xml_text)
    urls: List[str] = []
    for u in root.findall("sm:url", ns):
        loc_el = u.find("sm:loc", ns)
        if loc_el is not None and loc_el.text:
            url = loc_el.text.strip()
            # Chỉ giữ các link job
            if "/viec-lam/" in url:
                urls.append(url)
    return urls


def collect_job_urls_from_sitemap(limit: int) -> List[str]:
    """
    Đọc sitemap gốc, sau đó lần lượt đọc các sitemap con
    (featured_job_list, job_predefined_titles, jobs_0, jobs_1, ...)
    để thu thập các URL job.

    Trả về list URL (đã unique).
    """
    print(f"[SITEMAP] Load root: {SITEMAP_ROOT_URL}")
    root_xml = fetch_text(SITEMAP_ROOT_URL)
    children = parse_sitemap_index(root_xml)
    print(f"[SITEMAP] found {len(children)} sitemap children")

    job_urls: List[str] = []
    seen: Set[str] = set()

    # Ưu tiên một số sitemap có vẻ quan trọng
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
                    print(f"[SITEMAP] collected {len(job_urls)} job urls from sitemap")
                    return job_urls

    print(f"[SITEMAP] collected {len(job_urls)} job urls from sitemap")
    return job_urls


# ================== HÀM CRAWL NHIỀU JOB ==================


def crawl_many_jobs_from_sitemap():
    # 1) Thu thập URL job từ sitemap
    job_urls = collect_job_urls_from_sitemap(SITEMAP_MAX_JOBS)

    if not job_urls:
        print("[CRAWL] Không tìm thấy job nào từ sitemap.")
        return

    # Xáo trộn để tránh crawl theo thứ tự cố định
    random.shuffle(job_urls)

    total = len(job_urls)
    print(f"[CRAWL] TOTAL job URLs sẽ crawl: {total}")
    print(f"[CRAWL] JOB_MAX_RETRY = {JOB_MAX_RETRY}")

    for idx, url in enumerate(job_urls, start=1):
        print(f"\n[JOB {idx}/{total}] {url}")
        attempt = 0

        while attempt < JOB_MAX_RETRY:
            attempt += 1
            print(f"Attempt {attempt}/{JOB_MAX_RETRY}")
            try:
                # Truyền seq=idx để log được "Saved job {seq} - id {job_id}"
                crawl_and_save_one_job(url, seq=idx)
                # Nếu không raise exception thì coi như thành công, break khỏi vòng retry
                break
            except Exception as e:
                # In lỗi
                print(
                    f"  [ERROR] Crawl failed (attempt {attempt}): {e}\n"
                    # f"  [ERROR] Crawl failed (attempt {attempt}) for {url}: {e}\n"
                )

                # Nếu còn lượt retry thì sleep rồi thử lại
                if attempt < JOB_MAX_RETRY:
                    sleep_s = CRAWL_SLEEP_SECONDS
                    print(
                        f"  -> will retry {attempt} time after {sleep_s:.1f}s"
                    )
                    try:
                        time.sleep(sleep_s)
                    except KeyboardInterrupt:
                        print("  -> Interrupted during sleep, stopping batch.")
                        return
                else:
                    print("  -> reached max retry, mark as FAIL")

        # Nghỉ giữa các job để tránh 429 (trừ khi vừa bị 429 và đã sleep trong retry)
        try:
            time.sleep(CRAWL_SLEEP_SECONDS)
        except KeyboardInterrupt:
            print("Batch crawl interrupted by user.")
            return


def main():
    crawl_many_jobs_from_sitemap()


if __name__ == "__main__":
    main()

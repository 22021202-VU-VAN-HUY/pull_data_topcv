# app/topcv_parser.py
import json
from typing import Dict, Any, List, Optional

import requests
from bs4 import BeautifulSoup, Tag

from .config import settings
from .headless_fetch import fetch_html_headless

# ================== CẤU HÌNH REQUESTS ==================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ================== HỖ TRỢ FETCH HTML + FALLBACK HEADLESS ==================


def html_looks_blocked_or_empty(html: str) -> bool:
    """
    Heuristic đơn giản: HTML có vẻ bị chặn / không đúng trang job.
    Có thể chỉnh sửa thêm tuỳ bạn.
    """
    if not html or len(html) < 1000:
        return True
    lower = html.lower()
    bad_keywords = [
        "too many requests",
        "captcha",
        "access denied",
        "đã chặn truy cập",
    ]
    return any(k in lower for k in bad_keywords)


def fetch_html_with_fallback(url: str) -> str:
    """
    Tải HTML bằng requests; nếu bị chặn (429/403) hoặc HTML xấu
    và USE_HEADLESS_FALLBACK=true thì fallback sang Playwright.
    """
    use_headless = (
        str(getattr(settings, "USE_HEADLESS_FALLBACK", "false")).lower() == "true"
    )

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        status = resp.status_code

        # Nếu OK và HTML trông ổn: dùng luôn
        if status == 200 and not html_looks_blocked_or_empty(resp.text):
            return resp.text

        # Nếu bị chặn hoặc HTML xấu mà cho phép headless → fallback
        if use_headless and status in (403, 429):
            print(f"[HEADLESS] Fallback for {url} (status={status})")
            return fetch_html_headless(url, user_agent=HEADERS["User-Agent"])

        if use_headless and html_looks_blocked_or_empty(resp.text):
            print(f"[HEADLESS] Fallback for {url} (html looks blocked/empty)")
            return fetch_html_headless(url, user_agent=HEADERS["User-Agent"])

        # Trường hợp khác: raise như cũ
        resp.raise_for_status()
        return resp.text

    except requests.HTTPError as e:
        if use_headless:
            status = getattr(e.response, "status_code", None)
            print(f"[HEADLESS] HTTPError {status}, fallback for {url}: {e}")
            return fetch_html_headless(url, user_agent=HEADERS["User-Agent"])
        raise


def fetch_soup(url: str) -> BeautifulSoup:
    html = fetch_html_with_fallback(url)
    return BeautifulSoup(html, "html.parser")


# ================== PARSE JSON-LD ==================


def parse_jsonld(soup: BeautifulSoup) -> Dict[str, Any]:
    script = soup.find("script", {"type": "application/ld+json"})
    if not script:
        return {}
    raw = (script.string or script.get_text() or "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if isinstance(data, list):
        data = data[0]
    return data if isinstance(data, dict) else {}


def parse_job_from_jsonld(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bóc các field cơ bản từ JSON-LD JobPosting.
    """
    # Tên job
    title = data.get("title") or data.get("name")

    # Lương
    base_salary = data.get("baseSalary") or {}
    salary_value = base_salary.get("value") or {}
    salary = {
        "min": salary_value.get("minValue"),
        "max": salary_value.get("maxValue"),
        "currency": base_salary.get("currency"),
        "interval": salary_value.get("unitText"),
        "raw_text": None,  # sẽ bóc thêm từ UI
    }

    # Kinh nghiệm
    exp_req = data.get("experienceRequirements") or {}
    experience = {
        "months": exp_req.get("monthsOfExperience"),
        "raw_text": None,  # sẽ bóc thêm từ UI
    }

    # Deadline
    deadline = data.get("validThrough")

    # Thông tin chung
    general_info = {
        "cap_bac": data.get("occupationalCategory"),
        "hoc_van": None,
        "so_luong_tuyen": data.get("totalJobOpenings"),
        "hinh_thuc_lam_viec": data.get("employmentType"),
    }

    # Company
    org = data.get("hiringOrganization") or {}
    company = {
        "name": org.get("name"),
        "url": org.get("sameAs"),
        "logo": org.get("logo"),
        "size": None,
        "industry": data.get("industry"),
        "address": None,
    }

    # Địa điểm cơ bản từ JSON-LD
    job_location = (data.get("jobLocation") or {}).get("address") or {}
    locations: List[str] = []
    if job_location.get("addressRegion"):
        locations.append(job_location["addressRegion"])

    return {
        "title": title,
        "salary": salary,
        "experience": experience,
        "deadline": deadline,
        "general_info": general_info,
        "company": company,
        "locations": locations,
    }


# ================== PARSE CÁC SECTION CHI TIẾT ==================


def find_job_detail_container(soup: BeautifulSoup) -> Tag:
    heading = soup.find(
        string=lambda t: isinstance(t, str) and "Chi tiết tin tuyển dụng" in t
    )
    if not heading:
        return soup.body or soup

    container = heading.find_parent()
    for _ in range(5):
        if container and container.name != "body":
            container = container.parent
    return container or (soup.body or soup)


def get_section_by_title(container: BeautifulSoup, titles: List[str]) -> Dict[str, Optional[str]]:
    """
    Tìm 1 section theo tiêu đề (h2/h3).
    Trả về {"html": ..., "text": ...}.
    """
    heading = None
    for tag in container.find_all(["h2", "h3"]):
        txt = tag.get_text(strip=True)
        if any(t.lower() in txt.lower() for t in titles):
            heading = tag
            break

    if not heading:
        return {"html": None, "text": None}

    parts_html: List[str] = []
    parts_text: List[str] = []

    for sib in heading.find_next_siblings():
        if isinstance(sib, Tag) and sib.name in ["h2", "h3"]:
            break
        txt = sib.get_text(" ", strip=True)
        if not txt:
            continue
        parts_html.append(str(sib))
        parts_text.append(txt)

    if not parts_html:
        return {"html": None, "text": None}

    return {
        "html": "".join(parts_html),
        "text": "\n".join(parts_text),
    }


def parse_detail_sections(soup: BeautifulSoup) -> Dict[str, Dict[str, Optional[str]]]:
    container = find_job_detail_container(soup)

    sections = {
        "mo_ta_cong_viec": get_section_by_title(container, ["Mô tả công việc"]),
        "yeu_cau_ung_vien": get_section_by_title(container, ["Yêu cầu ứng viên"]),
        "thu_nhap": get_section_by_title(container, ["Thu nhập"]),
        "quyen_loi": get_section_by_title(
            container, ["Quyền lợi", "Quyền lợi được hưởng"]
        ),
        "phu_cap": get_section_by_title(container, ["Phụ cấp"]),
        "thiet_bi_lam_viec": get_section_by_title(
            container, ["Thiết bị làm việc", "Trang thiết bị làm việc"]
        ),
        "dia_diem_lam_viec": get_section_by_title(container, ["Địa điểm làm việc"]),
        "thoi_gian_lam_viec": get_section_by_title(container, ["Thời gian làm việc"]),
        "cach_thuc_ung_tuyen": get_section_by_title(
            container, ["Cách thức ứng tuyển"]
        ),
    }
    return sections


def parse_locations_from_section(section: Dict[str, Optional[str]]) -> List[str]:
    text = section.get("text") or ""
    if not text:
        return []
    parts = [p.strip(" -") for p in text.split("\n") if p.strip(" -")]
    if len(parts) == 1:
        extra = [x.strip() for x in parts[0].split(";") if x.strip()]
        if len(extra) > 1:
            parts = extra
    return parts


# ================== PARSE VỀ CÔNG TY (SIDEBAR) ==================


def parse_company_sidebar(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    result = {
        "size": None,
        "address": None,
        "industry": None,
    }

    sidebar = None
    for div in soup.find_all("div"):
        txt = div.get_text(" ", strip=True)
        if any(
            kw in txt
            for kw in ["Giới thiệu công ty", "Thông tin công ty", "Về công ty"]
        ):
            sidebar = div
            break

    if not sidebar:
        return result

    for row in sidebar.find_all(["div", "li"]):
        text = row.get_text(" ", strip=True)
        if not text:
            continue

        if "Quy mô" in text and "nhân viên" in text and result["size"] is None:
            # ví dụ: "Quy mô: 100-499 nhân viên"
            if ":" in text:
                result["size"] = text.split(":", 1)[-1].strip()
            else:
                result["size"] = text

        if ("Lĩnh vực" in text or "Ngành" in text) and result["industry"] is None:
            if ":" in text:
                result["industry"] = text.split(":", 1)[-1].strip()
            else:
                result["industry"] = text

        if ("Địa điểm" in text or "Địa chỉ" in text) and result["address"] is None:
            if ":" in text:
                result["address"] = text.split(":", 1)[-1].strip()
            else:
                result["address"] = text

    return result


# ================== PARSE BOX "THÔNG TIN CHUNG" ==================


def parse_general_info_box(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    result = {
        "cap_bac": None,
        "hoc_van": None,
        "experience_text": None,
        "hinh_thuc_lam_viec_text": None,
        "so_luong_tuyen_text": None,
        "salary_text": None,
    }

    heading = soup.find(
        string=lambda t: isinstance(t, str) and "Thông tin chung" in t
    )
    if not heading:
        return result

    container = heading.find_parent()
    for _ in range(3):
        if container and container.name != "body":
            container = container.parent

    if not container:
        return result

    for row in container.find_all(["div", "li"]):
        spans = row.find_all("span")
        if len(spans) < 2:
            continue
        label = spans[0].get_text(strip=True)
        value = spans[1].get_text(" ", strip=True)

        if "Cấp bậc" in label:
            result["cap_bac"] = value
        elif "Học vấn" in label or "Bằng cấp" in label:
            result["hoc_van"] = value
        elif "Kinh nghiệm" in label:
            result["experience_text"] = value
        elif "Hình thức làm việc" in label:
            result["hinh_thuc_lam_viec_text"] = value
        elif "Số lượng tuyển" in label:
            result["so_luong_tuyen_text"] = value
        elif "Mức lương" in label or "Thu nhập" in label:
            result["salary_text"] = value

    return result


# ================== GHÉP THÀNH 1 JOB ==================


def build_job_from_soup(url: str, soup: BeautifulSoup, jld: Dict[str, Any]) -> Dict[str, Any]:
    base = parse_job_from_jsonld(jld)

    detail_sections = parse_detail_sections(soup)

    # Địa điểm từ section "Địa điểm làm việc"
    loc_from_section = parse_locations_from_section(
        detail_sections.get("dia_diem_lam_viec", {})
    )
    locations = base["locations"][:]
    for loc in loc_from_section:
        if loc not in locations:
            locations.append(loc)

    # Company sidebar
    company_extra = parse_company_sidebar(soup)
    company = base["company"]
    for k, v in company_extra.items():
        if v:
            company[k] = v

    # Thông tin chung box
    general_extra = parse_general_info_box(soup)
    general_info = base["general_info"]

    if general_extra["cap_bac"]:
        general_info["cap_bac"] = general_extra["cap_bac"]
    if general_extra["hoc_van"]:
        general_info["hoc_van"] = general_extra["hoc_van"]

    experience = base["experience"]
    if general_extra["experience_text"]:
        experience["raw_text"] = general_extra["experience_text"]

    salary = base["salary"]
    if general_extra["salary_text"]:
        salary["raw_text"] = general_extra["salary_text"]

    general_info["hinh_thuc_lam_viec_raw"] = general_extra["hinh_thuc_lam_viec_text"]
    general_info["so_luong_tuyen_raw"] = general_extra["so_luong_tuyen_text"]

    job: Dict[str, Any] = {
        "url": url,
        "title": base["title"],
        "salary": salary,
        "locations": locations,
        "experience": experience,
        "detail_sections": detail_sections,
        "deadline": base["deadline"],
        "company": company,
        "general_info": general_info,
    }
    return job


def parse_job(url: str) -> Dict[str, Any]:
    """
    Hàm chính: nhận 1 job URL, trả về dict đầy đủ
    cho crawl_one_job / DB.
    """
    # Lần 1: requests (+fallback nếu bị 403/429/HTML xấu)
    html = fetch_html_with_fallback(url)
    soup = BeautifulSoup(html, "html.parser")
    jld = parse_jsonld(soup)
    job = build_job_from_soup(url, soup, jld)

    # Nếu bật headless và vẫn không có title → thử refetch bằng headless 1 lần
    use_headless = (
        str(getattr(settings, "USE_HEADLESS_FALLBACK", "false")).lower() == "true"
    )
    if use_headless and (not job.get("title") or not str(job["title"]).strip()):
        print(f"[HEADLESS] Missing title, refetch with headless: {url}")
        html2 = fetch_html_headless(url, user_agent=HEADERS["User-Agent"])
        soup2 = BeautifulSoup(html2, "html.parser")
        jld2 = parse_jsonld(soup2)
        job = build_job_from_soup(url, soup2, jld2)

    return job

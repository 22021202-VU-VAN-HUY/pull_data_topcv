# -*- coding: utf-8 -*-
"""
Parser chi tiết 1 job trên TopCV.

Trả về dict dạng:

{
  "url": ...,
  "title": ...,
  "salary": {
    "min": ...,
    "max": ...,
    "currency": ...,
    "interval": ...,
    "raw_text": ...
  },
  "locations": [...],
  "experience": {
    "months": ...,
    "raw_text": ...
  },
  "detail_sections": {
    "mo_ta_cong_viec": {"html": ..., "text": ...},
    "yeu_cau_ung_vien": {...},
    ...
  },
  "deadline": ...,
  "company": {
    "name": ...,
    "url": ...,
    "logo": ...,
    "size": ...,
    "industry": ...,
    "address": ...
  },
  "general_info": {
    "cap_bac": ...,
    "hoc_van": ...,
    "so_luong_tuyen": ...,
    "hinh_thuc_lam_viec": ...,
    "hinh_thuc_lam_viec_raw": ...,
    "so_luong_tuyen_raw": ...
  }
}
"""

import json
import re
from typing import Dict, Any, List, Optional

import requests
from bs4 import BeautifulSoup, Tag


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ----------------- HỖ TRỢ CƠ BẢN -----------------


def fetch_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


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


def get_section_by_title(container: BeautifulSoup, titles: List[str]) -> Dict[str, Optional[str]]:
    """
    Tìm section theo h2/h3 có chứa 1 trong các titles (case-insensitive).
    Trả về {"html": ..., "text": ...} (có thể None).
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


# ----------------- BÓC TỪ JSON-LD -----------------


def parse_job_from_jsonld(data: Dict[str, Any]) -> Dict[str, Any]:
    title = data.get("title") or data.get("name")

    base_salary = data.get("baseSalary") or {}
    salary_value = base_salary.get("value") or {}
    salary = {
        "min": salary_value.get("minValue"),
        "max": salary_value.get("maxValue"),
        "currency": base_salary.get("currency"),
        "interval": salary_value.get("unitText"),
        "raw_text": None,
    }

    exp_req = data.get("experienceRequirements") or {}
    experience = {
        "months": exp_req.get("monthsOfExperience"),
        "raw_text": None,
    }

    deadline = data.get("validThrough")

    general_info = {
        "cap_bac": data.get("occupationalCategory"),
        "hoc_van": None,
        "so_luong_tuyen": data.get("totalJobOpenings"),
        "hinh_thuc_lam_viec": data.get("employmentType"),
    }

    org = data.get("hiringOrganization") or {}
    company = {
        "name": org.get("name"),
        "url": org.get("sameAs"),
        "logo": org.get("logo"),
        "size": None,
        "industry": data.get("industry"),
        "address": None,
    }

    job_location = (data.get("jobLocation") or {}).get("address") or {}
    locations = []
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


# ----------------- BÓC CHI TIẾT TUYỂN DỤNG -----------------


def find_job_detail_container(soup: BeautifulSoup) -> Tag:
    heading = soup.find(string=lambda t: isinstance(t, str) and "Chi tiết tin tuyển dụng" in t)
    if not heading:
        return soup.body or soup

    container = heading.find_parent()
    for _ in range(5):
        if container and container.name != "body":
            container = container.parent
    return container or (soup.body or soup)


def parse_detail_sections(soup: BeautifulSoup) -> Dict[str, Dict[str, Optional[str]]]:
    container = find_job_detail_container(soup)

    sections = {
        "mo_ta_cong_viec": get_section_by_title(container, ["Mô tả công việc"]),
        "yeu_cau_ung_vien": get_section_by_title(container, ["Yêu cầu ứng viên"]),
        "thu_nhap": get_section_by_title(container, ["Thu nhập"]),
        "quyen_loi": get_section_by_title(container, ["Quyền lợi", "Quyền lợi được hưởng"]),
        "phu_cap": get_section_by_title(container, ["Phụ cấp"]),
        "thiet_bi_lam_viec": get_section_by_title(container, ["Thiết bị làm việc", "Trang thiết bị làm việc"]),
        "dia_diem_lam_viec": get_section_by_title(container, ["Địa điểm làm việc"]),
        "thoi_gian_lam_viec": get_section_by_title(container, ["Thời gian làm việc"]),
        "cach_thuc_ung_tuyen": get_section_by_title(container, ["Cách thức ứng tuyển"]),
    }

    return sections


def _cleanup_thu_nhap_section(section: Dict[str, Optional[str]], company_name: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Một số job có heading 'Thu nhập' nhưng nội dung lại là link company (không phải lương).
    Nếu text/html trùng tên công ty thì coi như không có dữ liệu thu nhập.
    """
    if not section or not company_name:
        return section

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip().casefold()

    text_norm = _norm(section.get("text") or "")
    company_norm = _norm(company_name)
    if text_norm and text_norm == company_norm:
        return {"html": None, "text": None}

    html = section.get("html") or ""
    if html:
        html_text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        if _norm(html_text) == company_norm:
            return {"html": None, "text": None}

    return section


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


# ----------------- COMPANY SIDEBAR -----------------


def parse_company_sidebar(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    """
    Tìm box 'Giới thiệu công ty' / 'Thông tin công ty' / 'Về công ty'
    rồi bóc ra: size, industry, address.
    """
    result = {
        "size": None,
        "address": None,
        "industry": None,
    }

    sidebar = None
    for div in soup.find_all("div"):
        txt = div.get_text(" ", strip=True)
        if not txt:
            continue
        if ("Giới thiệu công ty" in txt) or ("Thông tin công ty" in txt) or ("Về công ty" in txt):
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

        if (("Lĩnh vực" in text) or ("Ngành" in text)) and result["industry"] is None:
            if ":" in text:
                result["industry"] = text.split(":", 1)[-1].strip()
            else:
                result["industry"] = text

        if (("Địa điểm" in text) or ("Địa chỉ" in text)) and result["address"] is None:
            if ":" in text:
                result["address"] = text.split(":", 1)[-1].strip()
            else:
                result["address"] = text

    return result


# ----------------- THÔNG TIN CHUNG -----------------


def parse_general_info_box(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
    """
    Bóc box 'Thông tin chung':
    - Cấp bậc
    - Học vấn
    - Kinh nghiệm (raw text)
    - Hình thức làm việc (raw text)
    - Số lượng tuyển (raw text)
    - Mức lương/Thu nhập (raw text)
    """
    result = {
        "cap_bac": None,
        "hoc_van": None,
        "experience_text": None,
        "hinh_thuc_lam_viec_text": None,
        "so_luong_tuyen_text": None,
        "salary_text": None,
    }

    heading = soup.find(string=lambda t: isinstance(t, str) and "Thông tin chung" in t)
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
        elif ("Mức lương" in label) or ("Thu nhập" in label):
            result["salary_text"] = value

    return result


# ----------------- HÀM CHÍNH: PARSE 1 JOB -----------------


def _parse_job_from_soup(soup: BeautifulSoup, url: str) -> Dict[str, Any]:
    jld = parse_jsonld(soup)

    # 1) từ JSON-LD
    base = parse_job_from_jsonld(jld)

    # 2) chi tiết tuyển dụng
    detail_sections = parse_detail_sections(soup)

    # 3) địa điểm từ section
    loc_from_section = parse_locations_from_section(detail_sections.get("dia_diem_lam_viec", {}))
    locations = base["locations"][:]
    for loc in loc_from_section:
        if loc not in locations:
            locations.append(loc)

    # 4) company sidebar
    company_extra = parse_company_sidebar(soup)
    company = base["company"]
    for k, v in company_extra.items():
        if v:
            company[k] = v

    # Làm sạch trường "Thu nhập" nếu nhầm sang tên công ty
    detail_sections["thu_nhap"] = _cleanup_thu_nhap_section(
        detail_sections.get("thu_nhap", {}),
        company.get("name"),
    )

    # 5) thông tin chung box
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
    soup = fetch_soup(url)
    return _parse_job_from_soup(soup, url)


def parse_job_from_html(html: str, url: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    return _parse_job_from_soup(soup, url)

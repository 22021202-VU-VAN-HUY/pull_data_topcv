# app/api/jobs.py
from datetime import datetime, timezone

from flask import Blueprint, render_template, request, session, abort
from app.db import get_connection

jobs_bp = Blueprint("jobs", __name__)

# Map section_type -> tiêu đề như trên TopCV
SECTION_LABELS = {
    "mo_ta_cong_viec": "Mô tả công việc",
    "yeu_cau_ung_vien": "Yêu cầu ứng viên",
    "thu_nhap": "Thu nhập",
    "quyen_loi": "Quyền lợi",
    "dia_diem_lam_viec": "Địa điểm làm việc",
    "phuc_loi": "Phúc lợi",
    "thong_tin_khac": "Thông tin khác",
}

SECTION_ORDER = [
    "mo_ta_cong_viec",
    "yeu_cau_ung_vien",
    "thu_nhap",
    "quyen_loi",
    "dia_diem_lam_viec",
    "phuc_loi",
    "thong_tin_khac",
]


def _build_where_and_params(q: str):
    params = {}
    clauses = []
    if q:
        params["q"] = f"%{q}%"
        clauses.append("(j.title ILIKE %(q)s OR c.name ILIKE %(q)s)")
    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
    return where_sql, params


def _format_deadline(deadline):
    if not deadline:
        return None
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    days = (deadline.date() - now.date()).days
    date_str = deadline.strftime("%d/%m/%Y")

    if days >= 0:
        return f"Hạn nộp hồ sơ: {date_str} - Còn {days} ngày"
    else:
        return f"Hạn nộp hồ sơ: {date_str} - hết hạn"


@jobs_bp.route("/")
def index():
    """
    Trang chủ:
    - Không cần login.
    - Hiển thị danh sách job (DB thật).
    - Ưu tiên công việc còn hạn lên đầu.
    - Hiển thị: Tổng công việc: X - Y công việc còn hạn.
    """
    q = (request.args.get("q") or "").strip()
    page = int(request.args.get("page") or "1")
    page = max(page, 1)
    per_page = 9
    offset = (page - 1) * per_page

    user_id = session.get("user_id")
    where_sql, base_params = _build_where_and_params(q)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Tổng job (theo filter q)
            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM jobs j
                LEFT JOIN companies c ON j.company_id = c.id
                {where_sql}
                """,
                base_params,
            )
            row = cur.fetchone()
            total = row["cnt"] if row else 0

            # Tổng job còn hạn: deadline >= NOW()
            if where_sql:
                active_where = where_sql + " AND (j.deadline IS NOT NULL AND j.deadline >= NOW())"
            else:
                active_where = "WHERE j.deadline IS NOT NULL AND j.deadline >= NOW()"

            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM jobs j
                LEFT JOIN companies c ON j.company_id = c.id
                {active_where}
                """,
                base_params,
            )
            row2 = cur.fetchone()
            active_total = row2["cnt"] if row2 else 0

            # Danh sách job (ưu tiên job còn hạn)
            params = dict(base_params)
            params.update(
                {
                    "limit": per_page,
                    "offset": offset,
                    "user_id": user_id,
                }
            )

            cur.execute(
                f"""
                SELECT
                    j.id AS job_id,
                    j.title,
                    COALESCE(c.name, '') AS company,
                    COALESCE(
                        (
                            SELECT jl.location_text
                            FROM job_locations jl
                            WHERE jl.job_id = j.id
                            ORDER BY jl.is_primary DESC, jl.sort_order, jl.id
                            LIMIT 1
                        ),
                        ''
                    ) AS location_text,
                    COALESCE(j.salary_raw_text, 'Thoả thuận') AS salary_text,
                    j.deadline,
                    CASE WHEN ub.user_id IS NULL THEN FALSE ELSE TRUE END AS starred
                FROM jobs j
                LEFT JOIN companies c ON j.company_id = c.id
                LEFT JOIN user_job_bookmarks ub
                    ON ub.job_id = j.id AND ub.user_id = %(user_id)s
                {where_sql}
                ORDER BY
                    CASE
                        WHEN j.deadline IS NOT NULL AND j.deadline >= NOW() THEN 0  -- còn hạn
                        WHEN j.deadline IS NULL THEN 1                               -- không có hạn
                        ELSE 2                                                       -- đã hết hạn
                    END,
                    j.crawled_at DESC NULLS LAST,
                    j.id DESC
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                params,
            )
            rows = cur.fetchall()

    total_pages = max((total + per_page - 1) // per_page, 1) if total else 1

    jobs = []
    for r in rows:
        loc = r["location_text"] or ""
        jobs.append(
            {
                "job_id": r["job_id"],
                "title": r["title"],
                "company": r["company"] or "",
                "city": loc,
                "district": None,
                "salary_text": r["salary_text"],
                "starred": bool(r["starred"]),
                "deadline_text": _format_deadline(r["deadline"]),
            }
        )

    return render_template(
        "index.html",
        title="Trang chủ",
        q=q,
        jobs=jobs,
        page=page,
        total=total,
        active_total=active_total,
        total_pages=total_pages,
    )

def _fetch_job_sections(cur, job_id: int):
    """
    Lấy các section từ job_sections và format thành list
    [ {key,label,html,text}, ... ]
    Chỉ trả các section có nội dung.
    """
    cur.execute(
        """
        SELECT section_type, text_content, html_content, id
        FROM job_sections
        WHERE job_id = %(job_id)s
        ORDER BY id
        """,
        {"job_id": job_id},
    )
    rows = cur.fetchall()

    by_type = {}
    for row in rows:
        stype = row["section_type"]
        html = row.get("html_content")
        text = row.get("text_content")
        if not html and not text:
            continue
        by_type[stype] = {
            "key": stype,
            "label": SECTION_LABELS.get(
                stype, stype.replace("_", " ").title()
            ),
            "html": html,
            "text": text,
        }

    # Sắp xếp theo SECTION_ORDER, sau đó các type lạ
    result = []
    for st in SECTION_ORDER:
        s = by_type.get(st)
        if s:
            result.append(s)

    for st, s in by_type.items():
        if st not in SECTION_ORDER:
            result.append(s)

    return result


@jobs_bp.route("/jobs/<int:job_id>")
def job_detail(job_id: int):
    user_id = session.get("user_id")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    j.id AS job_id,
                    j.title,
                    j.url,
                    j.deadline,
                    j.salary_min,
                    j.salary_max,
                    j.salary_currency,
                    j.salary_interval,
                    COALESCE(j.salary_raw_text, 'Thoả thuận') AS salary_text,
                    j.experience_months,
                    j.experience_raw_text,
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
                    c.address                    AS company_address,

                    COALESCE(
                        (
                            SELECT jl.location_text
                            FROM job_locations jl
                            WHERE jl.job_id = j.id
                            ORDER BY jl.is_primary DESC, jl.sort_order, jl.id
                            LIMIT 1
                        ),
                        ''
                    ) AS location_text,

                    CASE WHEN ub.user_id IS NULL THEN FALSE ELSE TRUE END AS starred
                FROM jobs j
                LEFT JOIN companies c ON j.company_id = c.id
                LEFT JOIN user_job_bookmarks ub
                    ON ub.job_id = j.id AND ub.user_id = %(user_id)s
                WHERE j.id = %(job_id)s
                """,
                {"job_id": job_id, "user_id": user_id},
            )
            job_row = cur.fetchone()
            if not job_row:
                abort(404)

            # Lấy sections
            detail_sections = _fetch_job_sections(cur, job_id)

    # thông tin công ty 
    company_info = {
        "Tên công ty": job_row.get("company_name"),
        "Website": job_row.get("company_url"),
        "Quy mô": job_row.get("company_size"),
        "Lĩnh vực": job_row.get("company_industry"),
        "Địa chỉ": job_row.get("company_address"),
    }

    # thông tin chung 
    general_info = {
        "ID công việc": job_row["job_id"],
        "Cấp bậc": job_row.get("cap_bac"),
        "Học vấn": job_row.get("hoc_van"),
        "Số lượng tuyển": job_row.get("so_luong_tuyen")
        or job_row.get("so_luong_tuyen_raw"),
        "Hình thức làm việc": job_row.get("hinh_thuc_lam_viec")
        or job_row.get("hinh_thuc_lam_viec_raw"),
        "Kinh nghiệm": job_row.get("experience_raw_text"),
        "Địa điểm làm việc": job_row.get("location_text"),
    }

    job = {
        "job_id": job_row["job_id"],
        "title": job_row["title"],
        "company": job_row["company_name"],
        "url": job_row["url"],
        "city": job_row["location_text"] or "",
        "district": None,
        "source": "topcv",
        "expires_at": job_row["deadline"],
    }

    is_starred = bool(job_row["starred"])
    salary_text = job_row["salary_text"]
    deadline_text = _format_deadline(job_row["deadline"])

    # fallback
    fallback_description = None
    if not detail_sections:
        parts = []
        if job_row.get("experience_raw_text"):
            parts.append(f"Yêu cầu kinh nghiệm: {job_row['experience_raw_text']}")
        if salary_text:
            parts.append(f"Lương: {salary_text}")
        if parts:
            fallback_description = "\n\n".join(parts)

    return render_template(
        "job_detail.html",
        title=job["title"],
        job=job,
        salary_text=salary_text,
        deadline_text=deadline_text,
        detail_sections=detail_sections,
        fallback_description=fallback_description,
        company_info=company_info,
        general_info=general_info,
        is_starred=is_starred,
    )

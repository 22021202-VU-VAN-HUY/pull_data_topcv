# app/api/auth.py
from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
)
from werkzeug.security import generate_password_hash, check_password_hash

from app.db import get_connection

auth_bp = Blueprint("auth", __name__)


# ===================== Helpers =====================

def get_current_user():
    """Lấy user hiện tại từ session, trả về dict hoặc None."""
    user_id = session.get("user_id")
    if not user_id:
        return None

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    full_name,
                    email,
                    phone,
                    password_hash,
                    is_active
                FROM users
                WHERE id = %(user_id)s
                """,
                {"user_id": user_id},
            )
            row = cur.fetchone()

    if not row or not row["is_active"]:
        return None
    return row


# ===================== HTML PAGES =====================

@auth_bp.route("/login")
def login_page():
    if get_current_user():
        return redirect(url_for("jobs.index"))
    return render_template("login.html", title="Đăng nhập")


@auth_bp.route("/register")
def register_page():
    if get_current_user():
        return redirect(url_for("jobs.index"))
    return render_template("register.html", title="Đăng ký")


@auth_bp.route("/profile")
def profile_page():
    """Chuyển hướng sang trang mặc định (công việc đã lưu)."""
    user = get_current_user()
    if not user:
        return redirect(url_for("auth.login_page"))

    return redirect(url_for("auth.profile_section", section="bookmarks"))


@auth_bp.route("/profile/<section>")
def profile_section(section: str):
    user = get_current_user()
    if not user:
        return redirect(url_for("auth.login_page"))

    try:
        bookmark_page = int(request.args.get("page") or "1")
    except ValueError:
        bookmark_page = 1
    bookmark_page = max(bookmark_page, 1)
    bookmark_per_page = 9
    bookmark_total = 0
    bookmark_total_pages = 1
    sections = {
        "bookmarks": "Công việc đã lưu",
        "info": "Thông tin người dùng",
        "password": "Đổi mật khẩu",
    }

    if section not in sections:
        return redirect(url_for("auth.profile_section", section="bookmarks"))

    saved_jobs = []
    if section == "bookmarks":
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM user_job_bookmarks
                    WHERE user_id = %(user_id)s
                    """,
                    {"user_id": user["id"]},
                )
                row_cnt = cur.fetchone()
                bookmark_total = row_cnt["cnt"] if row_cnt else 0
                if bookmark_total:
                    bookmark_total_pages = max(
                        (bookmark_total + bookmark_per_page - 1) // bookmark_per_page, 1
                    )
                    bookmark_page = min(bookmark_page, bookmark_total_pages)
                else:
                    bookmark_page = 1
                offset = (bookmark_page - 1) * bookmark_per_page

                cur.execute(
                    """
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
                        COALESCE(j.salary_raw_text, 'Thoả thuận') AS salary_text
                    FROM user_job_bookmarks b
                    JOIN jobs j ON j.id = b.job_id
                    LEFT JOIN companies c ON j.company_id = c.id
                    WHERE b.user_id = %(user_id)s
                    ORDER BY j.crawled_at DESC NULLS LAST, j.id DESC
                    LIMIT %(limit)s OFFSET %(offset)s
                    """,
                    {
                        "user_id": user["id"],
                        "limit": bookmark_per_page,
                        "offset": offset,
                    },
                )
                rows = cur.fetchall()

        for r in rows:
            loc = r["location_text"] or ""
            saved_jobs.append(
                {
                    "job_id": r["job_id"],
                    "title": r["title"],
                    "company": r["company"],
                    "city": loc,
                    "district": None,
                    "salary_text": r["salary_text"],
                }
            )

    return render_template(
        "profile.html",
        title=f"{sections[section]} - Quản lý tài khoản",
        user=user,
        saved_jobs=saved_jobs,
        current_section=section,
        section_title=sections[section],
        bookmark_page=bookmark_page,
        bookmark_total_pages=bookmark_total_pages,
        bookmark_total=bookmark_total,
    )


# ===================== JSON API =====================

@auth_bp.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    phone = (data.get("phone") or "").strip()
    password = (data.get("password") or "").strip()

    if not full_name or not email or not password:
        return jsonify({"detail": "Thiếu thông tin bắt buộc."}), 400

    password_hash = generate_password_hash(password)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM users WHERE email = %(email)s",
                {"email": email},
            )
            if cur.fetchone():
                return jsonify({"detail": "Email đã được sử dụng."}), 400

            cur.execute(
                """
                INSERT INTO users (full_name, email, phone, password_hash, is_active)
                VALUES (%(full_name)s, %(email)s, %(phone)s, %(password_hash)s, TRUE)
                RETURNING id
                """,
                {
                    "full_name": full_name,
                    "email": email,
                    "phone": phone,
                    "password_hash": password_hash,
                },
            )
            row = cur.fetchone()
            user_id = row["id"]
        conn.commit()

    session["user_id"] = user_id
    return jsonify({"ok": True})


@auth_bp.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not email or not password:
        return jsonify({"detail": "Thiếu email hoặc mật khẩu."}), 400

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, full_name, email, phone, password_hash, is_active
                FROM users
                WHERE email = %(email)s
                """,
                {"email": email},
            )
            user = cur.fetchone()

    if not user or not user["is_active"]:
        return jsonify({"detail": "Tài khoản không tồn tại hoặc đã bị khóa."}), 401

    if not check_password_hash(user["password_hash"], password):
        return jsonify({"detail": "Email hoặc mật khẩu không đúng."}), 401

    session["user_id"] = user["id"]
    return jsonify({"ok": True})


@auth_bp.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@auth_bp.route("/api/me", methods=["GET"])
def api_me():
    user = get_current_user()
    if not user:
        return jsonify({"user": None})
    return jsonify(
        {
            "user": {
                "id": user["id"],
                "full_name": user["full_name"],
                "email": user["email"],
                "phone": user["phone"],
            }
        }
    )


@auth_bp.route("/api/me/update", methods=["POST"])
def api_me_update():
    user = get_current_user()
    if not user:
        return jsonify({"detail": "Chưa đăng nhập."}), 401

    data = request.get_json() or {}
    full_name = (data.get("full_name") or "").strip()
    phone = (data.get("phone") or "").strip()

    if not full_name:
        return jsonify({"detail": "Họ tên không được để trống."}), 400

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET full_name = %(full_name)s,
                    phone = %(phone)s,
                    updated_at = NOW()
                WHERE id = %(user_id)s
                """,
                {"full_name": full_name, "phone": phone, "user_id": user["id"]},
            )
        conn.commit()

    return jsonify({"ok": True})


@auth_bp.route("/api/me/change_password", methods=["POST"])
def api_change_password():
    user = get_current_user()
    if not user:
        return jsonify({"detail": "Chưa đăng nhập."}), 401

    data = request.get_json() or {}
    old_password = (data.get("old_password") or "").strip()
    new_password = (data.get("new_password") or "").strip()

    if not old_password or not new_password:
        return jsonify({"detail": "Thiếu mật khẩu cũ hoặc mới."}), 400

    if not check_password_hash(user["password_hash"], old_password):
        return jsonify({"detail": "Mật khẩu hiện tại không đúng."}), 400

    new_hash = generate_password_hash(new_password)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET password_hash = %(password_hash)s,
                    updated_at = NOW()
                WHERE id = %(user_id)s
                """,
                {"password_hash": new_hash, "user_id": user["id"]},
            )
        conn.commit()

    return jsonify({"ok": True})


@auth_bp.route("/api/star", methods=["POST"])
def api_toggle_star():
    """
    Toggle bookmark 1 job:
    - Nếu chưa bookmark -> INSERT
    - Nếu đã bookmark -> DELETE
    """
    user = get_current_user()
    if not user:
        return jsonify({"detail": "Chưa đăng nhập."}), 401

    job_id = request.args.get("job_id")
    try:
        job_id = int(job_id)
    except (TypeError, ValueError):
        return jsonify({"detail": "job_id không hợp lệ."}), 400

    starred = False

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM user_job_bookmarks
                WHERE user_id = %(user_id)s AND job_id = %(job_id)s
                """,
                {"user_id": user["id"], "job_id": job_id},
            )
            exists = cur.fetchone() is not None

            if exists:
                cur.execute(
                    """
                    DELETE FROM user_job_bookmarks
                    WHERE user_id = %(user_id)s AND job_id = %(job_id)s
                    """,
                    {"user_id": user["id"], "job_id": job_id},
                )
                starred = False
            else:
                cur.execute(
                    """
                    INSERT INTO user_job_bookmarks (user_id, job_id)
                    VALUES (%(user_id)s, %(job_id)s)
                    ON CONFLICT (user_id, job_id) DO NOTHING
                    """,
                    {"user_id": user["id"], "job_id": job_id},
                )
                starred = True
        conn.commit()

    return jsonify({"job_id": job_id, "starred": starred})

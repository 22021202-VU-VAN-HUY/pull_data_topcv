# app/api/rag/chat_logic.py

from __future__ import annotations

import logging
import re
import html
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from app.config import settings
from app.api.rag.retriever import fetch_full_job_detail, retrieve_jobs
from app.api.rag.query_parser import parse_user_query

logger = logging.getLogger(__name__)

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

_gemini_model: Optional[genai.GenerativeModel] = None


def get_gemini_model() -> genai.GenerativeModel:
    """
    Khởi tạo & cache Gemini model.
    """
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model

    api_key = getattr(settings, "GEMINI_API_KEY", "") or ""
    model_name = getattr(settings, "GEMINI_MODEL", "gemini-2.0-flash")

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY chưa được cấu hình trong .env / Settings.")

    genai.configure(api_key=api_key)
    _gemini_model = genai.GenerativeModel(model_name)
    logger.info("Gemini model initialized: %s", model_name)
    return _gemini_model


# ========= FORMAT LƯƠNG / CONTEXT =========


def _format_salary_block(meta: Dict[str, Any]) -> str:
    salary = meta.get("salary") or {}
    raw_text = salary.get("raw_text")
    if raw_text:
        return raw_text

    salary_min = salary.get("min")
    salary_max = salary.get("max")
    currency = salary.get("currency") or "VND"
    interval = salary.get("interval") or "MONTH"

    interval_vi = {
        "MONTH": "/tháng",
        "YEAR": "/năm",
        "HOUR": "/giờ",
    }.get(interval, "")

    if salary_min is None and salary_max is None:
        return "Thoả thuận"

    if salary_min is not None and salary_max is not None:
        return f"Từ {salary_min:,.0f} đến {salary_max:,.0f} {currency} {interval_vi}"

    if salary_min is not None:
        return f"Từ {salary_min:,.0f} {currency} {interval_vi}"

    return f"Đến {salary_max:,.0f} {currency} {interval_vi}"


def _get_company_name(meta: Dict[str, Any]) -> str:
    company = meta.get("company")
    if isinstance(company, dict):
        return company.get("name") or ""
    if isinstance(company, str):
        return company
    return ""


def _get_locations_text(meta: Dict[str, Any]) -> str:
    locs = meta.get("locations") or []
    if isinstance(locs, list):
        return ", ".join([str(x) for x in locs if x])
    return str(locs) if locs else ""


def _get_detail_text(
    detail_sections: Dict[str, Any],
    key: str,
    *,
    max_len: int = 400,
) -> str:
    sec = detail_sections.get(key) or {}
    if isinstance(sec, dict):
        text = sec.get("text") or ""
    elif isinstance(sec, str):
        text = sec
    else:
        text = ""

    text = (text or "").strip()
    if not text:
        return ""

    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _get_section_full_text(detail_sections: Dict[str, Any], key: str) -> str:
    sec = detail_sections.get(key) or {}
    if isinstance(sec, dict):
        return (sec.get("text") or "").strip()
    if isinstance(sec, str):
        return sec.strip()
    return ""


def _format_one_job_context(
    idx: int,
    doc: Dict[str, Any],
    *,
    is_current: bool = False,
) -> str:
    """
    Format 1 job trong context RAG, đã rút gọn để tiết kiệm token.
    """
    meta = doc.get("metadata") or {}
    job_id = meta.get("id") or doc.get("job_id")

    # URL nội bộ trong hệ thống Flask (ưu tiên dùng cho chatbot)
    app_url = f"/jobs/{job_id}" if job_id is not None else ""

    # URL gốc TopCV (vẫn giữ nếu bạn cần dùng sau này)
    source_url = meta.get("url") or ""

    title = meta.get("title") or ""
    company = _get_company_name(meta)
    locations = _get_locations_text(meta)
    salary_text = _format_salary_block(meta)

    general_info = meta.get("general_info") or {}
    cap_bac = general_info.get("cap_bac")
    hinh_thuc = general_info.get("hinh_thuc_lam_viec")

    detail_sections = meta.get("detail_sections") or {}
    mo_ta = _get_detail_text(detail_sections, "mo_ta_cong_viec", max_len=350)
    yeu_cau = _get_detail_text(detail_sections, "yeu_cau_ung_vien", max_len=350)
    quyen_loi = _get_detail_text(detail_sections, "quyen_loi", max_len=350)

    chunk_text = (doc.get("chunk_text") or "").strip()
    if chunk_text:
        max_chunk_len = 300
        if len(chunk_text) > max_chunk_len:
            chunk_text = chunk_text[: max_chunk_len - 3] + "..."

    score = doc.get("score")

    lines: List[str] = []
    prefix = f"[JOB {idx}]"
    if is_current:
        prefix += " (Job bạn đang xem)"

    lines.append(f"{prefix} ID nội bộ: {job_id}")
    if title:
        lines.append(f"Tiêu đề: {title}")
    if company:
        lines.append(f"Công ty: {company}")
    if locations:
        lines.append(f"Địa điểm: {locations}")
    if cap_bac:
        lines.append(f"Cấp bậc: {cap_bac}")
    if hinh_thuc:
        lines.append(f"Hình thức: {hinh_thuc}")
    lines.append(f"Mức lương: {salary_text}")

    # 👉 Link ưu tiên cho chatbot: URL nội bộ JobFinder
    if app_url:
        lines.append(
            f"Link chi tiết trên JobFinder (nên dùng cho người dùng): {app_url}"
        )

    # Link TopCV chỉ dùng làm tham khảo cho model
    if source_url:
        lines.append(f"Link TopCV gốc (tham khảo): {source_url}")

    if score is not None:
        lines.append(f"(Độ liên quan nội bộ: {score:.3f})")

    if mo_ta:
        lines.append("")
        lines.append("Mô tả công việc (tóm tắt):")
        lines.append(mo_ta)
    if yeu_cau:
        lines.append("")
        lines.append("Yêu cầu ứng viên (tóm tắt):")
        lines.append(yeu_cau)
    if quyen_loi:
        lines.append("")
        lines.append("Quyền lợi chính:")
        lines.append(quyen_loi)

    if chunk_text:
        lines.append("")
        lines.append("Đoạn thông tin nổi bật từ chỉ mục (rút gọn):")
        lines.append(chunk_text)

    return "\n".join(lines)


def _format_full_current_job(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    job_id = meta.get("id") or doc.get("job_id")
    title = meta.get("title") or ""
    company = _get_company_name(meta)
    locations = _get_locations_text(meta)
    salary_text = _format_salary_block(meta)

    general_info = meta.get("general_info") or {}
    cap_bac = general_info.get("cap_bac")
    hinh_thuc = general_info.get("hinh_thuc_lam_viec")
    hoc_van = general_info.get("hoc_van")
    so_luong = general_info.get("so_luong_tuyen")
    experience = general_info.get("experience")
    deadline = general_info.get("deadline")

    detail_sections = meta.get("detail_sections") or {}

    lines: List[str] = []
    lines.append("[JOB ĐANG XEM - NGỮ CẢNH CHI TIẾT]")
    lines.append(f"ID nội bộ: {job_id}")
    if title:
        lines.append(f"Tiêu đề: {title}")
    if company:
        lines.append(f"Công ty: {company}")
    if locations:
        lines.append(f"Địa điểm: {locations}")
    lines.append(f"Mức lương: {salary_text}")
    if cap_bac:
        lines.append(f"Cấp bậc: {cap_bac}")
    if hinh_thuc:
        lines.append(f"Hình thức làm việc: {hinh_thuc}")
    if hoc_van:
        lines.append(f"Yêu cầu học vấn: {hoc_van}")
    if so_luong:
        lines.append(f"Số lượng tuyển: {so_luong}")
    if experience:
        lines.append(f"Kinh nghiệm: {experience}")
    if deadline:
        lines.append(f"Hạn nộp hồ sơ: {deadline}")

    for stype in SECTION_ORDER:
        label = SECTION_LABELS.get(stype, stype)
        text = _get_section_full_text(detail_sections, stype)
        if not text:
            continue
        lines.append("")
        lines.append(f"{label}:")
        lines.append(text)

    for stype, sec in detail_sections.items():
        if stype in SECTION_ORDER:
            continue
        text = _get_section_full_text(detail_sections, stype)
        if text:
            lines.append("")
            lines.append(f"{stype}:")
            lines.append(text)

    return "\n".join(lines)


# ========= PHÂN LOẠI Ý ĐỊNH CƠ BẢN =========


def _is_greeting_only(message: str) -> bool:
    text = (message or "").strip().lower()
    if not text:
        return False

    greeting_keywords = [
        "xin chào",
        "chào bạn",
        "chào anh",
        "chào chị",
        "hello",
        "hi",
        "alo",
        "chào",
        "hey",
    ]

    job_intent_keywords = [
        "công việc",
        "job",
        "tuyển",
        "ứng tuyển",
        "việc làm",
        "lương",
        "tìm",
    ]

    if any(k in text for k in job_intent_keywords):
        return False

    # Câu chào thường ngắn, không kèm yêu cầu rõ.
    return any(k in text for k in greeting_keywords)


def _build_context_block(
    docs: List[Dict[str, Any]],
    *,
    current_job_id: Optional[int] = None,
    current_job_doc: Optional[Dict[str, Any]] = None,
) -> str:
    parts: List[str] = []

    # Ưu tiên nhúng toàn bộ nội dung job hiện tại để model trả lời chi tiết.
    full_doc = current_job_doc
    if full_doc is None and current_job_id is not None:
        try:
            full_doc = fetch_full_job_detail(int(current_job_id))
        except Exception as e:
            logger.warning("Không lấy được full job %s: %s", current_job_id, e)
            full_doc = None

    if not docs and not full_doc:
        return (
            "Không tìm được công việc phù hợp trong dữ liệu (không có document nào từ RAG)."
        )

    if full_doc:
        parts.append(_format_full_current_job(full_doc))
        parts.append("\n---\n")

    for i, d in enumerate(docs, start=1):
        is_current = False
        meta = d.get("metadata") or {}
        job_id = meta.get("id") or d.get("job_id")
        if current_job_id is not None:
            try:
                is_current = int(job_id) == int(current_job_id)
            except Exception:
                is_current = job_id == current_job_id

        parts.append(_format_one_job_context(i, d, is_current=is_current))
        parts.append("\n---\n")
    return "\n".join(parts)


def _build_history_block(history: List[Dict[str, str]]) -> str:
    if not history:
        return "Chưa có lịch sử hội thoại trước đó."

    lines: List[str] = ["Lịch sử hội thoại trước đó (tin nhắn mới nhất ở cuối):"]
    for turn in history:
        role = turn.get("role") or "user"
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        role_vi = "Người dùng" if role == "user" else "Trợ lý"
        lines.append(f"{role_vi}: {content}")
    return "\n".join(lines)


def _build_filters_block(query_filters: Dict[str, Any]) -> str:
    if not query_filters:
        return "Chưa suy ra được bộ lọc từ câu hỏi người dùng."

    lines: List[str] = ["Các bộ lọc suy ra từ câu hỏi:"]

    job_keywords = query_filters.get("job_keywords") or []
    if job_keywords:
        lines.append(f"- Từ khoá ngành/vị trí: {', '.join(job_keywords)}")

    locations = query_filters.get("locations") or []
    if locations:
        lines.append(f"- Địa điểm ưu tiên: {', '.join(locations)}")

    min_salary = query_filters.get("min_salary_vnd")
    max_salary = query_filters.get("max_salary_vnd")
    if min_salary or max_salary:
        salary_parts = []
        if min_salary:
            salary_parts.append(f"từ {int(min_salary):,} VND")
        if max_salary:
            salary_parts.append(f"đến {int(max_salary):,} VND")
        lines.append(f"- Mức lương mong muốn: {' '.join(salary_parts)}")

    skills = query_filters.get("skills") or []
    if skills:
        lines.append(f"- Kỹ năng/yêu cầu: {', '.join(skills)}")

    if len(lines) == 1:
        lines.append("- Chưa có ràng buộc cụ thể.")

    return "\n".join(lines)


def _build_retrieval_query(user_message: str, history: List[Dict[str, str]]) -> str:
    """
    Ghép thêm vài lượt hội thoại gần nhất để model retrieve không bị lạc ngữ cảnh
    (ví dụ: "công việc thứ 2", "mô tả công việc này").
    """
    base = (user_message or "").strip()
    if not history:
        return base

    # Lấy tối đa 4 lượt cuối, nối thành 1 đoạn ngắn gọn để embedding.
    tail_turns: List[str] = []
    for turn in history[-4:]:
        content = (turn.get("content") or "").strip()
        if content:
            tail_turns.append(content)

    if not tail_turns:
        return base

    history_text = " | ".join(tail_turns)

    # Giới hạn độ dài để tránh làm loãng embedding.
    max_len = 800
    if len(history_text) > max_len:
        history_text = history_text[-max_len:]

    if base:
        return f"{base} | Ngữ cảnh trước đó: {history_text}"
    return history_text


def _build_prompt(
    user_message: str,
    docs: List[Dict[str, Any]],
    history: List[Dict[str, str]],
    *,
    query_filters: Dict[str, Any],
    current_job_id: Optional[int],
    current_job_doc: Optional[Dict[str, Any]],
) -> str:
    system_prompt = (
        "Bạn là trợ lý tuyển dụng JobFinder (dữ liệu từ TopCV).\n"
        "- Trả lời bằng TIẾNG VIỆT, thân thiện, gần gũi như người thật đang trò chuyện.\n"
        "- CHỈ dùng thông tin trong phần 'NGỮ CẢNH CÔNG VIỆC (RAG)'; không bịa thêm job/công ty/lương/link ngoài ngữ cảnh.\n"
        "- ƯU TIÊN dùng URL nội bộ JobFinder (bắt đầu bằng /jobs/...) khi đưa link cho người dùng, "
        "không khuyến khích dùng link TopCV.\n"
        "- Danh sách job trong RAG đã được hệ thống lọc sẵn. Nếu ngữ cảnh RAG KHÔNG TRỐNG, luôn ưu tiên dùng các job này để gợi ý; không được trả lời rằng 'không có công việc phù hợp'.\n"
        "- Chỉ khi phần ngữ cảnh ghi rõ 'Không tìm được công việc phù hợp trong dữ liệu' thì mới được nói là không có job phù hợp.\n"
        "- Khi có block '[JOB ĐANG XEM - NGỮ CẢNH CHI TIẾT]', phải đọc toàn bộ mô tả/yêu cầu/quyền lợi của block đó để trả lời các câu hỏi về job hiện tại, kể cả các câu hỏi lắt léo về kỹ năng/phúc lợi.\n"
        "- Khi nói về lương, dùng min/max/currency/interval nếu có; nếu không có thì ghi 'Thoả thuận'.\n"
        "- Nếu câu hỏi nhắc tới 'công việc này', 'job hiện tại'... hãy ưu tiên job được đánh dấu (Job bạn đang xem) trong NGỮ CẢNH và trả lời trực tiếp theo dữ liệu của job đó.\n"
        "- Nếu câu hỏi mang tính tìm kiếm (ví dụ 'công việc nào cần cả A và B, lương 20tr'), hãy chọn các job trong ngữ cảnh phù hợp nhất thay vì chỉ dùng job đang xem.\n"
        "- Với câu hỏi dò chi tiết (phúc lợi, trợ cấp, kỹ năng...), hãy trích đúng đoạn liên quan trong mô tả/yêu cầu/quyền lợi nếu có; nếu không thấy thông tin, nói rõ là chưa thấy ghi trong mô tả.\n"
        "- Nếu người dùng hỏi về kỹ năng, hãy trích từ mô tả / yêu cầu ứng viên của các job trong ngữ cảnh.\n"
        "- Khi người dùng hỏi thêm thông tin về từng job, hãy đóng vai trò tư vấn: giải thích ngắn gọn, giúp họ hiểu yêu cầu, quyền lợi và cách ứng tuyển dựa trên dữ liệu có sẵn.\n"
        "- Câu trả lời ngắn gọn, rõ ràng, dùng bullet (-) và xuống dòng giữa các ý.\n"
    )

    context_block = _build_context_block(
        docs, current_job_id=current_job_id, current_job_doc=current_job_doc
    )
    history_block = _build_history_block(history)
    filters_block = _build_filters_block(query_filters)

    prompt = f"""{system_prompt}

================= NGỮ CẢNH CÔNG VIỆC (RAG) =================
{context_block}

================= PHÂN TÍCH YÊU CẦU NGƯỜI DÙNG =================
{filters_block}

================= LỊCH SỬ HỘI THOẠI =================
{history_block}

================= CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG =================
{user_message}

================= YÊU CẦU TRẢ LỜI =================
- Trả lời ngắn gọn, ưu tiên 2–4 bullet; mỗi bullet ≤ 2 câu.
- Mẫu bullet: "- <tiêu đề> – <công ty>; lương: <text>; địa điểm: <text>. [link](/jobs/<id>)"
- Ưu tiên dùng URL dạng /jobs/<id> khi gắn link cho người dùng.
- Giữ mỗi bullet trên một dòng, có khoảng trắng giữa các bullet để dễ đọc.
- Nếu có link, hãy đặt trong dấu [](...) để người dùng bấm được (hoặc chèn URL /jobs/<id> trực tiếp vào cuối bullet).
- Có thể mở đầu 1 câu chào hoặc đồng cảm ngắn để tăng tự nhiên, sau đó dùng bullet để tổng hợp. Giữ giọng điệu mạch lạc, tôn trọng người hỏi.
- Nếu phần RAG ghi 'Không tìm được công việc phù hợp trong dữ liệu', hãy nói rõ là không tìm thấy job phù hợp và gợi ý người dùng tìm lại.
- Không tự tạo thêm job hoặc link ngoài danh sách trong NGỮ CẢNH.
"""
    return prompt


# ========= CLEAN + HTML HOÁ CÂU TRẢ LỜI =========


def _markdown_links_to_html(text: str) -> str:
    """
    - [link](/jobs/123) -> <a href="/jobs/123">link</a>
    - /jobs/123 -> <a href="/jobs/123">Xem chi tiết</a>
    (Không động tới link TopCV để tránh user bị dẫn ra ngoài nếu không cần.)
    """
    if not text:
        return ""

    # Chỉ convert markdown có URL nội bộ /jobs/xxx
    md_pattern = re.compile(r"\[([^\]]+)\]\((/jobs/\d+)\)")
    text = md_pattern.sub(r'<a href="\2" class="chat-link">\1</a>', text)

    # Convert đường dẫn /jobs/123 trần thành link
    url_pattern = re.compile(r"(/jobs/\d+)")
    text = url_pattern.sub(r'<a href="\1" class="chat-link">Xem chi tiết</a>', text)

    return text


def _clean_answer(text: str) -> str:
    """
    Dọn các ký tự lạ / xuống dòng cho dễ đọc.
    Trả về HTML (dùng cho bubble.innerHTML ở frontend).
    """
    if not text:
        return ""

    # bullet unicode → "- "
    text = text.replace("\u2022", "- ").replace("•", "- ")

    # loại bỏ &nbsp và khoảng trắng lạ
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    # ép các bullet đứng trên dòng riêng nếu model trả về liền mạch
    text = re.sub(r"(?<!^)(?<!\n)\s*-\s+", "\n- ", text)

    # gọn bớt nhiều dòng trống liên tiếp
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    # escape HTML để tránh injection trước khi tự thêm anchor/BR
    text = html.escape(text)

    # chuyển markdown /jobs link → <a>
    text = _markdown_links_to_html(text)

    # cuối cùng: đổi \n thành <br> để xuống dòng trong HTML (giữ khoảng trắng giữa bullet)
    text = text.replace("\n\n", "<br><br>")
    text = text.replace("\n", "<br>")

    return text


def chat_with_rag(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    current_job_id: Optional[int] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Hàm chính: nhận câu hỏi + history (+ job_id đang xem) → RAG retrieve → Gemini generate.

    Trả về:
    {
      "answer": "<HTML>",       # đã có <br>, <a>...
      "context_jobs": [ ... ],  # dùng cho gợi ý job ở UI
      "query_filters": { ... }  # phân tích cấu trúc từ câu hỏi người dùng
    }
    """
    history = history or []
    user_message = (user_message or "").strip()
    if not user_message:
        return {
            "answer": "Bạn hãy nhập câu hỏi về công việc, mức lương hoặc kỹ năng nhé.",
            "context_jobs": [],
        }

    if _is_greeting_only(user_message):
        intro = (
            "Chào bạn! Mình là trợ lý JobFinder.\n"
            "- Tìm kiếm việc làm theo từ khoá, địa điểm, mức lương bạn mong muốn.\n"
            "- Giải đáp thắc mắc chi tiết về từng job (mô tả, yêu cầu, quyền lợi) và gửi link /jobs/<id> cho bạn xem nhanh.\n"
            "- Bạn có thể nói mong muốn hoặc gửi mã job đang xem, mình sẽ tư vấn thêm cho bạn."
        )

        return {
            "answer": _clean_answer(intro),
            "context_jobs": [],
            "query_filters": {},
        }

    # 0. Phân tích câu hỏi để lấy filter có cấu trúc
    query_filters: Dict[str, Any] = {}
    try:
        query_filters = parse_user_query(user_message)
    except Exception as e:
        logger.warning("Không phân tích được câu hỏi thành bộ lọc: %s", e)

    # 1. Retrieve từ vector DB
    try:
        k = top_k or getattr(settings, "RAG_DEFAULT_TOP_K", 5)
        retrieval_query = _build_retrieval_query(user_message, history)
        docs = retrieve_jobs(
            query=retrieval_query,
            top_k=k,
            filters=query_filters,
            current_job_id=current_job_id,
        )
    except Exception as e:
        logger.exception("Lỗi retrieve_jobs: %s", e)
        return {
            "answer": (
                "Hiện tại mình đang gặp lỗi khi tìm kiếm dữ liệu công việc. "
                "Bạn thử lại sau ít phút nhé."
            ),
            "context_jobs": [],
        }

    current_job_doc: Optional[Dict[str, Any]] = None
    if current_job_id is not None:
        try:
            current_job_doc = fetch_full_job_detail(int(current_job_id))
        except Exception as e:
            logger.warning("Không lấy được full job %s sau khi retrieve: %s", current_job_id, e)

    # 2. Build prompt
    prompt = _build_prompt(
        user_message=user_message,
        docs=docs,
        history=history,
        query_filters=query_filters,
        current_job_id=current_job_id,
        current_job_doc=current_job_doc,
    )

    # 3. Gọi Gemini (đã tránh dùng response.text trực tiếp)
    try:
        model = get_gemini_model()
        temperature = getattr(settings, "GEMINI_TEMPERATURE", 0.2) or 0.2
        max_tokens = getattr(settings, "GEMINI_MAX_OUTPUT_TOKENS", 2048) or 2048

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": float(temperature),
                "top_p": 0.9,
                "top_k": 32,
                "max_output_tokens": int(max_tokens),
            },
        )

        answer_text = ""
        try:
            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                logger.warning("Gemini trả về không có candidate nào.")
            else:
                cand0 = candidates[0]
                content = getattr(cand0, "content", None)
                parts = getattr(content, "parts", None) if content is not None else None

                if not parts:
                    logger.warning(
                        "Gemini candidate không có parts, finish_reason=%s",
                        getattr(cand0, "finish_reason", None),
                    )
                else:
                    chunks: List[str] = []
                    for p in parts:
                        t = getattr(p, "text", None)
                        if t:
                            chunks.append(t)
                    answer_text = "\n".join(chunks).strip()
        except Exception as inner:
            logger.warning("Không trích được text từ response Gemini: %s", inner)
            answer_text = ""

        answer_text = _clean_answer(answer_text)
    except Exception as e:
        logger.exception("Lỗi khi gọi Gemini: %s", e)
        return {
            "answer": (
                "Hiện chatbot đang gặp sự cố khi gọi mô hình ngôn ngữ. "
                "Bạn vui lòng thử lại sau nhé."
            ),
            "context_jobs": [],
        }

    if not answer_text:
        # fallback, cũng convert sang HTML cho thống nhất
        answer_text = _clean_answer(
            "Mình chưa nhận được phản hồi rõ ràng từ mô hình. "
            "Bạn thử hỏi lại một cách cụ thể hơn nhé."
        )

    # 4. Chuẩn hoá danh sách job để FE dùng (gợi ý job)
    context_jobs: List[Dict[str, Any]] = []
    for d in docs:
        meta = d.get("metadata") or {}
        salary_text = _format_salary_block(meta)
        job_id = meta.get("id") or d.get("job_id")
        app_url = f"/jobs/{job_id}" if job_id is not None else meta.get("url")

        context_jobs.append(
            {
                "job_id": job_id,
                "title": meta.get("title"),
                "company_name": _get_company_name(meta),
                "locations": _get_locations_text(meta),
                "salary_text": salary_text,
                "url": app_url,
                "score": d.get("score"),
            }
        )

    if current_job_doc:
        meta = current_job_doc.get("metadata") or {}
        job_id = meta.get("id") or current_job_doc.get("job_id")
        already_added = any(j.get("job_id") == job_id for j in context_jobs)
        if not already_added:
            context_jobs.insert(
                0,
                {
                    "job_id": job_id,
                    "title": meta.get("title"),
                    "company_name": _get_company_name(meta),
                    "locations": _get_locations_text(meta),
                    "salary_text": _format_salary_block(meta),
                    "url": f"/jobs/{job_id}" if job_id is not None else meta.get("url"),
                    "score": 1.0,
                },
            )

    return {
        "answer": answer_text,
        "context_jobs": context_jobs,
        "query_filters": query_filters,
    }

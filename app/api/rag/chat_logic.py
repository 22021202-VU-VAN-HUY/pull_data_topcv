# app/api/rag/chat_logic.py

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from app.config import settings
from app.api.rag.retriever import retrieve_jobs

logger = logging.getLogger(__name__)

_gemini_model: Optional[genai.GenerativeModel] = None


def get_gemini_model() -> genai.GenerativeModel:
    """
    Khởi tạo & cache Gemini model.
    """
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model

    api_key = getattr(settings, "GEMINI_API_KEY", "") or ""
    model_name = getattr(settings, "GEMINI_MODEL", "gemini-1.5-flash")

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


def _get_detail_text(detail_sections: Dict[str, Any], key: str) -> str:
    sec = detail_sections.get(key) or {}
    if isinstance(sec, dict):
        return sec.get("text") or ""
    if isinstance(sec, str):
        return sec
    return ""


def _format_one_job_context(idx: int, doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata") or {}
    job_id = meta.get("id") or doc.get("job_id")
    title = meta.get("title") or ""
    url = meta.get("url") or f"/jobs/{job_id}"

    company = _get_company_name(meta)
    locations = _get_locations_text(meta)
    salary_text = _format_salary_block(meta)

    general_info = meta.get("general_info") or {}
    cap_bac = general_info.get("cap_bac")
    hinh_thuc = general_info.get("hinh_thuc_lam_viec")

    detail_sections = meta.get("detail_sections") or {}
    mo_ta = _get_detail_text(detail_sections, "mo_ta_cong_viec")
    yeu_cau = _get_detail_text(detail_sections, "yeu_cau_ung_vien")
    quyen_loi = _get_detail_text(detail_sections, "quyen_loi")

    chunk_text = doc.get("chunk_text") or ""
    score = doc.get("score")

    lines: List[str] = []
    lines.append(f"[JOB {idx}] ID nội bộ: {job_id}")
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
    lines.append(f"Link chi tiết (nếu cần hiển thị cho người dùng): {url}")
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
        lines.append("Đoạn thông tin nổi bật từ chỉ mục:")
        lines.append(chunk_text)

    return "\n".join(lines)


def _build_context_block(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return (
            "Không tìm được công việc phù hợp trong dữ liệu (theo embedding). "
            "Nếu cần, hãy trả lời là KHÔNG CÓ job phù hợp."
        )

    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        parts.append(_format_one_job_context(i, d))
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


def _build_prompt(
    user_message: str,
    docs: List[Dict[str, Any]],
    history: List[Dict[str, str]],
) -> str:
    system_prompt = (
        "Bạn là trợ lý tuyển dụng cho nền tảng JobFinder (dữ liệu lấy từ TopCV).\n"
        "- Luôn trả lời bằng tiếng Việt, giọng thân thiện, dễ hiểu.\n"
        "- CHỈ ĐƯỢC sử dụng thông tin từ phần 'NGỮ CẢNH CÔNG VIỆC (RAG)' bên dưới.\n"
        "- TUYỆT ĐỐI KHÔNG được tự bịa thêm công việc, công ty, mức lương hoặc đường link "
        "nếu chúng không xuất hiện trong ngữ cảnh.\n"
        "- Nếu trong ngữ cảnh KHÔNG có công việc nào phù hợp với yêu cầu (ví dụ không có job IT ở Hà Nội), "
        "hãy nói rõ: hiện không tìm thấy công việc phù hợp trong dữ liệu, và gợi ý người dùng tìm kiếm lại.\n"
        "- Khi nói về lương, hãy sử dụng min/max/currency/interval nếu có; nếu không có thì nói 'Thoả thuận'.\n"
        "- Nếu người dùng hỏi về kỹ năng, hãy trích từ mô tả / yêu cầu ứng viên của các job trong ngữ cảnh.\n"
        "- Câu trả lời gọn, rõ ràng, ưu tiên dùng bullet (-) và chia đoạn bằng dòng trống.\n"
    )

    context_block = _build_context_block(docs)
    history_block = _build_history_block(history)

    prompt = f"""{system_prompt}

================= NGỮ CẢNH CÔNG VIỆC (RAG) =================
{context_block}

================= LỊCH SỬ HỘI THOẠI =================
{history_block}

================= CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG =================
{user_message}

================= YÊU CẦU TRẢ LỜI =================
- Trả lời ngắn gọn, rõ ý, dùng bullet (-) khi liệt kê nhiều công việc.
- Nếu có 3–5 công việc phù hợp nhất, hãy liệt kê:
  + Tiêu đề, công ty, mức lương (text), địa điểm.
  + Link chi tiết (dùng đúng link trong ngữ cảnh).
- Nếu không có công việc phù hợp, phải nói rõ là không tìm thấy trong dữ liệu.
- Không tự tạo thêm job hoặc link ngoài danh sách trong NGỮ CẢNH.
"""
    return prompt


def _clean_answer(text: str) -> str:
    """
    Dọn các ký tự lạ / xuống dòng cho dễ đọc.
    """
    if not text:
        return ""

    # thay bullet unicode bằng '- '
    text = text.replace("\u2022", "- ").replace("•", "- ")

    # bỏ &nbsp và các khoảng trắng lạ
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)

    # gọn bớt nhiều dòng trống liên tiếp
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ========= HÀM PUBLIC: chat_with_rag =========

def chat_with_rag(
    user_message: str,
    history: Optional[List[Dict[str, str]]] = None,
    *,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Hàm chính: nhận câu hỏi + history → RAG retrieve → Gemini generate.
    """
    history = history or []
    user_message = (user_message or "").strip()
    if not user_message:
        return {
            "answer": "Bạn hãy nhập câu hỏi về công việc, mức lương hoặc kỹ năng nhé.",
            "context_jobs": [],
        }

    # 1. Retrieve từ vector DB
    try:
        k = top_k or getattr(settings, "RAG_DEFAULT_TOP_K", 5)
        docs = retrieve_jobs(query=user_message, top_k=k)
    except Exception as e:
        logger.exception("Lỗi retrieve_jobs: %s", e)
        return {
            "answer": (
                "Hiện tại mình đang gặp lỗi khi tìm kiếm dữ liệu công việc. "
                "Bạn thử lại sau ít phút nhé."
            ),
            "context_jobs": [],
        }

    # 2. Build prompt
    prompt = _build_prompt(user_message=user_message, docs=docs, history=history)

    # 3. Gọi Gemini
    try:
        model = get_gemini_model()
        # nhiệt độ thấp để hạn chế bịa
        temperature = getattr(settings, "GEMINI_TEMPERATURE", 0.2) or 0.2

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": float(temperature),
                "top_p": 0.9,
                "top_k": 32,
                "max_output_tokens": 1024,
            },
        )
        answer_text = (getattr(response, "text", None) or "").strip()
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
        answer_text = (
            "Mình chưa nhận được phản hồi rõ ràng từ mô hình. "
            "Bạn thử hỏi lại một cách cụ thể hơn nhé."
        )

    # 4. Chuẩn hoá danh sách job để FE dùng (gợi ý job)
    context_jobs: List[Dict[str, Any]] = []
    for d in docs:
        meta = d.get("metadata") or {}
        salary_text = _format_salary_block(meta)
        context_jobs.append(
            {
                "job_id": meta.get("id") or d.get("job_id"),
                "title": meta.get("title"),
                "company_name": _get_company_name(meta),
                "locations": _get_locations_text(meta),
                "salary_text": salary_text,
                "url": meta.get("url"),
                "score": d.get("score"),
            }
        )

    return {
        "answer": answer_text,
        "context_jobs": context_jobs,
    }

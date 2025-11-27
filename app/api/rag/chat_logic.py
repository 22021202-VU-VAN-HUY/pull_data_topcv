# app/api/rag/chat_logic.py

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.config import settings
from app.api.rag.retriever import retrieve_jobs

import google.generativeai as genai


logger = logging.getLogger(__name__)

# Lazy init Gemini model
_gemini_model = None


def get_gemini_model():
    """
    Khởi tạo và cache model Gemini dựa trên settings.
    """
    global _gemini_model

    if _gemini_model is not None:
        return _gemini_model

    api_key = getattr(settings, "GEMINI_API_KEY", "") or ""
    model_name = getattr(settings, "GEMINI_MODEL", "gemini-1.5-flash")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY chưa được cấu hình. Hãy đặt trong file .env và app/config.py."
        )

    genai.configure(api_key=api_key)
    _gemini_model = genai.GenerativeModel(model_name)

    logger.info("Gemini model initialized: %s", model_name)
    return _gemini_model


# ========= RAG PROMPT BUILDING =========


def _format_salary_block(hit: Dict[str, Any]) -> str:
    """
    Ưu tiên salary_text, nếu không có thì build từ min/max.
    """
    salary_text = hit.get("salary_text")
    if salary_text:
        return salary_text

    salary_min = hit.get("salary_min")
    salary_max = hit.get("salary_max")
    currency = hit.get("salary_currency") or "VND"
    interval = hit.get("salary_interval") or "MONTH"

    interval_vi = {
        "MONTH": "/tháng",
        "YEAR": "/năm",
        "HOUR": "/giờ",
    }.get(interval, "")

    # Nếu không có min/max → trả về chuỗi chung
    if salary_min is None and salary_max is None:
        return "Thoả thuận"

    if salary_min is not None and salary_max is not None:
        return f"Từ {salary_min:,.0f} đến {salary_max:,.0f} {currency} {interval_vi}"

    if salary_min is not None:
        return f"Từ {salary_min:,.0f} {currency} {interval_vi}"

    # chỉ có max
    return f"Đến {salary_max:,.0f} {currency} {interval_vi}"


def _format_one_job_context(idx: int, hit: Dict[str, Any]) -> str:
    """
    Định dạng 1 context job để đưa vào prompt cho LLM.
    """
    job_id = hit.get("job_id")
    title = hit.get("title") or ""
    company = hit.get("company_name") or hit.get("company") or ""
    city = hit.get("city") or ""
    url = hit.get("job_url") or hit.get("url") or ""
    chunk_type = hit.get("chunk_type") or ""
    chunk_text = hit.get("chunk_text") or hit.get("text") or ""
    score = hit.get("score")

    salary_block = _format_salary_block(hit)

    lines = [f"[JOB {idx}] ID: {job_id}"]
    if title:
        lines.append(f"Tiêu đề: {title}")
    if company:
        lines.append(f"Công ty: {company}")
    if city:
        lines.append(f"Địa điểm: {city}")
    if salary_block:
        lines.append(f"Mức lương: {salary_block}")
    if chunk_type:
        lines.append(f"Loại thông tin: {chunk_type}")
    if chunk_text:
        lines.append(f"Chi tiết: {chunk_text}")
    if url:
        lines.append(f"Link: {url}")
    if score is not None:
        lines.append(f"(Độ liên quan nội bộ: {score:.3f})")

    return "\n".join(lines)


def build_context_block(retrieved: List[Dict[str, Any]]) -> str:
    """
    Xây dựng khối context gồm N job gần nhất theo RAG.
    """
    if not retrieved:
        return "Không tìm được công việc nào phù hợp trong dữ liệu."

    parts = []
    for i, hit in enumerate(retrieved, start=1):
        parts.append(_format_one_job_context(i, hit))

    return "\n\n---\n\n".join(parts)


def build_history_block(history: List[Dict[str, str]]) -> str:
    """
    History là list các dict: {"role": "user"/"assistant", "content": "..."}.
    Gom lại thành text để LLM hiểu ngữ cảnh.
    """
    if not history:
        return "Chưa có lịch sử hội thoại trước đó."

    lines = ["Lịch sử hội thoại trước đó (tin nhắn gần nhất ở cuối):"]
    for msg in history:
        role = msg.get("role") or "user"
        content = (msg.get("content") or "").strip()
        if not content:
            continue

        role_vi = "Người dùng" if role == "user" else "Trợ lý"
        lines.append(f"{role_vi}: {content}")

    return "\n".join(lines)


def build_rag_prompt(
    user_message: str,
    retrieved: List[Dict[str, Any]],
    history: List[Dict[str, str]],
) -> str:
    """
    Combine: system instructions + context (jobs) + history + current question.
    """
    system_prompt = (
        "Bạn là trợ lý tuyển dụng của nền tảng JobFinder, nhiệm vụ:\n"
        "- Giải thích & tư vấn cho ứng viên dựa trên thông tin công việc đã cho.\n"
        "- Ưu tiên các công việc còn hạn tuyển (đã được hệ thống lọc sẵn).\n"
        "- Luôn trả lời bằng tiếng Việt, giọng thân thiện, rõ ràng.\n"
        "- Khi nói về lương, hãy dùng đúng thông tin min/max/currency/chu kỳ nếu có.\n"
        "- Nếu người dùng hỏi về mức lương, hãy nêu ra khoảng lương cho từng job liên quan, "
        "so sánh ngắn gọn.\n"
        "- Nếu người dùng hỏi về kỹ năng, hãy trích ra từ các phần mô tả/ yêu cầu ứng viên.\n"
        "- Nếu không đủ thông tin trong context, hãy nói thẳng là không chắc thay vì bịa."
    )

    context_block = build_context_block(retrieved)
    history_block = build_history_block(history)

    prompt = f"""{system_prompt}

================= NGỮ CẢNH CÔNG VIỆC (RAG) =================
{context_block}

================= LỊCH SỬ HỘI THOẠI =================
{history_block}

================= CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG =================
{user_message}

================= YÊU CẦU TRẢ LỜI =================
- Trả lời ngắn gọn, súc tích, tối đa 3–6 đoạn nhỏ.
- Nếu đề xuất nhiều job, hãy đánh số hoặc dùng bullet.
- Có thể gợi ý 1–3 công việc phù hợp nhất kèm link (nếu có).
- Đừng nhắc lại nguyên văn toàn bộ context, chỉ tóm tắt phần cần thiết.
- Nếu câu hỏi chung chung (ví dụ: 'em nên chọn job nào?'), hãy hỏi lại 1–2 câu để làm rõ tiêu chí
  (mức lương mong muốn, địa điểm, kinh nghiệm, kỹ năng...)."""

    return prompt


# ========= PUBLIC ENTRYPOINT =========


def chat_with_rag(
    user_message: str,
    history: List[Dict[str, str]] | None = None,
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Hàm chính: nhận câu hỏi + history → RAG retrieve + Gemini generate.
    Trả về:
        {
          "answer": "...",
          "context_jobs": [...],   # để FE hiển thị gợi ý jobs
        }
    """
    history = history or []
    user_message = (user_message or "").strip()

    if not user_message:
        return {
            "answer": "Bạn hãy nhập câu hỏi về công việc, lương hoặc kỹ năng nhé.",
            "context_jobs": [],
        }

    # 1) Retrieve jobs từ vector DB (đã ưu tiên job còn hạn bên retriever)
    try:
        retrieved = retrieve_jobs(query=user_message, top_k=top_k)
    except Exception as e:
        logger.exception("Lỗi retrieve_jobs: %s", e)
        return {
            "answer": "Hiện tại mình đang gặp lỗi khi tìm kiếm dữ liệu công việc. "
                      "Bạn thử lại sau ít phút nhé.",
            "context_jobs": [],
        }

    # 2) Build prompt cho LLM
    prompt = build_rag_prompt(
        user_message=user_message,
        retrieved=retrieved,
        history=history,
    )

    # 3) Gọi Gemini
    try:
        model = get_gemini_model()
        response = model.generate_content(prompt)
        answer_text = (getattr(response, "text", None) or "").strip()
    except Exception as e:
        logger.exception("Lỗi gọi Gemini: %s", e)
        return {
            "answer": "Hiện tại chatbot đang gặp trục trặc khi gọi mô hình ngôn ngữ. "
                      "Bạn vui lòng thử lại sau nhé.",
            "context_jobs": [],
        }

    if not answer_text:
        answer_text = (
            "Mình chưa nhận được phản hồi rõ ràng từ mô hình. "
            "Bạn thử hỏi lại câu khác hoặc cụ thể hơn nhé."
        )

    # 4) Chuẩn hóa danh sách job để FE dễ dùng
    context_jobs: List[Dict[str, Any]] = []
    for hit in retrieved:
        context_jobs.append(
            {
                "job_id": hit.get("job_id"),
                "title": hit.get("title"),
                "company_name": hit.get("company_name") or hit.get("company"),
                "city": hit.get("city"),
                "salary_min": hit.get("salary_min"),
                "salary_max": hit.get("salary_max"),
                "salary_currency": hit.get("salary_currency"),
                "salary_interval": hit.get("salary_interval"),
                "salary_text": hit.get("salary_text") or _format_salary_block(hit),
                "url": hit.get("job_url") or hit.get("url"),
                "chunk_type": hit.get("chunk_type"),
                "score": hit.get("score"),
            }
        )

    return {
        "answer": answer_text,
        "context_jobs": context_jobs,
    }

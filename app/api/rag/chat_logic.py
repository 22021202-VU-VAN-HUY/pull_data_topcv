# app/api/rag/chat_logic.py

from __future__ import annotations

import html
import json
import logging
import re
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from app.config import settings
from app.api.rag.retriever import retrieve_jobs
from app.api.rag.query_parser import parse_user_query

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Bạn là trợ lý tuyển dụng của nền tảng JobFinder.

NGUYÊN TẮC CHUNG:
- Trả lời bằng TIẾNG VIỆT, giọng thân thiện, tự nhiên, dễ hiểu.
- CHỈ dùng thông tin trong NGỮ CẢNH CÔNG VIỆC (context) được cung cấp.
- KHÔNG tự bịa ra mức lương, yêu cầu, quyền lợi, địa điểm hoặc tên công việc mới.
- Nếu người dùng hỏi chủ đề ngoài tuyển dụng (ví dụ giá vàng, thời tiết, bóng đá...), hãy phản hồi ngắn gọn, lịch sự: nói rằng bạn tập trung hỗ trợ việc làm, đưa một câu trả lời chung chung nếu phù hợp, và mời người dùng quay lại câu hỏi liên quan công việc.
- Nếu ngữ cảnh không đủ thông tin để trả lời một phần nào đó của câu hỏi,
  hãy nói rõ: "Trong mô tả công việc hiện tại không ghi rõ về vấn đề này."
  hoặc "Em không tìm thấy thông tin này trong dữ liệu hiện có."
- ƯU TIÊN dùng URL nội bộ JobFinder dạng /jobs/<id> nếu cần đưa link job cho người dùng.
- Không cần giải thích về hệ thống RAG hay cơ sở dữ liệu, chỉ trả lời như một HR đang tư vấn ứng viên.
""".strip()

UNIFIED_PROMPT = """
{system_prompt}

Dưới đây là thông tin đã được hệ thống truy xuất từ cơ sở dữ liệu việc làm (context).
Bạn CHỈ ĐƯỢC sử dụng thông tin trong context để trả lời.

INTENT: {intent}
FILTERS (JSON): {filters_json}

CONTEXT (các job, mỗi job có id, tiêu đề, công ty, lương, địa điểm, mô tả, yêu cầu, quyền lợi,...):
----------------
{context}
----------------

Câu hỏi của người dùng:
"{question}"

CÁCH TRẢ LỜI THEO INTENT:

1) Nếu INTENT = "ask_detail":
   - Người dùng đang hỏi chi tiết về 1 công việc cụ thể (ví dụ: lương, yêu cầu, quyền lợi, địa điểm...).
   - Trả lời NGẮN GỌN, trực tiếp, 2–5 câu.
   - KHÔNG cần liệt kê danh sách job, KHÔNG cần đưa link.
   - Nếu câu hỏi về lương: nêu rõ khoảng lương nếu context có
     (ví dụ: "Mức lương khoảng từ 12.000.000 đến 14.000.000 VND/tháng.").
   - Nếu context không có thông tin được hỏi (edge-case): nói rõ
     "Trong mô tả công việc hiện tại không ghi rõ về vấn đề này." và không bịa thêm.

2) Nếu INTENT = "search_jobs":
   - Người dùng muốn TÌM KIẾM hoặc GỢI Ý CÔNG VIỆC MỚI.
   - Hãy chọn 3–5 job PHÙ HỢP NHẤT trong context.
   - Trả lời dạng bullet, mỗi job 1 dòng:
     - {{title}} – {{company}}; lương: {{tóm tắt lương hoặc "thoả thuận"}}; địa điểm: {{địa điểm chính}}. Link: /jobs/{{id}}
   - Nếu không có job phù hợp, hãy giải thích ngắn gọn và gợi ý người dùng nới tiêu chí (KHÔNG bịa job).

3) Nếu INTENT = "compare_jobs":
   - Người dùng muốn SO SÁNH một vài job trong context.
   - Hãy so sánh ngắn gọn về:
     + Mức lương (ai cao hơn / thấp hơn / tương đương)
     + Yêu cầu ứng viên (job nào đòi hỏi nhiều hơn)
     + Quyền lợi nổi bật nếu có.
   - Kết luận 1–2 câu: nên chọn job nào nếu ưu tiên lương, hoặc ưu tiên yêu cầu nhẹ.
   - KHÔNG bắt buộc phải đưa link.

4) Nếu INTENT = "other":
   - Trả lời ngắn gọn, thân thiện. Nếu câu hỏi liên quan tuyển dụng thì dựa trên context.
   - Nếu câu hỏi không liên quan đến việc làm: đáp ngắn gọn, lịch sự (có thể một câu trả lời chung chung),
     nhắc bạn là trợ lý tuyển dụng và mời người dùng đặt câu hỏi về công việc.
   - Nếu context không đủ để trả lời nội dung tuyển dụng, hãy nói rõ không tìm thấy thông tin và không bịa thêm.

NGUYÊN TẮC BẮT BUỘC:
- Chỉ dùng thông tin trong CONTEXT để khẳng định chi tiết (lương, yêu cầu, quyền lợi, địa điểm...).
- Nếu context không đủ, hãy nói rõ "không tìm thấy thông tin trong dữ liệu hiện có" thay vì đoán.
- ƯU TIÊN dùng URL nội bộ JobFinder dạng /jobs/<id> khi cần đưa link job.
- Trả lời bằng tiếng Việt, giọng thân thiện, tự nhiên.
""".strip()

_unified_model: Optional[genai.GenerativeModel] = None


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


def _get_title_upper(meta: Dict[str, Any]) -> str:
    """Lấy tiêu đề và chuyển thành chữ in hoa để đồng nhất hiển thị."""

    title = meta.get("title") or ""
    return title.upper()


def build_context_text(retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Ghép các chunk lại thành 1 context text để đưa vào LLM.
    Ưu tiên include thông tin job_id, title, company cho dễ đọc.
    """

    parts: List[str] = []
    for d in retrieved_docs:
        meta = d.get("metadata") or {}
        job_id = meta.get("id") or d.get("job_id")
        title = meta.get("title") or ""
        company_obj = meta.get("company") or {}
        company_name = (
            company_obj.get("name")
            if isinstance(company_obj, dict)
            else str(company_obj or "")
        )

        header = f"[JOB {job_id}] {title} – {company_name}".strip()
        salary_text = _format_salary_block(meta)
        location_text = _get_locations_text(meta)
        details: List[str] = []
        if salary_text:
            details.append(f"lương: {salary_text}")
        if location_text:
            details.append(f"địa điểm: {location_text}")
        if details:
            header = f"{header} ({'; '.join(details)})"

        chunk_text = d.get("chunk_text") or ""
        parts.append(header + "\n" + chunk_text)

    return "\n\n".join(parts)


def _get_unified_model() -> genai.GenerativeModel:
    global _unified_model
    if _unified_model is not None:
        return _unified_model

    api_key = getattr(settings, "GEMINI_API_KEY", "") or ""
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY chưa được cấu hình.")

    model_name = getattr(settings, "GEMINI_CHAT_MODEL", "") or "gemini-2.0-flash"

    genai.configure(api_key=api_key)
    _unified_model = genai.GenerativeModel(model_name)
    return _unified_model


def generate_answer_unified(
    user_message: str, filters: Dict[str, Any], retrieved_docs: List[Dict[str, Any]]
) -> str:
    model = _get_unified_model()
    context_text = build_context_text(retrieved_docs)
    filters_json = json.dumps(filters or {}, ensure_ascii=False)

    prompt = UNIFIED_PROMPT.format(
        system_prompt=SYSTEM_PROMPT,
        intent=(filters.get("intent") or "other"),
        filters_json=filters_json,
        context=context_text[:12000],
        question=user_message,
    )

    resp = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 32,
            "max_output_tokens": 512,
        },
    )

    text = ""
    try:
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            cand = candidates[0]
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                buf: List[str] = []
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        buf.append(t)
                text = "".join(buf).strip()

        if not text:
            raw = getattr(resp, "text", None)
            if raw:
                text = raw.strip()
    except Exception:
        raw = getattr(resp, "text", None)
        if raw:
            text = raw.strip()

    return text or "Hiện tại em chưa trả lời được câu hỏi này từ dữ liệu có sẵn."


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


# = PHÂN LOẠI Ý ĐỊNH CƠ BẢN =
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


# == CLEAN + HTML HOÁ CÂU TRẢ LỜI ==


def _markdown_links_to_html(text: str) -> str:
    """
    - [link](/jobs/123) -> <a href="/jobs/123">link</a>
    - /jobs/123 hoặc jobs/123 -> <a href="/jobs/123">Xem chi tiết</a>
    (Không động tới link TopCV để tránh user bị dẫn ra ngoài nếu không cần.)
    """
    if not text:
        return ""

    # Chỉ convert markdown có URL nội bộ /jobs/xxx (cho phép thiếu dấu "/" ở đầu)
    md_pattern = re.compile(r"\[([^\]]+)\]\((/?jobs/\d+)\)")
    text = md_pattern.sub(
        lambda m: (
            f'<a href="/{m.group(2).lstrip("/")}" class="chat-link">{m.group(1)}</a>'
        ),
        text,
    )

    # Convert đường dẫn /jobs/123 hoặc jobs/123 trần thành link có anchor "Xem chi tiết"
    url_pattern = re.compile(r"/?jobs/\d+")
    text = url_pattern.sub(lambda m: f'<a href="/{m.group(0).lstrip("/")}" class="chat-link">Xem chi tiết</a>', text)
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

    # 2. Gọi Gemini với unified prompt
    try:
        answer_raw = generate_answer_unified(user_message, query_filters, docs)
        answer_text = _clean_answer(answer_raw)
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
                "title": _get_title_upper(meta),
                "company_name": _get_company_name(meta),
                "locations": _get_locations_text(meta),
                "salary_text": salary_text,
                "url": app_url,
                "score": d.get("score"),
            }
        )

    return {
        "answer": answer_text,
        "context_jobs": context_jobs,
        "query_filters": query_filters,
    }

# app/api/rag/query_parser.py

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from app.config import settings

logger = logging.getLogger(__name__)

_parser_model: Optional[genai.GenerativeModel] = None


def _get_parser_model() -> genai.GenerativeModel:
    """
    Model nhẹ dùng riêng cho việc phân tích câu hỏi → JSON filter.
    Ưu tiên dùng model flash để rẻ & nhanh.
    """
    global _parser_model
    if _parser_model is not None:
        return _parser_model

    api_key = getattr(settings, "GEMINI_API_KEY", "") or ""
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY chưa được cấu hình.")

    # Cho phép override qua env; mặc định dùng flash
    model_name = (
        getattr(settings, "GEMINI_QUERY_MODEL", "") or "gemini-2.0-flash"
    )

    genai.configure(api_key=api_key)
    _parser_model = genai.GenerativeModel(model_name)
    logger.info("Query parser model initialized: %s", model_name)
    return _parser_model


def _default_filters() -> Dict[str, Any]:
    return {
        "intent": "other",   # NEW
        "job_keywords": [],   # từ khoá ngành / chức danh
        "locations": [],      # ["Hà Nội", "Hồ Chí Minh", ...]
        "min_salary_vnd": None,
        "max_salary_vnd": None,
        "skills": [],         # ["thuyết trình", "giao tiếp", ...]
    }


def parse_user_query(user_message: str) -> Dict[str, Any]:
    """
    Dùng Gemini để bóc tách câu hỏi thành filter có cấu trúc.
    Luôn trả về dict với đủ key (có thể None / [] nếu không suy ra được).
    """
    base = _default_filters()
    msg = (user_message or "").strip()
    if not msg:
        return base

    try:
        model = _get_parser_model()

        prompt = f"""
Bạn là module phân tích câu hỏi tuyển dụng, nhiệm vụ là TRẢ VỀ JSON DUY NHẤT.

Hãy đọc câu hỏi của người dùng (tiếng Việt) và trích xuất các trường sau:

- intent: một trong các giá trị:
  - "search_jobs": nếu người dùng muốn TÌM KIẾM hoặc GỢI Ý CÔNG VIỆC MỚI
    (ví dụ: "tìm giúp em việc marketing ở HCM", "có job IT nào lương trên 15tr không?")
  - "ask_detail": nếu người dùng đang hỏi chi tiết về MỘT CÔNG VIỆC CỤ THỂ
    (ví dụ: "công việc này lương bao nhiêu?", "job kế toán bên ABC yêu cầu gì?")
  - "compare_jobs": nếu người dùng muốn SO SÁNH CÁC CÔNG VIỆC
    (ví dụ: "giữa 2 job A và B thì job nào lương cao hơn?")
  - "other": các trường hợp còn lại.

- job_keywords: danh sách từ khoá ngành, vị trí, lĩnh vực (ví dụ: ["IT", "lập trình", "developer"]).
- locations: danh sách địa điểm (tỉnh/thành, quận, khu vực...), ưu tiên dạng tên tỉnh/thành (ví dụ: ["Hà Nội"]).
- min_salary_vnd: mức lương tối thiểu *ước tính* theo VND, nếu người dùng nói "từ 10tr", "trên 15 triệu"...
- max_salary_vnd: mức lương tối đa *ước tính* theo VND, nếu người dùng nói "đến 20tr"...
- skills: danh sách kỹ năng hoặc yêu cầu (ví dụ: ["thuyết trình", "giao tiếp", "tiếng Anh"]).

Yêu cầu:
- Nếu không suy ra được một trường thì để giá trị null (với số) hoặc [] (với danh sách).
- intent luôn phải là một trong: "search_jobs", "ask_detail", "compare_jobs", "other".
- TẤT CẢ TIỀN LƯƠNG quy đổi sang VND, ví dụ "10 triệu" → 10000000.
- Chỉ TRẢ VỀ JSON THUẦN, KHÔNG giải thích thêm.

Câu hỏi người dùng:
\"\"\"{msg}\"\"\" 
"""

        # ⚠️ Quan trọng: ép model trả về application/json
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "top_p": 0.9,
                "top_k": 32,
                "max_output_tokens": 128,
                "response_mime_type": "application/json",
            },
        )

        text = ""

        # Ưu tiên lấy từ candidates[0].content.parts[*].text
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

            # fallback sang resp.text nếu trên không lấy được
            if not text:
                raw = getattr(resp, "text", None)
                if raw:
                    text = raw.strip()
        except Exception as e:
            logger.warning(
                "parse_user_query: lỗi khi đọc candidates, fallback resp.text: %s",
                e,
            )
            raw = getattr(resp, "text", None)
            if raw:
                text = raw.strip()

        if not text:
            logger.warning(
                "parse_user_query: model không trả về text (finish_reason có thể là MAX_TOKENS)."
            )
            return base

        # Vì đã ép response_mime_type=application/json,
        # thông thường text chính là JSON thuần, nhưng vẫn để phòng hờ.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            logger.warning(
                "parse_user_query: không tìm thấy JSON trong đáp án: %r",
                text[:200],
            )
            return base

        json_str = text[start : end + 1]
        data = json.loads(json_str)

        # Merge vào base để đảm bảo đủ key
        result = _default_filters()
        for k in result.keys():
            if k in data:
                result[k] = data.get(k)
        return result

    except Exception as e:
        logger.exception("parse_user_query lỗi: %s", e)
        return base

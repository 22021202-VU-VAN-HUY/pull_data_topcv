# evaluate_chatbot.py

import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from google import genai

from app.api.rag.chat_logic import chat_with_rag
from .question_schema import TestQuestion
from .testing_config import (
    QUESTIONS_FILE,
    EVAL_OUTPUT_FILE,
    SUMMARY_OUTPUT_FILE,
    JUDGE_MODEL_NAME,
    ENABLE_LLM_JUDGE,
    MAX_QUESTIONS,
    SLEEP_BETWEEN_QUESTIONS,
    GEMINI_API_KEY,
)


# ========== DATA CLASS KẾT QUẢ ==========


@dataclass
class EvalResult:
    question_id: str
    intent: str
    difficulty: str
    specificity: str
    judge_score: Optional[float] # điểm số /5.0
    question_text: str
    expected_behavior: str
    gold_context_ids: List[int]

    # Kết quả từ chatbot
    answer: str
    retrieved_job_ids: List[int]

    # LLM-as-judge
    judge_reason: Optional[str]

    # Metric retriever
    recall_at_5: Optional[float]
    recall_at_10: Optional[float]

    


# == LOAD DATASET CÂU HỎI ==


def load_test_questions(path: str, max_questions: Optional[int] = None) -> List[TestQuestion]:
    questions: List[TestQuestion] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = TestQuestion(
                id=obj["id"],
                intent=obj["intent"],
                difficulty=obj["difficulty"],
                specificity=obj["specificity"],
                question_text=obj["question_text"],
                expected_behavior=obj["expected_behavior"],
                gold_context_ids=obj.get("gold_context_ids", []),
                anti_preference=obj.get("anti_preference"),
            )
            questions.append(q)
            if max_questions is not None and len(questions) >= max_questions:
                break
    return questions


# == GỌI CHATBOT (GỌI THẲNG RAG) ==


def call_chatbot(question_text: str) -> Dict[str, Any]:
    """
    Gọi trực tiếp pipeline chat_with_rag trong app/api/rag/chat_logic.py

    chat_with_rag(
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        *,
        current_job_id: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]

    Trả về (theo chat_logic của bạn):
    {
      "answer": "<HTML/text>",
      "context_jobs": [
        {
          "job_id": int | None,
          "title": str,
          "company_name": str,
          "locations": str,
          "salary_text": str,
          "url": str,
          "score": float | None,
        },
        ...
      ],
      "query_filters": { ... }
    }
    """
    return chat_with_rag(user_message=question_text)


def extract_answer_context_and_ids(
    chatbot_response: Dict[str, Any]
) -> (str, List[Dict[str, Any]], List[int]):
    """
    Lấy ra:
      - answer: text/HTML trả cho user
      - context_jobs: list job mà RAG dùng làm ngữ cảnh
      - retrieved_job_ids: list job_id dùng để tính Recall@k
    """
    answer = chatbot_response.get("answer") or ""
    context_jobs = chatbot_response.get("context_jobs") or []
    if not isinstance(context_jobs, list):
        context_jobs = []

    job_ids: List[int] = []
    for job in context_jobs:
        # chat_logic dùng key "job_id"
        jid = job.get("job_id") or job.get("id")
        if isinstance(jid, int):
            job_ids.append(jid)
        elif isinstance(jid, str) and jid.isdigit():
            job_ids.append(int(jid))

    return answer, context_jobs, job_ids


# ========== METRICS: RECALL@K ==========


def compute_recall_at_k(gold_ids: List[int], retrieved_ids: List[int], k: int) -> Optional[float]:
    if not gold_ids:
        return None  # câu hỏi không kỳ vọng context cụ thể
    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    gold_set = set(gold_ids)
    hit = len(gold_set.intersection(top_k))
    total = len(gold_set)
    return hit / total if total > 0 else None


# ========== LLM-AS-JUDGE ==========


def build_judge_prompt(
    question: TestQuestion,
    answer: str,
    context_jobs: List[Dict[str, Any]],
) -> str:
    """
    Prompt cho Gemini làm giám khảo.

    Thay vì dùng gold_context_ids (có thể sai / chỉ là gợi ý),
    ta đưa cho judge:
      - câu hỏi + expected_behavior
      - câu trả lời
      - danh sách job mà RAG đã dùng làm ngữ cảnh (context_jobs, dạng tóm tắt)
    """

    # Lấy toàn bộ context_jobs, chỉ giữ field quan trọng để prompt gọn
    jobs_for_prompt: List[Dict[str, Any]] = []
    for job in context_jobs:
        jobs_for_prompt.append(
            {
                # Cố ý KHÔNG gửi job_id để judge khỏi bám vào ID
                "title": job.get("title"),
                "company_name": job.get("company_name"),
                "locations": job.get("locations"),
                "salary_text": job.get("salary_text"),
                "score": job.get("score"),
            }
        )

    data = {
        "question": {
            "id": question.id,
            "intent": question.intent,
            "difficulty": question.difficulty,
            "specificity": question.specificity,
            "text": question.question_text,
            "expected_behavior": question.expected_behavior,
            # KHÔNG gửi gold_context_ids nữa cho judge
        },
        "model_answer": {
            "text": answer,
        },
        "retrieved_context_jobs": jobs_for_prompt,
    }

    instructions = (
        "Bạn là giám khảo đánh giá chất lượng câu trả lời của chatbot tuyển dụng JobFinder.\n"
        "Nhiệm vụ:\n"
        "1. Đọc kỹ câu hỏi (question) và mô tả hành vi mong đợi (expected_behavior).\n"
        "2. Đọc câu trả lời của chatbot (model_answer.text).\n"
        "3. Tham khảo danh sách job mà hệ thống đã dùng làm ngữ cảnh "
        "(retrieved_context_jobs) nếu cần.\n"
        "4. Chấm điểm từ 0.0 đến 5.0 theo tiêu chí:\n"
        "   - 0: Sai hoàn toàn / vô nghĩa / không trả lời đúng ý.\n"
        "   - 1–2: Trả lời được một phần nhỏ, còn nhiều thiếu sót hoặc lan man.\n"
        "   - 3–4: Trả lời khá tốt, đáp ứng phần lớn expected_behavior, còn thiếu chút chi tiết.\n"
        "   - 5: Trả lời rất tốt, đầy đủ, bám sát expected_behavior, không bịa thông tin.\n"
        "\n"
        "Lưu ý QUAN TRỌNG:\n"
        "- retrieved_context_jobs CHỈ là tóm tắt (tiêu đề, công ty, địa điểm, lương...), "
        "KHÔNG chứa toàn bộ nội dung tin tuyển dụng.\n"
        "- Bạn KHÔNG được kết luận rằng các chi tiết về kinh nghiệm/kỹ năng/phụ cấp/...(trong mô tả chi tiết) trong câu trả lời là 'bịa đặt' "
        "chỉ vì chúng có thể không xuất hiện trong retrieved_context_jobs.\n"
        "- Bạn cũng KHÔNG có quyền truy cập trực tiếp vào toàn bộ database công việc.\n"
        "- Hãy tập trung đánh giá xem câu trả lời có:\n"
        "  + Đúng intent (ví dụ: mô tả yêu cầu kinh nghiệm & kỹ năng khi intent là 'ask_detail').\n"
        "  + Phù hợp với loại công việc và chức danh (ví dụ: BrSE, kế toán trưởng, nhân viên sale...).\n"
        "  + Tránh mâu thuẫn rõ ràng với câu hỏi hoặc với thông tin tóm tắt trong retrieved_context_jobs.\n"
        "- Nếu câu trả lời đưa ra các chi tiết nghe có vẻ hợp lý với vị trí đó, "
        "nhưng bạn không thể kiểm chứng 100%, hãy đánh giá chủ yếu theo mức độ hợp lý và độ bám sát expected_behavior "
        "thay vì phạt nặng vì nghi ngờ bịa.\n"
        "- Chỉ chấm 0–1 điểm khi câu trả lời hoàn toàn lạc đề, sai hẳn intent (trường hợp tìm đúng tên Job thì vẫn có thể cho điểm cao hơn bởi vì chatbot được yêu cầu rằng không bịa và thông tin nhận được của chatbot có thể nhiều hơn), "
        "hoặc chứa thông tin rõ ràng vô lý/mâu thuẫn với loại công việc.\n"
        "- Câu trả lời phải bằng tiếng Việt, rõ ràng, dễ hiểu.\n"
        "\n"
        "ĐẦU RA BẮT BUỘC: chỉ trả về MỘT object JSON với 2 field:\n"
        '{\"score\": number, \"reason\": string}\n'
        "Không thêm text nào khác ngoài JSON."
    )

    prompt = instructions + "\n\nDỮ LIỆU ĐẦU VÀO:\n" + json.dumps(data, ensure_ascii=False, indent=2)
    return prompt

def extract_json_object(raw_text: str) -> str:
    """
    Lấy object JSON đầu tiên trong chuỗi text (từ '{' đến '}' khớp cặp).
    """
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Không tìm thấy JSON object trong output của judge.")
    return raw_text[start : end + 1]


def judge_answer_with_gemini(
    client: genai.Client,
    question: TestQuestion,
    answer: str,
    context_jobs: List[Dict[str, Any]],
) -> (float, str):
    prompt = build_judge_prompt(question, answer, context_jobs)
    resp = client.models.generate_content(
        model=JUDGE_MODEL_NAME,
        contents=prompt,
    )
    raw = resp.text or ""

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = json.loads(extract_json_object(raw))

    score = float(obj.get("score", 0.0))
    reason = str(obj.get("reason", ""))
    return score, reason


# ========== MAIN EVAL LOOP ==========


def main() -> None:
    # Chuẩn bị LLM judge (nếu bật)
    judge_client: Optional[genai.Client] = None
    if ENABLE_LLM_JUDGE:
        judge_client = genai.Client(api_key=GEMINI_API_KEY)

    # Load bộ câu hỏi
    questions = load_test_questions(QUESTIONS_FILE, max_questions=MAX_QUESTIONS)
    print(f"Đã load {len(questions)} câu hỏi từ {QUESTIONS_FILE}")

    # Xoá file kết quả cũ nếu có
    open(EVAL_OUTPUT_FILE, "w", encoding="utf-8").close()

    all_scores: List[float] = []
    all_recall5: List[float] = []
    all_recall10: List[float] = []

    for idx, q in enumerate(questions, start=1):
        print(f"\n=== CÂU {idx}/{len(questions)}: {q.id} ===")
        print(f"- Intent: {q.intent}, difficulty={q.difficulty}, spec={q.specificity}")

        # 1) Gọi chatbot (chat_with_rag)
        try:
            chatbot_resp = call_chatbot(q.question_text)
        except Exception as e:
            print(f"❌ Lỗi gọi chatbot: {e}")
            answer = ""
            context_jobs: List[Dict[str, Any]] = []
            retrieved_ids: List[int] = []
        else:
            answer, context_jobs, retrieved_ids = extract_answer_context_and_ids(chatbot_resp)

        print(f"- Answer (rút gọn 100 ký tự): {answer[:100]!r}")
        print(f"- Retrieved job_ids (top 10): {retrieved_ids[:10]}")

        # 2) Tính Recall@5, Recall@10 dựa trên gold_context_ids
        r5 = compute_recall_at_k(q.gold_context_ids, retrieved_ids, k=5)
        r10 = compute_recall_at_k(q.gold_context_ids, retrieved_ids, k=10)

        if r5 is not None:
            all_recall5.append(r5)
        if r10 is not None:
            all_recall10.append(r10)

        # 3) Gọi LLM-as-judge (nếu bật)
        judge_score: Optional[float] = None
        judge_reason: Optional[str] = None

        if ENABLE_LLM_JUDGE and judge_client is not None and answer.strip():
            try:
                judge_score, judge_reason = judge_answer_with_gemini(
                    judge_client, q, answer, context_jobs
                )
                all_scores.append(judge_score)
                print(f"- Judge score: {judge_score:.2f}")
            except Exception as e:
                print(f"❌ Lỗi judge: {e}")

        # 4) Lưu kết quả vào JSONL
        result = EvalResult(
            question_id=q.id,
            intent=q.intent,
            difficulty=q.difficulty,
            specificity=q.specificity,
            question_text=q.question_text,
            expected_behavior=q.expected_behavior,
            gold_context_ids=q.gold_context_ids,
            answer=answer,
            retrieved_job_ids=retrieved_ids,
            recall_at_5=r5,
            recall_at_10=r10,
            judge_score=judge_score,
            judge_reason=judge_reason,
        )

        with open(EVAL_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

        # 5) Nghỉ giữa các câu để tránh quota
        if SLEEP_BETWEEN_QUESTIONS and SLEEP_BETWEEN_QUESTIONS > 0:
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    # 6) Ghi summary
    summary: Dict[str, Any] = {
        "num_questions": len(questions),
        "avg_judge_score": sum(all_scores) / len(all_scores) if all_scores else None,
        "avg_recall_at_5": sum(all_recall5) / len(all_recall5) if all_recall5 else None,
        "avg_recall_at_10": sum(all_recall10) / len(all_recall10) if all_recall10 else None,
    }

    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== TÓM TẮT ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Kết quả chi tiết lưu ở: {EVAL_OUTPUT_FILE}")
    print(f"Tóm tắt lưu ở: {SUMMARY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

import json
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .testing_config import (
    EVAL_OUTPUT_FILE,
    SUMMARY_OUTPUT_FILE,
)


def load_eval_results(path: str) -> List[Dict[str, Any]]:
    """Load toàn bộ eval_results_v1.jsonl thành list dict."""
    results: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            results.append(obj)
    return results


def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_std(values: List[float]) -> Optional[float]:
    if len(values) <= 1:
        return None
    try:
        return float(statistics.pstdev(values))
    except Exception:
        return None


def safe_median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    try:
        return float(statistics.median(values))
    except Exception:
        return None


def is_empty_or_error_answer(answer: str) -> bool:
    """Heuristic: câu trả lời rỗng hoặc có vẻ là lỗi hệ thống."""
    if not answer or not answer.strip():
        return True

    # Có thể bổ sung pattern lỗi hệ thống nếu backend có format cố định
    error_patterns = [
        "lỗi hệ thống",
        "error",
        "exception",
        "timeout",
        "không thể xử lý yêu cầu",
    ]
    lower = answer.lower()
    return any(pat in lower for pat in error_patterns)


def is_refusal_answer(answer: str) -> bool:
    """
    Heuristic: chatbot từ chối trả lời / ngoài phạm vi.
    Chỉ là xấp xỉ, dùng để thống kê tương đối.
    """
    if not answer or not answer.strip():
        return False

    lower = answer.lower()
    patterns = [
        "em không hỗ trợ chủ đề này",
        "ngoài phạm vi hỗ trợ",
        "em không thể trả lời câu hỏi này",
        "không có quyền trả lời",
        "không thể giúp với yêu cầu này",
    ]
    return any(pat in lower for pat in patterns)


def compute_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    num_questions = len(results)

    # Các mảng phục vụ thống kê
    judge_scores: List[float] = []
    recall5_list: List[float] = []
    recall10_list: List[float] = []

    # Cho các câu có gold
    recall5_with_gold: List[float] = []
    recall10_with_gold: List[float] = []

    num_with_gold = 0
    num_without_gold = 0

    # Đếm quality distribution
    num_score_ge_4 = 0
    num_score_le_2 = 0

    # Answerability / failure
    num_empty_or_error_answers = 0
    num_refusals = 0

    # Breakdown theo intent / difficulty
    by_intent: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "judge_scores": [],
            "recall5": [],
            "recall10": [],
        }
    )
    by_difficulty: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "judge_scores": [],
            "recall5": [],
            "recall10": [],
        }
    )

    # Retrieval–answer relationship
    scores_when_recall10_gt_0: List[float] = []
    scores_when_recall10_eq_0: List[float] = []

    for row in results:
        # Lấy field cơ bản
        judge_score = row.get("judge_score")
        recall_at_5 = row.get("recall_at_5")
        recall_at_10 = row.get("recall_at_10")
        intent = row.get("intent", "unknown")
        difficulty = row.get("difficulty", "unknown")
        answer = row.get("answer", "")
        gold_ids = row.get("gold_context_ids") or []

        # Gold coverage
        if gold_ids:
            num_with_gold += 1
        else:
            num_without_gold += 1

        # Judge score
        if isinstance(judge_score, (int, float)):
            js = float(judge_score)
            judge_scores.append(js)
            if js >= 4.0:
                num_score_ge_4 += 1
            if js <= 2.0:
                num_score_le_2 += 1

            # Breakdown theo intent / difficulty
            by_intent[intent]["count"] += 1
            by_intent[intent]["judge_scores"].append(js)
            by_difficulty[difficulty]["count"] += 1
            by_difficulty[difficulty]["judge_scores"].append(js)

        # Recall@5, Recall@10 (global, bỏ qua None)
        if isinstance(recall_at_5, (int, float)):
            r5 = float(recall_at_5)
            recall5_list.append(r5)
            if gold_ids:
                recall5_with_gold.append(r5)
            # breakdown
            by_intent[intent]["recall5"].append(r5)
            by_difficulty[difficulty]["recall5"].append(r5)

        if isinstance(recall_at_10, (int, float)):
            r10 = float(recall_at_10)
            recall10_list.append(r10)
            if gold_ids:
                recall10_with_gold.append(r10)
            # breakdown
            by_intent[intent]["recall10"].append(r10)
            by_difficulty[difficulty]["recall10"].append(r10)

            # Retrieval–answer relationship
            if isinstance(judge_score, (int, float)):
                if r10 > 0:
                    scores_when_recall10_gt_0.append(float(judge_score))
                elif r10 == 0:
                    scores_when_recall10_eq_0.append(float(judge_score))

        # Answerability / failure
        if is_empty_or_error_answer(answer):
            num_empty_or_error_answers += 1
        if is_refusal_answer(answer):
            num_refusals += 1

    # === Tổng quan ===
    avg_judge_score = safe_mean(judge_scores)
    std_judge_score = safe_std(judge_scores)
    median_judge_score = safe_median(judge_scores)
    min_judge_score = min(judge_scores) if judge_scores else None
    max_judge_score = max(judge_scores) if judge_scores else None

    avg_recall5 = safe_mean(recall5_list)
    avg_recall10 = safe_mean(recall10_list)

    avg_recall5_with_gold = safe_mean(recall5_with_gold)
    avg_recall10_with_gold = safe_mean(recall10_with_gold)

    num_with_judge_score = len(judge_scores) if judge_scores else 0

    pct_judge_score_ge_4 = (
        num_score_ge_4 / num_with_judge_score if num_with_judge_score else None
    )
    pct_judge_score_le_2 = (
        num_score_le_2 / num_with_judge_score if num_with_judge_score else None
    )

    # Answerability percentages (trên tổng số câu)
    pct_empty_or_error_answers = (
        num_empty_or_error_answers / num_questions if num_questions else None
    )
    pct_refusals = num_refusals / num_questions if num_questions else None

    # Breakdown theo intent
    by_intent_summary: Dict[str, Any] = {}
    for intent, agg in by_intent.items():
        scores = agg["judge_scores"]
        r5s = agg["recall5"]
        r10s = agg["recall10"]
        by_intent_summary[intent] = {
            "count": agg["count"],
            "avg_judge_score": safe_mean(scores),
            "avg_recall_at_5": safe_mean(r5s),
            "avg_recall_at_10": safe_mean(r10s),
        }

    # Breakdown theo difficulty
    by_difficulty_summary: Dict[str, Any] = {}
    for diff, agg in by_difficulty.items():
        scores = agg["judge_scores"]
        r5s = agg["recall5"]
        r10s = agg["recall10"]
        by_difficulty_summary[diff] = {
            "count": agg["count"],
            "avg_judge_score": safe_mean(scores),
            "avg_recall_at_5": safe_mean(r5s),
            "avg_recall_at_10": safe_mean(r10s),
        }

    # Retrieval–answer quality relationship
    avg_score_when_recall10_gt_0 = safe_mean(scores_when_recall10_gt_0)
    avg_score_when_recall10_eq_0 = safe_mean(scores_when_recall10_eq_0)

    summary: Dict[str, Any] = {
        "num_questions": num_questions,

        "overall": {
            "avg_judge_score": avg_judge_score,
            "std_judge_score": std_judge_score,
            "median_judge_score": median_judge_score,
            "min_judge_score": min_judge_score,
            "max_judge_score": max_judge_score,
            "avg_recall_at_5": avg_recall5,
            "avg_recall_at_10": avg_recall10,
        },

        "quality_distribution": {
            "num_with_judge_score": num_with_judge_score,
            "pct_judge_score_ge_4": pct_judge_score_ge_4,
            "pct_judge_score_le_2": pct_judge_score_le_2,
        },

        "gold_coverage": {
            "num_with_gold": num_with_gold,
            "num_without_gold": num_without_gold,
            "avg_recall_at_5_with_gold": avg_recall5_with_gold,
            "avg_recall_at_10_with_gold": avg_recall10_with_gold,
        },

        "by_intent": by_intent_summary,
        "by_difficulty": by_difficulty_summary,

        "answerability": {
            "pct_empty_or_error_answers": pct_empty_or_error_answers,
            "pct_refusals": pct_refusals,
        },

        "retrieval_vs_answer_quality": {
            "avg_judge_score_when_recall_at_10_gt_0": avg_score_when_recall10_gt_0,
            "avg_judge_score_when_recall_at_10_eq_0": avg_score_when_recall10_eq_0,
        },
    }

    return summary


def main() -> None:
    print(f"Đang load kết quả từ {EVAL_OUTPUT_FILE} ...")
    results = load_eval_results(EVAL_OUTPUT_FILE)
    print(f"Đã load {len(results)} dòng eval.")

    summary = compute_summary(results)

    with open(SUMMARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Đã ghi summary vào: {SUMMARY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

# generate_questions.py

import json
import os
import time
from typing import List, Dict, Any

from google import genai
from .config_questions import GEMINI_API_KEY

from .config_questions import (
    TOTAL_QUESTIONS,
    INTENT_DISTRIBUTION,
    DIFFICULTY_DISTRIBUTION,
    SPECIFICITY_DISTRIBUTION,
    JOBS_PER_CHUNK,
)

# Đường dẫn file jobs (tính từ root project khi chạy: python -m create_question.generate_questions)
JOBS_FILE = "create_question/jobs_for_chat_v2.jsonl"
OUTPUT_FILE = "testing_chatbot/questions_dataset_v2.jsonl"

# Model Gemini sử dụng (không phải 2.0 flash)
MODEL_NAME = "gemini-2.5-flash-lite"

# Giới hạn token/phút 
TOKENS_PER_MIN_LIMIT = 125_000
EST_TOKENS_PER_JOB = 230
EST_TOKENS_PER_QUESTION = 220
BASE_PROMPT_TOKENS_EST = 2500

def load_all_jobs(path: str) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(json.loads(line))
    return jobs


def split_jobs_into_chunks(jobs: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    """
    Chia jobs thành các chunk với số lượng tối đa chunk_size job mỗi chunk.
    """
    chunks: List[List[Dict[str, Any]]] = []
    for i in range(0, len(jobs), chunk_size):
        chunks.append(jobs[i: i + chunk_size])
    return chunks


def compute_batch_sizes(total_questions: int, num_batches: int) -> List[int]:
    """
    Chia total_questions đều cho num_batches.
    Ví dụ: 1000 câu / 4 batch -> mỗi batch 250 câu.
           Nếu không chia hết thì vài batch đầu sẽ +1.
    """
    if num_batches <= 0:
        raise ValueError("num_batches phải > 0")

    base = total_questions // num_batches
    remainder = total_questions % num_batches
    sizes: List[int] = []
    for i in range(num_batches):
        size = base + (1 if i < remainder else 0)
        sizes.append(size)
    return sizes


def compute_intent_counts_for_batch(batch_size: int) -> Dict[str, int]:
    """
    Tính số câu mỗi intent cho 1 batch dựa trên INTENT_DISTRIBUTION.
    """
    counts: Dict[str, int] = {}
    for intent, ratio in INTENT_DISTRIBUTION.items():
        counts[intent] = round(ratio * batch_size)

    # Điều chỉnh cho đủ batch_size (cộng/trừ vào search_jobs)
    diff = batch_size - sum(counts.values())
    counts["search_jobs"] = counts.get("search_jobs", 0) + diff
    return counts


def simplify_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rút gọn job để giảm token gửi lên model:
    Chỉ giữ các field quan trọng: id, title, salary, locations, experience, company.
    Bỏ các đoạn mô tả HTML dài, detail_sections, v.v.
    """
    simple: Dict[str, Any] = {}

    # id là bắt buộc
    if "id" in job:
        simple["id"] = job["id"]

    # tiêu đề
    if "title" in job:
        simple["title"] = job["title"]

    # công ty (tùy tên field)
    if "company" in job:
        simple["company"] = job["company"]
    elif "company_name" in job:
        simple["company"] = job["company_name"]

    # lương
    if "salary" in job:
        simple["salary"] = job["salary"]

    # địa điểm
    if "locations" in job:
        simple["locations"] = job["locations"]

    # kinh nghiệm
    if "experience" in job:
        simple["experience"] = job["experience"]

    return simple


def build_user_prompt(
    batch_index: int,
    batch_size: int,
    intent_counts: Dict[str, int],
    jobs_subset: List[Dict[str, Any]],
) -> str:
    """
    Tạo prompt tiếng Việt gửi cho model.
    Yêu cầu model trả về MỘT mảng JSON gồm batch_size object câu hỏi.
    """
    lines: List[str] = []

    lines.append(
        f"Bạn là LLM hỗ trợ tạo bộ câu hỏi kiểm thử cho trợ lý tuyển dụng JobFinder.\n"
        f"Batch hiện tại: {batch_index + 1}.\n"
        f"Hãy sinh ra TỔNG CỘNG {batch_size} câu hỏi tuyển dụng theo schema cho trước.\n"
        f"Câu trả lời phải là MỘT MẢNG JSON (JSON array) gồm đúng {batch_size} object.\n"
        f"KHÔNG thêm bất kỳ text nào ngoài mảng JSON."
    )

    # Thông tin về intent
    lines.append("\n=== PHÂN BỔ INTENT TRONG BATCH NÀY ===")
    for intent, count in intent_counts.items():
        lines.append(f"- {intent}: {count} câu")

    # Thông tin phân bổ difficulty / specificity (chỉ để model hiểu, không cần exact)
    lines.append("\n=== PHÂN BỔ ĐỘ KHÓ (TRÊN TOÀN DATASET) ===")
    for diff, ratio in DIFFICULTY_DISTRIBUTION.items():
        lines.append(f"- {diff}: {int(ratio * 100)}%")

    lines.append("\n=== PHÂN BỔ SPECIFICITY THEO INTENT (TRÊN TOÀN DATASET) ===")
    for intent, spec_cfg in SPECIFICITY_DISTRIBUTION.items():
        lines.append(
            f"- {intent}: specific ~ {int(spec_cfg['specific'] * 100)}%, "
            f"non_specific ~ {int(spec_cfg['non_specific'] * 100)}%"
        )

    # Ví dụ schema
    schema_example = {
        "id": "Q_TEMP_001",
        "intent": "search_jobs",
        "difficulty": "easy",
        "specificity": "non_specific",
        "question_text": "Em muốn tìm việc nhân viên kinh doanh tại Hà Nội, lương từ 12 triệu trở lên.",
        "expected_behavior": (
            "Chatbot phân tích ngành (kinh doanh), địa điểm (Hà Nội), mức lương tối thiểu, "
            "sau đó gợi ý 3–5 job phù hợp từ context, trình bày rõ ràng, không bịa thêm job."
        ),
        "gold_context_ids": [12345, 67890],
        "anti_preference": None,
    }

    lines.append("\n=== SCHEMA CHO MỖI CÂU HỎI (MỖI OBJECT TRONG ARRAY) ===")
    lines.append("Mỗi object phải có các field sau:")
    lines.append(json.dumps(schema_example, ensure_ascii=False, indent=2))

    lines.append(
        "\nGiải thích ngắn gọn:\n"
        "- intent: một trong 4 giá trị: 'search_jobs', 'ask_detail', 'compare_jobs', 'other'.\n"
        "  + search_jobs: người dùng muốn TÌM/GỢI Ý job.\n"
        "  + ask_detail: hỏi chi tiết 1 job cụ thể trong danh sách jobs bên dưới.\n"
        "  + compare_jobs: so sánh 2–3 job cụ thể trong danh sách jobs.\n"
        "  + other: chào hỏi, jailbreak, ngoài phạm vi, có liên quan đến tuyển dụng hoặc không liên quan đến tuyển dụng như: hỏi thời tiết, giá cổ phiếu,...\n\n"
        "- difficulty: 'easy' | 'medium' | 'hard'.\n"
        "- specificity:\n"
        "  + 'specific': câu hỏi nhắm vào job/company cụ thể (dựa trên jobs_subset).\n"
        "  + 'non_specific': câu hỏi chung chung.\n\n"
        "- question_text: câu hỏi tiếng Việt tự nhiên, giống người đi tìm việc.\n"
        "- expected_behavior: 1–3 câu mô tả chatbot nên làm gì, THỰC TẾ & bám RAG, nếu có mô tả về job thì mô tả thêm tên Job (kèm ID nếu có).\n"
        "- gold_context_ids: danh sách id (số nguyên) của job trong jobs_subset phù hợp với câu hỏi.\n"
        "  + search_jobs: 0–5 job phù hợp.\n"
        "  + ask_detail: thường 1 job id.\n"
        "  + compare_jobs: 2–3 job id.\n"
        "  + other: thường [].\n"
        "- anti_preference:\n"
        "  + Bình thường: null.\n"
        "  + Với câu jailbreak / ngoài phạm vi: ghi rõ không được bịa lương, không được làm theo yêu cầu phá luật, v.v."
    )

    lines.append(
        "\n=== YÊU CẦU ĐẦU RA RẤT QUAN TRỌNG ===\n"
        f"- Chỉ in ra MỘT mảng JSON (JSON array) chứa ĐÚNG {batch_size} object.\n"
        "- Không in text thừa trước hoặc sau mảng JSON.\n"
        "- Mỗi object trong array tuân thủ schema trên.\n"
        "- Phải đảm bảo số intent đúng như intent_counts đã cho (xấp xỉ nếu có rounding)."
    )

    # Rút gọn jobs_subset trước khi nhúng vào prompt
    simple_jobs = [simplify_job(j) for j in jobs_subset]

    lines.append("\n=== DANH SÁCH JOBS_SUBSET (RÚT GỌN) ===")
    lines.append(
        "Dưới đây là danh sách job rút gọn (id, title, company, salary, locations, experience). "
        "Hãy dựa vào các job này để đặt câu hỏi và điền gold_context_ids tương ứng:"
    )
    lines.append(json.dumps(simple_jobs, ensure_ascii=False, indent=2))

    return "\n".join(lines)


def call_gemini_api(client: genai.Client, prompt: str) -> str:
    """
    Gọi Gemini API, trả về raw text (mảng JSON hoặc text chứa JSON).
    """
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )
    return response.text or ""


def extract_json_array(raw_text: str) -> str:
    """
    Tìm đoạn JSON array đầu tiên trong raw_text (từ '[' đến ']' cuối cùng).
    Dùng khi model trả về thêm giải thích ngoài JSON.
    """
    start = raw_text.find("[")
    end = raw_text.rfind("]")

    if start == -1 or end == -1 or end < start:
        raise ValueError("Không tìm thấy JSON array trong output của model.")

    return raw_text[start: end + 1]


def ensure_list_of_questions(raw_text: str, expected_count: int) -> List[Dict[str, Any]]:
    """
    Parse raw_text (mảng JSON), đảm bảo là list và có đúng expected_count phần tử.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Model trả về output rỗng, không thể parse JSON.")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Thử trích xuất JSON array từ trong text
        array_str = extract_json_array(raw_text)
        data = json.loads(array_str)

    if not isinstance(data, list):
        raise ValueError("Model không trả về JSON array như yêu cầu.")

    if len(data) != expected_count:
        print(
            f"⚠️ Cảnh báo: số lượng object trong array = {len(data)}, "
            f"không khớp expected_count = {expected_count}."
        )

    questions: List[Dict[str, Any]] = [q for q in data if isinstance(q, dict)]
    return questions


def assign_question_ids(questions: List[Dict[str, Any]], start_index: int) -> int:
    """
    Gán id dạng 0001, 0002, ...
    Trả về index tiếp theo sau khi gán.
    """
    idx = start_index
    for q in questions:
        q["id"] = f"Q{idx:06d}"
        idx += 1
    return idx


def save_questions_jsonl(questions: List[Dict[str, Any]], path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

# ước lượng số token
def estimate_tokens_for_batch(jobs_subset: List[Dict[str, Any]], batch_size: int) -> int:
    num_jobs = len(jobs_subset)
    est_input_tokens = BASE_PROMPT_TOKENS_EST + EST_TOKENS_PER_JOB * num_jobs
    est_output_tokens = EST_TOKENS_PER_QUESTION * batch_size
    return est_input_tokens + est_output_tokens


def main() -> None:
    if not os.path.exists(JOBS_FILE):
        raise FileNotFoundError(f"Không tìm thấy file {JOBS_FILE} trong thư mục hiện tại.")

    jobs = load_all_jobs(JOBS_FILE)
    print(f"Đã load {len(jobs)} job từ {JOBS_FILE}")

    if len(jobs) == 0:
        raise ValueError("File jobs_for_chatgpt.jsonl rỗng.")

    chunks = split_jobs_into_chunks(jobs, JOBS_PER_CHUNK)
    num_batches = len(chunks)
    batch_sizes = compute_batch_sizes(TOTAL_QUESTIONS, num_batches)

    print(f"Sẽ sinh {TOTAL_QUESTIONS} câu hỏi trong {num_batches} batch.")
    print(f"Các batch size dự kiến: {batch_sizes}")

    # Tạo client Gemini dùng key từ env_config
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Xóa file output cũ nếu có
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    current_q_index = 1

    for batch_idx, (jobs_subset, batch_size) in enumerate(zip(chunks, batch_sizes)):
        print(f"\n=== BATCH {batch_idx + 1}/{num_batches} ===")
        print(f"- Số job trong batch: {len(jobs_subset)}")
        print(f"- Số câu hỏi cần sinh: {batch_size}")

        intent_counts = compute_intent_counts_for_batch(batch_size)
        print(f"- Intent counts: {intent_counts}")

        user_prompt = build_user_prompt(
            batch_index=batch_idx,
            batch_size=batch_size,
            intent_counts=intent_counts,
            jobs_subset=jobs_subset,
        )

        # ƯỚC LƯỢNG TOKEN + SLEEP ĐỂ TRÁNH VƯỢT TPM FREE-TIER
        est_tokens = estimate_tokens_for_batch(jobs_subset, batch_size)
        sleep_seconds = est_tokens * 60.0 / TOKENS_PER_MIN_LIMIT -3

        # In log cho dễ debug
        print(f"- Ước lượng ~{est_tokens} tokens cho batch này.")
        print(f"- Tạm dừng khoảng {sleep_seconds:.1f} giây để tránh vượt quota TPM...")

        time.sleep(sleep_seconds)

        raw_output = call_gemini_api(client, user_prompt)

        try:
            questions = ensure_list_of_questions(raw_output, batch_size)
        except Exception as e:
            print("❌ Lỗi parse JSON từ model:", e)
            debug_file = f"debug_batch_{batch_idx + 1}.txt"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(raw_output or "")
            print(f"Đã ghi raw output vào {debug_file} để debug.")
            raise

        current_q_index = assign_question_ids(questions, current_q_index)
        save_questions_jsonl(questions, OUTPUT_FILE)

        print(f"- Đã ghi {len(questions)} câu hỏi vào {OUTPUT_FILE}")

    print(f"\n Hoàn thành. Tổng số câu hỏi (theo id) đã sinh: {current_q_index - 1}")
    print(f"File output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

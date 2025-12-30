# config_questions.py

TOTAL_QUESTIONS = 300  # Tổng số câu hỏi muốn sinh

# Tỉ lệ intent: search_jobs 40%, ask_detail 30%, compare_jobs 15%, other 15%
INTENT_DISTRIBUTION = {
    "search_jobs": 0.40,
    "ask_detail": 0.30,
    "compare_jobs": 0.15,
    "other": 0.15,
}

# Phân bổ độ khó (global, để ghi vào hướng dẫn cho LLM)
DIFFICULTY_DISTRIBUTION = {
    "easy": 0.40,
    "medium": 0.40,
    "hard": 0.20,
}

# Phân bổ specificity theo intent (cũng chỉ dùng để ghi vào prompt)
SPECIFICITY_DISTRIBUTION = {
    "search_jobs": {"specific": 0.20, "non_specific": 0.80},
    "ask_detail": {"specific": 0.70, "non_specific": 0.30},
    "compare_jobs": {"specific": 1.00, "non_specific": 0.00},
    "other": {"specific": 0.25, "non_specific": 0.75},
}

# Kích thước 1 "chunk" job làm context cho mỗi batch
# Ví dụ: 1000 job / 50 = 20 batch; 1000 câu hỏi / 20 batch = ~50 câu/batch
JOBS_PER_CHUNK = 100

OPENAI_API_KEY = 
GEMINI_API_KEY = 
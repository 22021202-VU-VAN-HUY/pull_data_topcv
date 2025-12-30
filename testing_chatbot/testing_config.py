# testing_config.py
QUESTIONS_FILE = "testing_chatbot/questions_dataset_v2.jsonl"

# File ghi kết quả chấm 
EVAL_OUTPUT_FILE = "testing_chatbot/eval_results_v2.jsonl"

# (Tuỳ chọn) File tóm tắt kết quả 
SUMMARY_OUTPUT_FILE = "testing_chatbot/eval_summary_v2.json"

JUDGE_MODEL_NAME = "gemini-2.5-pro"

# bật chấm bằng LLM-as-judge
ENABLE_LLM_JUDGE = True

# Số câu tối đa để chấm (None = chấm hết)
MAX_QUESTIONS = None 
SLEEP_BETWEEN_QUESTIONS = 1.5 # với 1.2 thì có 3 time limit token

GEMINI_API_KEY = 
# question_schema.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TestQuestion:
    id: str
    intent: str
    difficulty: str
    specificity: str
    question_text: str
    expected_behavior: str
    gold_context_ids: List[int]
    anti_preference: Optional[str] = None

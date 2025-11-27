# app/api/chat.py

from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.api.rag.chat_logic import chat_with_rag

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Endpoint chính cho chatbot.

    Body (JSON) dự kiến:
    {
      "message": "câu hỏi hiện tại",
      "history": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ]
    }
    """
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []

    result = chat_with_rag(
        user_message=message,
        history=history,
        top_k=5,
    )

    return jsonify(result), 200

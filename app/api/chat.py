# app/api/chat.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging

from flask import Blueprint, request, jsonify

from app.api.rag.chat_logic import handle_chat_turn

logger = logging.getLogger(__name__)

bp = Blueprint("chat", __name__, url_prefix="/api")


@bp.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Endpoint chính cho chatbot RAG.

    Request JSON:
      {
        "message": "...",              # bắt buộc
        "history": [ {role, content} ],# optional
        "current_job_id": 123          # optional
      }

    Response JSON:
      {
        "answer": "...",
        "history": [...],
        "jobs": [ {...}, ... ]
      }
    """
    data = request.get_json(silent=True) or {}

    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        result = handle_chat_turn(data)
    except Exception as e:
        logger.exception("Chat error: %s", e)
        return jsonify({"error": "internal_error"}), 500

    return jsonify(result)

# app/api/server.py
import os
from flask import Flask

from app.api.jobs import jobs_bp
from app.api.auth import auth_bp
from app.api.chat import chat_bp


def create_app() -> Flask:
    # __file__ = .../app/api/server.py
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    templates_dir = os.path.join(base_dir, "web", "templates")
    static_dir = os.path.join(base_dir, "web", "static")

    app = Flask(
        __name__,
        template_folder=templates_dir,
        static_folder=static_dir,
    )

    # Secret key cho session (sẽ đọc từ .env)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")

    # Đăng ký blueprint
    app.register_blueprint(jobs_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(chat_bp)

    return app

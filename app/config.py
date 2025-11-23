import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)


class Settings:
    # DB
    POSTGRES_USER = os.getenv("POSTGRES_USER", "topcv_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "topcv_password")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "topcv_db")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5440"))

    # Crawler
    CRAWL_SLEEP_SECONDS = float(os.getenv("CRAWL_SLEEP_SECONDS", "3"))
    CRAWL_MAX_RETRY = int(os.getenv("CRAWL_MAX_RETRY", "3"))
    CRAWL_MAX_PAGES = int(os.getenv("CRAWL_MAX_PAGES", "1"))
    CRAWL_USE_PLAYWRIGHT = os.getenv("CRAWL_USE_PLAYWRIGHT", "false").lower() == "true"

    TOPCV_LISTING_URL = os.getenv("TOPCV_LISTING_URL", "https://www.topcv.vn/viec-lam")
    TOPCV_BASE_URL = os.getenv("TOPCV_BASE_URL", "https://www.topcv.vn")


settings = Settings()

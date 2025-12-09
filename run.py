import os
from dotenv import load_dotenv

from app.api import create_app

# Load .env
load_dotenv()

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("WEB_PORT", "5000"))
    app.run(debug=True, host="127.0.0.1", port=port)

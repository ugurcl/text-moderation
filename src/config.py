import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RATE_LIMIT = os.getenv("RATE_LIMIT", "60/minute")

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(BASE_DIR / "models")))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "data" / "predictions.db")))

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

API_KEY = os.getenv("API_KEY", "")

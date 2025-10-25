import os
from dotenv import load_dotenv

load_dotenv()

DB = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "sslmode": "disable",
    "connect_timeout": 5,
}

LLM_API_KEY = os.getenv("XAI_API_KEY")
LLM_MODEL = os.getenv("GROK_MODEL")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.x.ai/v1")

ALL_PROXY    = os.getenv("ALL_PROXY") or None
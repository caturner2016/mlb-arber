import os
from dotenv import load_dotenv

load_dotenv()

KALSHI_EMAIL = os.getenv("KALSHI_EMAIL", "")
KALSHI_PASSWORD = os.getenv("KALSHI_PASSWORD", "")
KALSHI_BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"

DAILY_STOP_LOSS = float(os.getenv("DAILY_STOP_LOSS", "20.0"))
MAX_BET = float(os.getenv("MAX_BET", "1.0"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.05"))
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.25"))
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "30"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(os.path.dirname(__file__), "bot_state.db")
LOG_PATH = os.path.join(os.path.dirname(__file__), "bot.log")

MIN_HOURS_TO_CLOSE = float(os.getenv("MIN_HOURS_TO_CLOSE", "2.0"))

ELO_START = 1500.0
ELO_K = 32.0
SURFACE_BLEND = 0.6  # surface-specific elo weight vs overall
MIN_CONTRACTS = 1
DATA_YEARS = list(range(2015, 2026))

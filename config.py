import os
import yaml

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path) as _f:
    _cfg = yaml.safe_load(_f)

KALSHI_KEY_ID        = _cfg["kalshi_key_id"]
KALSHI_PRIVATE_KEY_PATH = os.path.join(os.path.dirname(__file__), _cfg["kalshi_private_key_path"])

DAILY_STOP_LOSS      = float(_cfg["daily_stop_loss"])
MAX_BET              = float(_cfg["max_bet"])
MIN_EDGE             = float(_cfg["min_edge"])
KELLY_FRACTION       = float(_cfg["kelly_fraction"])
CHECK_INTERVAL       = int(_cfg["check_interval"])
MIN_HOURS_TO_CLOSE   = float(_cfg["min_hours_to_close"])

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
DB_PATH    = os.path.join(os.path.dirname(__file__), "bot_state.db")
LOG_PATH   = os.path.join(os.path.dirname(__file__), "bot.log")

ELO_START     = 1500.0
ELO_K         = 32.0
SURFACE_BLEND = 0.6
MIN_CONTRACTS = 1

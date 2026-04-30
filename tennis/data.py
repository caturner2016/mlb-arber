import os
import logging
import requests
import pandas as pd
from config import DATA_DIR

log = logging.getLogger(__name__)

# Sackmann: ATP + WTA up to 2024
ATP_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
WTA_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"
SACKMANN_YEARS = list(range(2015, 2025))

# TML-Database: ATP only, updated daily, 2025+
TML_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master/{year}.csv"
TML_YEARS = [2025, 2026]

COLS = ["tourney_date", "surface", "winner_name", "loser_name"]


def _fetch(url: str, cache_name: str, force: bool = False) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, cache_name)
    if os.path.exists(path) and not force:
        try:
            return pd.read_csv(path, usecols=COLS, low_memory=False)
        except Exception:
            pass
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        df = pd.read_csv(path, usecols=COLS, low_memory=False)
        log.info("Downloaded %s (%d rows)", cache_name, len(df))
        return df
    except Exception as e:
        log.warning("Could not fetch %s: %s", cache_name, e)
        return None


def load_matches() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []

    for year in SACKMANN_YEARS:
        for url, name in ((ATP_URL, f"atp_{year}.csv"), (WTA_URL, f"wta_{year}.csv")):
            df = _fetch(url.format(year=year), name)
            if df is not None:
                frames.append(df)

    for year in TML_YEARS:
        df = _fetch(TML_URL.format(year=year), f"tml_{year}.csv")
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No tennis data loaded — check network connection")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["winner_name", "loser_name", "surface"])
    combined["surface"] = combined["surface"].str.lower().str.strip()
    combined["surface"] = combined["surface"].replace("carpet", "hard")
    combined = combined.sort_values("tourney_date").reset_index(drop=True)
    log.info("Loaded %d total matches", len(combined))
    return combined


def refresh_current_year():
    """Re-download current year TML file to pick up today's results."""
    import datetime
    year = datetime.date.today().year
    _fetch(TML_URL.format(year=year), f"tml_{year}.csv", force=True)

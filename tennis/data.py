import os
import logging
import requests
import pandas as pd
from config import DATA_DIR, DATA_YEARS

log = logging.getLogger(__name__)

ATP_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_{year}.csv"
WTA_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_matches_{year}.csv"

COLS = ["tourney_date", "surface", "winner_name", "loser_name"]


def _fetch_year(url: str, year: int) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, os.path.basename(url.format(year=year)))
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, usecols=COLS, low_memory=False)
            return df
        except Exception:
            pass
    try:
        r = requests.get(url.format(year=year), timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        df = pd.read_csv(path, usecols=COLS, low_memory=False)
        log.info("Downloaded %s", os.path.basename(path))
        return df
    except Exception as e:
        log.warning("Could not fetch %s %d: %s", url.split("/")[-1], year, e)
        return None


def load_matches() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []
    for year in DATA_YEARS:
        for url in (ATP_URL, WTA_URL):
            df = _fetch_year(url, year)
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
    """Re-download the current year's files to pick up new results."""
    import datetime
    current = datetime.date.today().year
    for url in (ATP_URL, WTA_URL):
        path = os.path.join(DATA_DIR, os.path.basename(url.format(year=current)))
        if os.path.exists(path):
            os.remove(path)
        _fetch_year(url, current)

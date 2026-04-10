"""
MLB data fetcher and local cache.

Pulls from MLB Stats API — no auth required.

What we collect:
  - Batter game logs (per player per game: H, AB, HR, SO, opponent team, venue, is_home)
  - Pitcher game logs (per pitcher per game: IP, H, ER, BB, SO, HR, opponent team)
  - Player handedness (bats: L/R/S, throws: L/R)
  - Today's schedule with probable starters

Cache: model/cache/ as parquet files.
Refresh: python -m model.data --refresh
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

log = logging.getLogger(__name__)

MLB_API   = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Seconds between requests to be polite to the MLB API
_REQUEST_DELAY = 0.15


def _get(path: str, params: dict | None = None) -> dict:
    url = MLB_API + path
    time.sleep(_REQUEST_DELAY)
    r = requests.get(url, params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()


# ── Player IDs ─────────────────────────────────────────────────────────────────

def get_all_player_ids(season: int) -> pd.DataFrame:
    """All active MLB players for a season. Returns DataFrame: player_id, full_name, position."""
    cache_path = CACHE_DIR / f"players_{season}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    log.info(f"Fetching player list for {season}…")
    data = _get("/sports/1/players", {"season": season, "gameType": "R"})

    rows = []
    for p in data.get("people", []):
        rows.append({
            "player_id":  p.get("id"),
            "full_name":  p.get("fullName", ""),
            "bat_side":   p.get("batSide", {}).get("code", "R"),   # L / R / S
            "pitch_hand": p.get("pitchHand", {}).get("code", "R"), # L / R
            "position":   p.get("primaryPosition", {}).get("abbreviation", ""),
        })

    df = pd.DataFrame(rows)
    df.to_parquet(cache_path, index=False)
    return df


def get_player_hand(player_id: int, season: int = 2025) -> dict:
    """Returns {'bat': 'L'/'R'/'S', 'pitch': 'L'/'R'} for a single player."""
    df = get_all_player_ids(season)
    row = df[df["player_id"] == player_id]
    if row.empty:
        return {"bat": "R", "pitch": "R"}
    return {"bat": row.iloc[0]["bat_side"], "pitch": row.iloc[0]["pitch_hand"]}


# ── Game logs ──────────────────────────────────────────────────────────────────

def _fetch_game_log(player_id: int, season: int, group: str) -> list[dict]:
    """Fetch raw game log from MLB Stats API."""
    try:
        data = _get(
            f"/people/{player_id}/stats",
            {"stats": "gameLog", "season": season, "group": group},
        )
    except Exception as e:
        log.debug(f"Game log {player_id} {season} {group}: {e}")
        return []

    rows = []
    for stat_block in data.get("stats", []):
        for split in stat_block.get("splits", []):
            stat = split.get("stat", {})
            row  = {
                "player_id":       player_id,
                "date":            split.get("date", ""),
                "game_pk":         split.get("game", {}).get("gamePk", 0),
                "opponent_team_id": split.get("opponent", {}).get("id", 0),
                "team_id":         split.get("team", {}).get("id", 0),
                "is_home":         int(split.get("isHome", False)),
                "venue_id":        split.get("venue", {}).get("id", 0) or 0,
            }
            row.update(stat)
            rows.append(row)
    return rows


def get_batter_game_logs(player_ids: list[int], seasons: list[int]) -> pd.DataFrame:
    """Batting game logs for a list of player IDs and seasons."""
    cache_key = f"batting_logs_{'_'.join(map(str, seasons))}.parquet"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached_ids = set(df["player_id"].unique())
        new_ids    = [pid for pid in player_ids if pid not in cached_ids]
        if not new_ids:
            return df
        log.info(f"Fetching {len(new_ids)} new batters…")
    else:
        df       = pd.DataFrame()
        new_ids  = player_ids

    all_rows: list[dict] = []
    for i, pid in enumerate(new_ids):
        if i % 50 == 0:
            log.info(f"  Batter logs: {i}/{len(new_ids)}")
        for season in seasons:
            all_rows.extend(_fetch_game_log(pid, season, "hitting"))

    if all_rows:
        new_df = pd.DataFrame(all_rows)
        new_df["date"] = pd.to_datetime(new_df["date"])
        # Standardize numeric columns
        for col in ["atBats", "hits", "homeRuns", "strikeOuts", "baseOnBalls",
                    "doubles", "triples", "rbi", "runs"]:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0).astype(int)

        df = pd.concat([df, new_df], ignore_index=True) if not df.empty else new_df
        df.to_parquet(cache_path, index=False)

    return df


def get_pitcher_game_logs(player_ids: list[int], seasons: list[int]) -> pd.DataFrame:
    """Pitching game logs for a list of pitcher IDs and seasons."""
    cache_key = f"pitching_logs_{'_'.join(map(str, seasons))}.parquet"
    cache_path = CACHE_DIR / cache_key

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        cached_ids = set(df["player_id"].unique())
        new_ids    = [pid for pid in player_ids if pid not in cached_ids]
        if not new_ids:
            return df
        log.info(f"Fetching {len(new_ids)} new pitchers…")
    else:
        df      = pd.DataFrame()
        new_ids = player_ids

    all_rows: list[dict] = []
    for i, pid in enumerate(new_ids):
        if i % 30 == 0:
            log.info(f"  Pitcher logs: {i}/{len(new_ids)}")
        for season in seasons:
            all_rows.extend(_fetch_game_log(pid, season, "pitching"))

    if all_rows:
        new_df = pd.DataFrame(all_rows)
        new_df["date"] = pd.to_datetime(new_df["date"])
        for col in ["strikeOuts", "inningsPitched", "hits", "earnedRuns",
                    "baseOnBalls", "homeRuns", "outs", "numberOfPitches"]:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0)

        df = pd.concat([df, new_df], ignore_index=True) if not df.empty else new_df
        df.to_parquet(cache_path, index=False)

    return df


# ── Today's schedule with probable pitchers ────────────────────────────────────

def get_todays_matchups(target_date: Optional[date] = None) -> list[dict]:
    """
    Returns today's games with probable starting pitchers.
    Each entry: {game_pk, home_team_id, away_team_id, venue_id, venue_name,
                 home_pitcher_id, away_pitcher_id, game_time}
    """
    d = target_date or date.today()
    data = _get("/schedule", {
        "sportId": 1,
        "date":    d.strftime("%Y-%m-%d"),
        "hydrate": "probablePitcher,team,venue",
    })

    matchups = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            teams = g.get("teams", {})
            home  = teams.get("home", {})
            away  = teams.get("away", {})
            venue = g.get("venue", {})

            matchups.append({
                "game_pk":         g.get("gamePk", 0),
                "game_time":       g.get("gameDate", ""),
                "status":          g.get("status", {}).get("abstractGameState", ""),
                "home_team_id":    home.get("team", {}).get("id", 0),
                "home_team_name":  home.get("team", {}).get("name", ""),
                "away_team_id":    away.get("team", {}).get("id", 0),
                "away_team_name":  away.get("team", {}).get("name", ""),
                "venue_id":        venue.get("id", 0),
                "venue_name":      venue.get("name", ""),
                "home_pitcher_id": home.get("probablePitcher", {}).get("id"),
                "away_pitcher_id": away.get("probablePitcher", {}).get("id"),
            })

    return matchups


def get_todays_lineups(target_date: Optional[date] = None) -> dict[int, list[int]]:
    """
    Returns {game_pk: [batter_id, ...]} for each game today.
    Lineups are sometimes not available until closer to game time.
    """
    d = target_date or date.today()
    data = _get("/schedule", {
        "sportId": 1,
        "date":    d.strftime("%Y-%m-%d"),
        "hydrate": "lineups",
    })

    result: dict[int, list[int]] = {}
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            pk      = g.get("gamePk", 0)
            lineups = g.get("lineups", {})
            batters: list[int] = []
            for side in ("homePlayers", "awayPlayers"):
                for p in lineups.get(side, []):
                    if p.get("allPositions", [{}])[0].get("abbreviation", "") not in ("P",):
                        batters.append(p.get("id", 0))
            if batters:
                result[pk] = batters
    return result


# ── Bulk data pull for training ────────────────────────────────────────────────

def pull_training_data(seasons: list[int] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull all batter and pitcher game logs for training seasons.
    Returns (batter_df, pitcher_df).
    """
    if seasons is None:
        seasons = [2022, 2023, 2024]

    log.info(f"Pulling training data for seasons {seasons}…")

    # Get all players for these seasons
    all_players   = pd.concat([get_all_player_ids(s) for s in seasons]).drop_duplicates("player_id")
    all_ids       = all_players["player_id"].tolist()

    # Separate pitchers vs hitters
    pitcher_pos   = {"P", "SP", "RP"}
    pitcher_ids   = all_players[all_players["position"].isin(pitcher_pos)]["player_id"].tolist()
    batter_ids    = all_players[~all_players["position"].isin(pitcher_pos)]["player_id"].tolist()

    log.info(f"  {len(batter_ids)} batters  {len(pitcher_ids)} pitchers")

    batter_df  = get_batter_game_logs(batter_ids, seasons)
    pitcher_df = get_pitcher_game_logs(pitcher_ids, seasons)

    log.info(f"  Batter rows: {len(batter_df):,}  Pitcher rows: {len(pitcher_df):,}")
    return batter_df, pitcher_df


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Delete cache and re-pull")
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024])
    args = parser.parse_args()

    if args.refresh:
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
        print("Cache cleared.")

    batters, pitchers = pull_training_data(args.seasons)
    print(f"\nBatter game logs:  {len(batters):,} rows")
    print(f"Pitcher game logs: {len(pitchers):,} rows")
    print(f"\nSample batter columns: {list(batters.columns[:10])}")

    # Today's matchups
    matchups = get_todays_matchups()
    print(f"\nToday's games: {len(matchups)}")
    for m in matchups:
        hp = m["home_pitcher_id"] or "TBD"
        ap = m["away_pitcher_id"] or "TBD"
        print(f"  {m['away_team_name']:25s} @ {m['home_team_name']:25s}  "
              f"{m['venue_name']:25s}  SP: {ap} vs {hp}")

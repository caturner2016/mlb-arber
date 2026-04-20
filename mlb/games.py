"""Fetch today's MLB games from the MLB Stats API (no auth required)."""

from __future__ import annotations

import requests
from dataclasses import dataclass
from datetime import date


MLB_API = "https://statsapi.mlb.com/api/v1"


@dataclass
class Game:
    game_pk: int
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    status: str  # "Preview", "Live", "Final", etc.


def get_todays_games(target_date: date | None = None) -> list[Game]:
    """Return all MLB games scheduled for today (or target_date)."""
    d = target_date or date.today()
    url = f"{MLB_API}/schedule"
    params = {
        "sportId": 1,
        "date": d.strftime("%Y-%m-%d"),
        "hydrate": "team,game(content(summary))",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            teams = g.get("teams", {})
            home = teams.get("home", {}).get("team", {})
            away = teams.get("away", {}).get("team", {})
            games.append(Game(
                game_pk=g["gamePk"],
                home_team=home.get("name", ""),
                away_team=away.get("name", ""),
                home_team_id=home.get("id", 0),
                away_team_id=away.get("id", 0),
                status=g.get("status", {}).get("abstractGameState", ""),
            ))
    return games


def get_live_games(target_date: date | None = None) -> list[Game]:
    """Return only games currently in progress."""
    return [g for g in get_todays_games(target_date) if g.status == "Live"]


if __name__ == "__main__":
    games = get_todays_games()
    print(f"Games today: {len(games)}")
    for g in games:
        print(f"  [{g.game_pk}] {g.away_team} @ {g.home_team}  ({g.status})")

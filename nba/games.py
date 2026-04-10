"""
NBA live games — today's schedule from the NBA CDN.

Fast, no auth required. Updates in real-time during games.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests


NBA_SCOREBOARD = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"


@dataclass
class NBAGame:
    game_id: str        # e.g. "0022301234"
    home_team: str
    away_team: str
    status: str         # "Not Started", "In Progress", "Final"
    period: int         # current period (1-4, 5+ = OT)
    clock: str          # game clock e.g. "PT05M30.00S"


def get_todays_nba_games() -> list[NBAGame]:
    try:
        r = requests.get(NBA_SCOREBOARD, timeout=8,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"NBA schedule error: {e}")
        return []

    games = []
    for g in data.get("scoreboard", {}).get("games", []):
        games.append(NBAGame(
            game_id=g.get("gameId", ""),
            home_team=g.get("homeTeam", {}).get("teamName", ""),
            away_team=g.get("awayTeam", {}).get("teamName", ""),
            status=g.get("gameStatusText", ""),
            period=g.get("period", 0),
            clock=g.get("gameClock", ""),
        ))
    return games


if __name__ == "__main__":
    games = get_todays_nba_games()
    print(f"Today's NBA games: {len(games)}")
    for g in games:
        print(f"  [{g.game_id}] {g.away_team} @ {g.home_team}  {g.status}  Q{g.period} {g.clock}")

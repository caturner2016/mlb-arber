"""
Live batting order tracker.

Polls the MLB Stats API live feed to determine who is currently batting
and who is coming up 3-4 batters ahead — the target window for the
lineup arb strategy.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import aiohttp

MLB_LIVE = "https://statsapi.mlb.com/api/v1.1"
MLB_API  = "https://statsapi.mlb.com/api/v1"

log = logging.getLogger(__name__)


@dataclass
class BattingState:
    game_pk:       int
    inning:        int
    is_top:        bool          # True = away batting, False = home batting
    current_batter_id: int
    batting_order: list[int]     # 9 player IDs in order (for the team at bat)
    id_to_name:    dict[int, str] = field(default_factory=dict)
    player_hits:   dict[int, int] = field(default_factory=dict)  # player_id → hits today

    def upcoming(self, lookahead: int) -> int | None:
        """Player ID of the batter `lookahead` spots ahead of current."""
        if not self.batting_order or self.current_batter_id not in self.batting_order:
            return None
        idx = self.batting_order.index(self.current_batter_id)
        return self.batting_order[(idx + lookahead) % 9]

    def name(self, player_id: int) -> str:
        return self.id_to_name.get(player_id, "")


async def get_batting_state(
    game_pk: int,
    session: aiohttp.ClientSession,
    id_name_cache: dict[int, str],
) -> BattingState | None:
    """
    Fetch the live feed and return current batting state.
    Returns None if game is not in progress.
    """
    url = f"{MLB_LIVE}/game/{game_pk}/feed/live"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                log.debug(f"Feed {game_pk} status {resp.status}")
                return None
            data = await resp.json(content_type=None)
    except Exception as e:
        log.warning(f"Feed error game {game_pk}: {e}")
        return None

    live = data.get("liveData", {})
    linescore = live.get("linescore", {})
    inning    = linescore.get("currentInning", 0)
    is_top    = linescore.get("isTopInning", True)

    batter_info = linescore.get("offense", {}).get("batter", {})
    current_batter_id = batter_info.get("id")
    if not current_batter_id or inning == 0:
        return None

    # Batting order for the team currently at bat
    boxscore = live.get("boxscore", {}).get("teams", {})
    team_key = "away" if is_top else "home"
    batting_order = boxscore.get(team_key, {}).get("battingOrder", [])

    # Build / update id→name cache from gameData.players
    players_data = data.get("gameData", {}).get("players", {})
    for key, p in players_data.items():
        pid = p.get("id")
        name = p.get("fullName", "")
        if pid and name:
            id_name_cache[pid] = name

    # Collect hits for all batters in both teams from boxscore
    player_hits: dict[int, int] = {}
    for team_key in ("home", "away"):
        team_players = boxscore.get(team_key, {}).get("players", {})
        for pkey, pdata in team_players.items():
            pid = pdata.get("person", {}).get("id")
            hits = pdata.get("stats", {}).get("batting", {}).get("hits", 0)
            if pid is not None:
                player_hits[pid] = int(hits)

    return BattingState(
        game_pk=game_pk,
        inning=inning,
        is_top=is_top,
        current_batter_id=current_batter_id,
        batting_order=batting_order,
        id_to_name=id_name_cache,
        player_hits=player_hits,
    )

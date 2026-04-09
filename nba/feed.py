"""
NBA live boxscore feed — maximum speed via NBA CDN.

Endpoint: cdn.nba.com/static/json/liveData/boxscore/boxscore_{gameId}.json
Updates every ~5 seconds during live games. No auth required.

Stat correction protection:
  A stat must appear in 2 CONSECUTIVE polls before firing.
  This prevents acting on stats that get corrected/removed immediately.
  Cost: ~2-4 seconds of extra latency. Worth it.

Prop types tracked:
  points     — cumulative points scored
  rebounds   — total rebounds (offensive + defensive)
  assists    — assists
  threes     — 3-pointers made
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp


NBA_BOXSCORE = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"

# Points thresholds only — baskets are never reversed, zero correction risk
THRESHOLDS: dict[str, list[int]] = {
    "points": [10, 15, 20, 25, 30, 35, 40],
}

CONFIRM_POLLS = 2   # stat must appear this many consecutive polls before firing


@dataclass
class NBAStatEvent:
    game_id: str
    player_id: int
    player_name: str
    stat_type: str      # "points", "rebounds", "assists", "threes"
    threshold: int      # the threshold crossed (e.g. 25 for "25+ points")
    current_value: int  # actual current stat value
    period: int
    clock: str
    feed_detected_at: float = field(default_factory=time.monotonic)


class NBAGamePoller:
    """
    Polls one NBA game's live boxscore.
    Tracks confirmed stat crossings with a 2-poll confirmation buffer.
    """

    def __init__(self, game_id: str) -> None:
        self.game_id = game_id
        # fired[(player_id, stat_type, threshold)] = True once fired
        self._fired: set[tuple] = set()
        # pending[(player_id, stat_type, threshold)] = consecutive poll count
        self._pending: dict[tuple, int] = defaultdict(int)
        # last known stat values for change detection
        self._last_stats: dict[tuple[int, str], int] = {}

    async def poll(self, session: aiohttp.ClientSession) -> list[NBAStatEvent]:
        url = NBA_BOXSCORE.format(game_id=self.game_id)
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=3),
                headers={"User-Agent": "Mozilla/5.0"},
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return []

        game = data.get("game", {})
        period = game.get("period", 0)
        clock  = game.get("gameClock", "")
        status = game.get("gameStatus", 1)  # 1=not started, 2=live, 3=final

        if status not in (2, 3):
            return []

        detected_at = time.monotonic()
        ready_events: list[NBAStatEvent] = []

        for team_key in ("homeTeam", "awayTeam"):
            for player in game.get(team_key, {}).get("players", []):
                pid  = player.get("personId", 0)
                name = player.get("name", "")
                stats = player.get("statistics", {})

                stat_map = {
                    "points": int(stats.get("points", 0) or 0),
                }

                for stat_type, value in stat_map.items():
                    for threshold in THRESHOLDS[stat_type]:
                        key = (pid, stat_type, threshold)

                        if key in self._fired:
                            continue

                        if value >= threshold:
                            self._pending[key] += 1
                            if self._pending[key] >= CONFIRM_POLLS:
                                self._fired.add(key)
                                del self._pending[key]
                                ready_events.append(NBAStatEvent(
                                    game_id=self.game_id,
                                    player_id=pid,
                                    player_name=name,
                                    stat_type=stat_type,
                                    threshold=threshold,
                                    current_value=value,
                                    period=period,
                                    clock=clock,
                                    feed_detected_at=detected_at,
                                ))
                        else:
                            # Stat dropped (correction) — reset confirmation
                            if key in self._pending:
                                del self._pending[key]

        return ready_events


async def stream_nba_games(
    game_ids: list[str],
    poll_interval: float = 1.0,
) -> AsyncIterator[NBAStatEvent]:
    """
    Poll all NBA games concurrently. Yields confirmed stat events.
    1.0s interval — NBA CDN updates every ~5s so 1s is plenty.
    """
    pollers = {gid: NBAGamePoller(gid) for gid in game_ids}

    connector = aiohttp.TCPConnector(
        limit=len(game_ids) * 2 + 4,
        ttl_dns_cache=600,
        keepalive_timeout=60,
    )

    async with aiohttp.ClientSession(
        connector=connector,
        headers={"Accept-Encoding": "gzip", "Connection": "keep-alive"},
    ) as session:
        while True:
            cycle_start = time.monotonic()

            results = await asyncio.gather(
                *[p.poll(session) for p in pollers.values()],
                return_exceptions=True,
            )

            for game_events in results:
                if isinstance(game_events, list):
                    for event in game_events:
                        yield event

            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.0, poll_interval - elapsed)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    import sys, asyncio

    async def test(game_id: str) -> None:
        print(f"Polling NBA game {game_id} at 1s (Ctrl+C to stop)")
        async for ev in stream_nba_games([game_id], poll_interval=1.0):
            lag = (time.monotonic() - ev.feed_detected_at) * 1000
            print(
                f"  {ev.player_name:25s} {ev.stat_type:10s} {ev.threshold}+  "
                f"(has {ev.current_value})  Q{ev.period} {ev.clock}  "
                f"lag={lag:.0f}ms"
            )

    gid = sys.argv[1] if len(sys.argv) > 1 else "0022301234"
    asyncio.run(test(gid))

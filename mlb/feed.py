"""
MLB Stats API live play-by-play feed — optimised for minimum latency.

Strategy:
- aiohttp async HTTP (no GIL blocking, faster than requests for concurrent polls)
- All games polled concurrently via asyncio.gather
- Only new plays yielded (tracked by lastEventIndex per game)
- Minimal JSON parsing — extract only player_id + event_type on the hot path
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp


MLB_LIVE_API = "https://statsapi.mlb.com/api/v1.1"

# Event types we care about — maps MLB feed event strings to our canonical names
PROP_EVENT_MAP: dict[str, str] = {
    "Home Run": "home_run",
    "Single": "hit",
    "Double": "hit",
    "Triple": "hit",
    "Strikeout": "strikeout",
    "Strikeout - DP": "strikeout",
    "Walk": "walk",
    "Hit By Pitch": "hit_by_pitch",
    "Sac Fly": "sac_fly",
    "Field Error": "error",
    "Stolen Base 2B": "stolen_base",
    "Stolen Base 3B": "stolen_base",
    "Stolen Base Home": "stolen_base",
    "Runner Out": "runner_out",
}


@dataclass
class PropEvent:
    game_pk: int
    player_id: int          # batter (or pitcher for strikeout props)
    pitcher_id: int         # always the current pitcher
    event_type: str         # canonical name from PROP_EVENT_MAP
    raw_event: str          # original MLB event string
    inning: int
    half: str               # "top" or "bottom"
    home_score: int
    away_score: int
    feed_detected_at: float = field(default_factory=time.monotonic)  # monotonic ns for latency


@dataclass
class GamePoller:
    game_pk: int
    _last_event_index: int = -1

    async def poll(self, session: aiohttp.ClientSession) -> list[PropEvent]:
        """
        Single poll of one game. Returns only new events since last call.
        Designed to be called in a tight loop — returns immediately if no new plays.
        """
        url = f"{MLB_LIVE_API}/game/{self.game_pk}/feed/live"
        params = {"fields": "liveData,plays,allPlays,result,about,matchup,playEvents"}

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return []

        plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
        if not plays:
            return []

        # Identify new plays by index
        new_events: list[PropEvent] = []
        detected_at = time.monotonic()

        for play in plays:
            about = play.get("about", {})
            idx = about.get("atBatIndex", -1)
            if idx <= self._last_event_index:
                continue
            # Only process completed at-bats
            if not about.get("isComplete", False):
                continue

            result = play.get("result", {})
            raw_event = result.get("event", "")
            canonical = PROP_EVENT_MAP.get(raw_event)
            if canonical is None:
                self._last_event_index = max(self._last_event_index, idx)
                continue

            matchup = play.get("matchup", {})
            batter_id = matchup.get("batter", {}).get("id", 0)
            pitcher_id = matchup.get("pitcher", {}).get("id", 0)

            linescore = data.get("liveData", {}).get("linescore", {})
            home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0)
            away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0)

            new_events.append(PropEvent(
                game_pk=self.game_pk,
                player_id=batter_id,
                pitcher_id=pitcher_id,
                event_type=canonical,
                raw_event=raw_event,
                inning=about.get("inning", 0),
                half="top" if about.get("halfInning") == "top" else "bottom",
                home_score=home_score,
                away_score=away_score,
                feed_detected_at=detected_at,
            ))
            self._last_event_index = max(self._last_event_index, idx)

        return new_events


async def stream_all_games(
    game_pks: list[int],
    poll_interval: float = 1.0,
) -> AsyncIterator[PropEvent]:
    """
    Concurrently poll all games and yield PropEvents as they arrive.
    poll_interval: seconds between poll cycles (1.0 = fastest practical rate).
    """
    pollers = {pk: GamePoller(game_pk=pk) for pk in game_pks}
    connector = aiohttp.TCPConnector(limit=len(game_pks) + 4, ttl_dns_cache=300)

    async with aiohttp.ClientSession(
        connector=connector,
        headers={"Accept-Encoding": "gzip"},
    ) as session:
        while True:
            cycle_start = time.monotonic()

            # Fire all polls concurrently
            results = await asyncio.gather(
                *[p.poll(session) for p in pollers.values()],
                return_exceptions=True,
            )

            for game_events in results:
                if isinstance(game_events, Exception):
                    continue
                for event in game_events:
                    yield event

            # Sleep only the remainder of the interval (absorb poll time)
            elapsed = time.monotonic() - cycle_start
            sleep_for = max(0.0, poll_interval - elapsed)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    # Quick test: replay a known game
    import sys

    async def test(game_pk: int) -> None:
        print(f"Polling game {game_pk} — press Ctrl+C to stop")
        async for event in stream_all_games([game_pk], poll_interval=1.0):
            lag_ms = (time.monotonic() - event.feed_detected_at) * 1000
            print(
                f"  [{event.game_pk}] {event.raw_event:20s} "
                f"batter={event.player_id} pitcher={event.pitcher_id} "
                f"inning={event.inning}{event.half[0].upper()} "
                f"score={event.away_score}-{event.home_score} "
                f"lag={lag_ms:.1f}ms"
            )

    game_pk = int(sys.argv[1]) if len(sys.argv) > 1 else 745456
    asyncio.run(test(game_pk))

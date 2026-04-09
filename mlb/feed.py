"""
MLB live play-by-play feed — maximum speed edition.

Speed optimizations:
1. `timecode` parameter: returns ONLY events after a given timestamp,
   making the response tiny (vs full game history). Safe to poll at 0.5s.
2. aiohttp async: non-blocking, all games polled concurrently via gather().
3. Persistent TCP connections: reused across polls (no reconnect overhead).
4. Minimal JSON parsing: only extract player_id + event_type on hot path.
5. Fields filter: request only the fields we need.

Upgrade path for faster data:
  Sportradar Push API (~1-3s faster than public MLB API)
  Stats Perform / Opta (used by professional trading desks)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp


MLB_LIVE = "https://statsapi.mlb.com/api/v1.1"

# Minimal fields — reduces response size dramatically
LIVE_FIELDS = (
    "liveData,plays,allPlays,result,about,matchup,"
    "liveData,linescore,teams,runs"
)

# MLB event strings → canonical prop types
PROP_EVENT_MAP: dict[str, str] = {
    "Home Run":          "home_run",
    "Single":            "hit",
    "Double":            "hit",
    "Triple":            "hit",
    "Strikeout":         "strikeout",
    "Strikeout - DP":    "strikeout",
    "Walk":              "walk",
    "Hit By Pitch":      "hit_by_pitch",
    "Sac Fly":           "sac_fly",
    "Stolen Base 2B":    "stolen_base",
    "Stolen Base 3B":    "stolen_base",
    "Stolen Base Home":  "stolen_base",
}


@dataclass
class PropEvent:
    game_pk: int
    player_id: int          # batter
    pitcher_id: int
    event_type: str         # canonical (from PROP_EVENT_MAP)
    raw_event: str          # original MLB string e.g. "Home Run"
    player_name: str        # full name for direct cache lookup
    inning: int
    half: str               # "top" or "bottom"
    home_score: int
    away_score: int
    feed_detected_at: float = field(default_factory=time.monotonic)


class GamePoller:
    """
    Polls one game using the timecode trick for minimal response size.
    Timecode format: YYYYMMDD_HHMMSS  (UTC)
    """

    def __init__(self, game_pk: int) -> None:
        self.game_pk = game_pk
        self._last_event_index = -1
        self._timecode: str | None = None   # set after first new event

    async def poll(self, session: aiohttp.ClientSession) -> list[PropEvent]:
        url = f"{MLB_LIVE}/game/{self.game_pk}/feed/live"
        params: dict[str, str] = {"fields": LIVE_FIELDS}
        if self._timecode:
            params["timecode"] = self._timecode   # only get events after this

        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return []

        plays = data.get("liveData", {}).get("plays", {}).get("allPlays", [])
        if not plays:
            return []

        new_events: list[PropEvent] = []
        detected_at = time.monotonic()

        for play in plays:
            about = play.get("about", {})
            idx = about.get("atBatIndex", -1)
            if idx <= self._last_event_index:
                continue
            if not about.get("isComplete", False):
                continue

            result = play.get("result", {})
            raw_event = result.get("event", "")
            canonical = PROP_EVENT_MAP.get(raw_event)

            # Always update index regardless of whether we care about this event
            self._last_event_index = max(self._last_event_index, idx)

            if canonical is None:
                continue

            matchup = play.get("matchup", {})
            batter = matchup.get("batter", {})
            pitcher = matchup.get("pitcher", {})

            linescore = data.get("liveData", {}).get("linescore", {})
            teams = linescore.get("teams", {})

            new_events.append(PropEvent(
                game_pk=self.game_pk,
                player_id=batter.get("id", 0),
                pitcher_id=pitcher.get("id", 0),
                player_name=batter.get("fullName", ""),
                event_type=canonical,
                raw_event=raw_event,
                inning=about.get("inning", 0),
                half="top" if about.get("halfInning") == "top" else "bottom",
                home_score=teams.get("home", {}).get("runs", 0),
                away_score=teams.get("away", {}).get("runs", 0),
                feed_detected_at=detected_at,
            ))

            # Advance timecode to now so next poll only fetches newer events
            import datetime
            self._timecode = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        return new_events


async def stream_all_games(
    game_pks: list[int],
    poll_interval: float = 0.5,
) -> AsyncIterator[PropEvent]:
    """
    Poll all games concurrently, yield events as they arrive.
    Default 0.5s — safe with timecode trick (tiny responses per poll).
    """
    pollers = {pk: GamePoller(pk) for pk in game_pks}

    # Persistent connection pool — no reconnect overhead per poll
    connector = aiohttp.TCPConnector(
        limit=len(game_pks) * 2 + 4,
        ttl_dns_cache=600,
        keepalive_timeout=60,
    )

    async with aiohttp.ClientSession(
        connector=connector,
        headers={
            "Accept-Encoding": "gzip",
            "Connection": "keep-alive",
        },
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

    async def test(game_pk: int) -> None:
        print(f"Polling game {game_pk} at 0.5s  (Ctrl+C to stop)")
        async for ev in stream_all_games([game_pk], poll_interval=0.5):
            lag = (time.monotonic() - ev.feed_detected_at) * 1000
            print(
                f"  {ev.raw_event:20s}  {ev.player_name:25s} "
                f"inn={ev.inning}{ev.half[0].upper()} "
                f"score={ev.away_score}-{ev.home_score}  "
                f"process_lag={lag:.1f}ms"
            )

    pk = int(sys.argv[1]) if len(sys.argv) > 1 else 745456
    asyncio.run(test(pk))

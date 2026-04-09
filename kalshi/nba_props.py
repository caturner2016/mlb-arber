"""
Kalshi NBA prop market cache.

Auto-discovers NBA series tickers via the Kalshi events API,
then builds a lookup map: (player_name, stat_type, threshold) → PropMarket.

NBA prop types tracked:
  points   — e.g. "LeBron James: 25+ Points"
  rebounds — e.g. "Nikola Jokic: 10+ Rebounds"
  assists  — e.g. "LeBron James: 8+ Assists"
  threes   — e.g. "Stephen Curry: 4+ Threes"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from kalshi.client import KalshiClient


log = logging.getLogger(__name__)

# Known NBA series tickers — will also auto-discover via events API
NBA_SERIES_HINTS = [
    "KXNBAPTS",   # points
    "KXNBAREB",   # rebounds
    "KXNBAAST",   # assists
    "KXNBA3PT",   # 3-pointers
    "KXNBA",      # general
    "KXNBAPROP",  # props
]

# Points only — baskets are never reversed, zero correction risk
STAT_PATTERNS = [
    (r"(\d+)\+?\s*points?", "points"),
    (r"(\d+)\+?\s*pts?",    "points"),
]

NBA_KEYWORDS = {"nba", "basketball", "points"}


@dataclass
class NBAPropMarket:
    ticker: str
    title: str
    player_name: str    # lowercase
    stat_type: str      # "points", "rebounds", "assists", "threes"
    threshold: int
    yes_ask: int | None


def _extract_stat(title: str) -> tuple[str, int] | None:
    lower = title.lower()
    for pattern, stat_type in STAT_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            return stat_type, int(m.group(1))
    return None


def _extract_player(title: str) -> str:
    # "LeBron James: 25+ Points" → "lebron james"
    part = title.split(":")[0].strip()
    if part:
        return part.lower()
    # Fallback: first two capitalized words
    words = title.split()
    caps = [w for w in words if w and w[0].isupper() and w[0].isalpha()]
    return " ".join(caps[:2]).lower() if len(caps) >= 2 else ""


def _is_nba(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in NBA_KEYWORDS)


class NBAPropCache:
    """
    Pre-loaded at startup. Maps:
      (player_name_lower, stat_type, threshold) → NBAPropMarket
    Rebuilt every 30 minutes.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, int], NBAPropMarket] = {}

    async def build(self, kalshi: KalshiClient) -> None:
        log.info("Building Kalshi NBA prop cache…")

        # Discover NBA series via events API
        series_set = set(NBA_SERIES_HINTS)
        discovered = await self._discover_nba_series(kalshi)
        series_set.update(discovered)

        # Fetch markets from all series
        import asyncio
        batches = await asyncio.gather(
            *[self._load_series(kalshi, s) for s in series_set],
            return_exceptions=True,
        )

        new_cache: dict[tuple[str, str, int], NBAPropMarket] = {}
        for batch in batches:
            if isinstance(batch, list):
                for prop in batch:
                    key = (prop.player_name, prop.stat_type, prop.threshold)
                    new_cache[key] = prop

        self._cache = new_cache
        log.info(f"NBA prop cache ready: {len(self._cache)} markets")

    async def _discover_nba_series(self, kalshi: KalshiClient) -> list[str]:
        series: set[str] = set()
        cursor = None
        for _ in range(5):
            try:
                import aiohttp
                session = await kalshi._session_()
                params = {"status": "open", "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                from kalshi.client import KALSHI_BASE
                async with session.get(
                    KALSHI_BASE + "/events",
                    headers=kalshi._auth_headers("GET", "/events"),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json(content_type=None)
            except Exception:
                break

            for event in data.get("events", []):
                title  = event.get("title", "") or event.get("sub_title", "")
                cat    = event.get("category", "")
                ticker = event.get("series_ticker", "")
                if _is_nba(title) or _is_nba(cat) or "nba" in ticker.lower():
                    if ticker:
                        series.add(ticker)

            cursor = data.get("cursor")
            if not cursor:
                break

        return list(series)

    async def _load_series(self, kalshi: KalshiClient, series: str) -> list[NBAPropMarket]:
        results: list[NBAPropMarket] = []
        cursor = None
        while True:
            try:
                data = await kalshi.get_markets(series, cursor)
            except Exception as e:
                log.debug(f"NBA series {series}: {e}")
                break

            for m in data.get("markets", []):
                ticker = m.get("ticker", "")
                title  = m.get("title", "") or m.get("subtitle", "")

                stat = _extract_stat(title)
                if not stat:
                    continue
                stat_type, threshold = stat

                player = _extract_player(title)
                if not player:
                    continue

                results.append(NBAPropMarket(
                    ticker=ticker,
                    title=title,
                    player_name=player,
                    stat_type=stat_type,
                    threshold=threshold,
                    yes_ask=m.get("yes_ask"),
                ))

            cursor = data.get("cursor")
            if not cursor or not data.get("markets"):
                break

        return results

    def lookup(self, player_name: str, stat_type: str, threshold: int) -> NBAPropMarket | None:
        lower = player_name.lower().strip()
        key = (lower, stat_type, threshold)
        if key in self._cache:
            return self._cache[key]
        # Fallback: last name match
        last = lower.split()[-1] if lower.split() else ""
        for (name, stype, thr), prop in self._cache.items():
            if stype == stat_type and thr == threshold and last and name.endswith(last):
                return prop
        return None

    def __len__(self) -> int:
        return len(self._cache)


if __name__ == "__main__":
    import asyncio, yaml

    async def main() -> None:
        cfg    = yaml.safe_load(open("config.yaml"))
        kalshi = KalshiClient(
            key_id=cfg.get("kalshi_key_id", ""),
            private_key_path=cfg.get("kalshi_private_key_path", ""),
            paper_mode=True,
        )
        cache = NBAPropCache()
        await cache.build(kalshi)
        print(f"\nNBA prop markets: {len(cache)}")
        for (name, stype, thr), prop in list(cache._cache.items())[:20]:
            print(f"  {name:25s} {stype:10s} {thr}+  → {prop.ticker}  ask={prop.yes_ask}¢")
        await kalshi.close()

    asyncio.run(main())

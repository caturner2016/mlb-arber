"""
Kalshi MLB prop market cache.

Series tickers (confirmed working):
  KXMLBHR  — home run markets  (ticker ends in -1 for 1+ HR)
  KXMLBHIT — hits markets      (ticker ends in -1/-2/-3 for 1+/2+/3+ hits)

Market title format: "Player Name: Home Run" or "Player Name: Hits 1+"
Player name is always before the first colon.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kalshi.client import KalshiClient


log = logging.getLogger(__name__)

HR_SERIES  = "KXMLBHR"
HIT_SERIES = "KXMLBHIT"


@dataclass
class PropMarket:
    ticker: str
    title: str
    player_name: str   # lowercase, as extracted from title
    prop_type: str     # "home_run" or "hit"
    threshold: int     # 1, 2, or 3
    yes_ask: int | None


class PropCache:
    """
    Pre-loaded at startup. Maps:
      hr_cache:   player_name (lower) → PropMarket
      hits_cache: "player_name:threshold" → PropMarket
    Refreshed every 30 minutes in background.
    """

    def __init__(self) -> None:
        self.hr_cache:   dict[str, PropMarket] = {}
        self.hits_cache: dict[str, PropMarket] = {}

    async def build(self, kalshi: KalshiClient) -> None:
        log.info("Building Kalshi prop cache…")
        hr_raw   = await self._load_series(kalshi, HR_SERIES,  "home_run")
        hits_raw = await self._load_series(kalshi, HIT_SERIES, "hit")

        new_hr:   dict[str, PropMarket] = {}
        new_hits: dict[str, PropMarket] = {}

        for prop in hr_raw:
            # Only keep 1+ HR markets
            if prop.threshold == 1:
                new_hr[prop.player_name] = prop

        for prop in hits_raw:
            key = f"{prop.player_name}:{prop.threshold}"
            new_hits[key] = prop

        self.hr_cache   = new_hr
        self.hits_cache = new_hits
        log.info(f"Cache ready: {len(new_hr)} HR markets, {len(new_hits)} hits markets")

    async def _load_series(
        self, kalshi: KalshiClient, series: str, prop_type: str
    ) -> list[PropMarket]:
        results: list[PropMarket] = []
        cursor = None

        while True:
            try:
                data = await kalshi.get_markets(series, cursor)
            except Exception as e:
                log.warning(f"Cache load error ({series}): {e}")
                break

            markets = data.get("markets", [])
            for m in markets:
                ticker = m.get("ticker", "")
                title  = m.get("title", "")

                # Player name is before the colon: "Aaron Judge: Home Run"
                player = title.split(":")[0].strip().lower()
                if not player:
                    continue

                # Threshold from ticker suffix (-1, -2, -3)
                threshold = None
                if ticker.endswith("-1"):
                    threshold = 1
                elif ticker.endswith("-2"):
                    threshold = 2
                elif ticker.endswith("-3"):
                    threshold = 3

                if threshold is None:
                    continue

                yes_ask = m.get("yes_ask")
                results.append(PropMarket(
                    ticker=ticker,
                    title=title,
                    player_name=player,
                    prop_type=prop_type,
                    threshold=threshold,
                    yes_ask=yes_ask,
                ))

            cursor = data.get("cursor")
            if not cursor or not markets:
                break

        return results

    def get_hr_ticker(self, player_name: str) -> PropMarket | None:
        """Look up 1+ HR market by player name. Tries full name then last name."""
        lower = player_name.lower().strip()
        if lower in self.hr_cache:
            return self.hr_cache[lower]
        # Fallback: match by last name
        last = lower.split()[-1] if lower.split() else ""
        for key, prop in self.hr_cache.items():
            if last and key.endswith(last):
                return prop
        return None

    def get_hit_ticker(self, player_name: str, threshold: int) -> PropMarket | None:
        """Look up hits market by player name and threshold (1, 2, or 3)."""
        lower = player_name.lower().strip()
        key   = f"{lower}:{threshold}"
        if key in self.hits_cache:
            return self.hits_cache[key]
        # Fallback: last name
        last = lower.split()[-1] if lower.split() else ""
        for k, prop in self.hits_cache.items():
            name_part = k.split(":")[0]
            thr_part  = int(k.split(":")[1]) if ":" in k else 0
            if last and name_part.endswith(last) and thr_part == threshold:
                return prop
        return None


if __name__ == "__main__":
    import asyncio, sys, yaml

    async def main() -> None:
        cfg    = yaml.safe_load(open("config.yaml"))
        kalshi = KalshiClient(
            key_id=cfg.get("kalshi_key_id", ""),
            private_key_path=cfg.get("kalshi_private_key_path", ""),
            paper_mode=True,
        )
        cache = PropCache()
        await cache.build(kalshi)
        print(f"\nHR markets ({len(cache.hr_cache)}):")
        for name, prop in list(cache.hr_cache.items())[:10]:
            print(f"  {name!r:30s} → {prop.ticker}  ask={prop.yes_ask}¢")
        print(f"\nHits markets ({len(cache.hits_cache)}):")
        for key, prop in list(cache.hits_cache.items())[:10]:
            print(f"  {key!r:35s} → {prop.ticker}  ask={prop.yes_ask}¢")
        await kalshi.close()

    asyncio.run(main())

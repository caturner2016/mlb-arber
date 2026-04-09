"""
Discover today's MLB player prop markets on Kalshi.

Kalshi MLB prop tickers live under series like KXMLB*.
We search for markets with titles containing player names and prop keywords,
then map them to (player_name, prop_type) for trigger matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any

import aiohttp


KALSHI_API = "https://trading-api.kalshi.com/trade-api/v2"

# Keywords in Kalshi market titles that map to our prop types
PROP_KEYWORDS: dict[str, str] = {
    "home run": "home_run",
    "homer": "home_run",
    "hr": "home_run",
    "hit": "hit",
    "strikeout": "strikeout",
    "strike out": "strikeout",
    "rbi": "rbi",
    "stolen base": "stolen_base",
    "walk": "walk",
}

# Series tickers to search — add more as Kalshi expands
MLB_SERIES = [
    "KXMLB",
    "KXMLBHR",    # home run specific
    "KXMLBHIT",
    "KXMLBK",     # strikeout
]


@dataclass
class PropMarket:
    ticker: str
    title: str
    player_name: str       # extracted from title
    prop_type: str         # canonical: home_run, hit, strikeout, etc.
    yes_ask: int | None    # current ask in cents at discovery time
    close_time: str        # when the market closes


def _extract_player_name(title: str) -> str:
    """
    Best-effort extraction of a player name from a Kalshi market title.

    Examples:
      "Will Aaron Judge hit a home run today?" → "Aaron Judge"
      "Aaron Judge HR 4/9" → "Aaron Judge"
    """
    # Pattern: "Will <Name> <verb>" → extract Name
    m = re.match(r"Will\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+", title)
    if m:
        return m.group(1)
    # Fallback: first two capitalized words
    words = title.split()
    caps = [w for w in words if w and w[0].isupper() and w.isalpha()]
    if len(caps) >= 2:
        return f"{caps[0]} {caps[1]}"
    return ""


def _detect_prop_type(title: str) -> str | None:
    lower = title.lower()
    for kw, ptype in PROP_KEYWORDS.items():
        if kw in lower:
            return ptype
    return None


async def get_todays_props(
    session: aiohttp.ClientSession,
    target_date: date | None = None,
) -> list[PropMarket]:
    """
    Query Kalshi for all active MLB player prop markets.
    Returns parsed PropMarket list.
    """
    target = target_date or date.today()
    date_str = target.strftime("%Y-%m-%d")
    props: list[PropMarket] = []
    seen: set[str] = set()

    # Try each known MLB series
    for series in MLB_SERIES:
        cursor = None
        while True:
            params: dict[str, Any] = {
                "series_ticker": series,
                "status": "open",
                "limit": 200,
            }
            if cursor:
                params["cursor"] = cursor

            try:
                async with session.get(
                    f"{KALSHI_API}/markets",
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json(content_type=None)
            except aiohttp.ClientError:
                break

            markets = data.get("markets", [])
            for m in markets:
                ticker = m.get("ticker", "")
                title = m.get("title", "") or m.get("subtitle", "")
                close_time = m.get("close_time", "")

                # Skip if already seen or not a player prop
                if ticker in seen:
                    continue

                prop_type = _detect_prop_type(title)
                if not prop_type:
                    continue

                player_name = _extract_player_name(title)
                if not player_name:
                    continue

                # Get best YES ask from the market data
                yes_ask = m.get("yes_ask")

                props.append(PropMarket(
                    ticker=ticker,
                    title=title,
                    player_name=player_name,
                    prop_type=prop_type,
                    yes_ask=yes_ask,
                    close_time=close_time,
                ))
                seen.add(ticker)

            cursor = data.get("cursor")
            if not cursor or not markets:
                break

    return props


async def build_trigger_map(
    props: list[PropMarket],
    player_id_map: dict[str, int],
) -> dict[tuple[int, str], PropMarket]:
    """
    Map (player_id, prop_type) → PropMarket.
    This is what main.py checks on every incoming game event.
    """
    from mlb.players import resolve_player

    trigger_map: dict[tuple[int, str], PropMarket] = {}
    for prop in props:
        pid = resolve_player(prop.player_name, player_id_map)
        if pid is None:
            continue
        key = (pid, prop.prop_type)
        # If duplicate, keep the one with lower yes_ask (more conservative)
        if key not in trigger_map or (prop.yes_ask or 100) < (trigger_map[key].yes_ask or 100):
            trigger_map[key] = prop

    return trigger_map


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        connector = aiohttp.TCPConnector(limit=5)
        async with aiohttp.ClientSession(connector=connector) as session:
            props = await get_todays_props(session)
            print(f"Found {len(props)} MLB prop markets")
            for p in props[:20]:
                print(f"  [{p.ticker}] {p.title!r} → player={p.player_name!r} type={p.prop_type} ask={p.yes_ask}¢")

    asyncio.run(main())

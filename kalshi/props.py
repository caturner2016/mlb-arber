"""
Discover today's MLB player prop markets on Kalshi.

Uses the Kalshi events API to auto-discover MLB series tickers rather than
guessing them — then fetches all open markets under those series.
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
    " hr ": "home_run",
    "hit a home": "home_run",
    "record a hit": "hit",
    "get a hit": "hit",
    "record a strikeout": "strikeout",
    "strikeout": "strikeout",
    "strike out": "strikeout",
    "record an rbi": "rbi",
    " rbi": "rbi",
    "stolen base": "stolen_base",
    "steal a base": "stolen_base",
}

# Baseball-related keywords to identify MLB events/series
BASEBALL_KEYWORDS = {"mlb", "baseball", "home run", "strikeout", "pitcher", "batter"}


@dataclass
class PropMarket:
    ticker: str
    title: str
    player_name: str
    prop_type: str
    yes_ask: int | None
    close_time: str


def _extract_player_name(title: str) -> str:
    """Extract player name from Kalshi market title."""
    # "Will Aaron Judge hit a home run" → "Aaron Judge"
    m = re.match(r"Will\s+([A-Z][a-z]+(?:\s+[A-Z][a-z']+)+)\s+", title)
    if m:
        return m.group(1).strip()
    # "Aaron Judge: Home Run" style
    m = re.match(r"([A-Z][a-z]+(?:\s+[A-Z][a-z']+)+)\s*[:\-]", title)
    if m:
        return m.group(1).strip()
    # Fallback: first two capitalized words
    words = title.split()
    caps = [w.strip("'s") for w in words if w and w[0].isupper() and w[0].isalpha()]
    if len(caps) >= 2:
        return f"{caps[0]} {caps[1]}"
    return ""


def _detect_prop_type(title: str) -> str | None:
    lower = " " + title.lower() + " "
    for kw, ptype in PROP_KEYWORDS.items():
        if kw in lower:
            return ptype
    return None


def _is_baseball_related(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in BASEBALL_KEYWORDS)


async def _discover_mlb_series(session: aiohttp.ClientSession) -> list[str]:
    """
    Hit the Kalshi events endpoint to find active MLB/baseball series tickers.
    Returns a list of series tickers like ["KXMLBHR", "KXMLBHIT", ...].
    """
    series_tickers: set[str] = set()
    cursor = None

    for _ in range(10):  # max 10 pages
        params: dict[str, Any] = {"status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor

        try:
            async with session.get(
                f"{KALSHI_API}/events",
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    break
                data = await resp.json(content_type=None)
        except aiohttp.ClientError:
            break

        events = data.get("events", [])
        for event in events:
            title = event.get("title", "") or event.get("sub_title", "")
            series = event.get("series_ticker", "")
            category = event.get("category", "")

            if _is_baseball_related(title) or _is_baseball_related(category) or _is_baseball_related(series):
                if series:
                    series_tickers.add(series)

        cursor = data.get("cursor")
        if not cursor or not events:
            break

    return list(series_tickers)


async def _fetch_markets_for_series(
    session: aiohttp.ClientSession,
    series: str,
) -> list[dict]:
    """Fetch all open markets for a given series ticker."""
    markets: list[dict] = []
    cursor = None

    while True:
        params: dict[str, Any] = {"series_ticker": series, "status": "open", "limit": 200}
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

        batch = data.get("markets", [])
        markets.extend(batch)
        cursor = data.get("cursor")
        if not cursor or not batch:
            break

    return markets


async def _broad_market_search(session: aiohttp.ClientSession) -> list[dict]:
    """
    Fallback: search all open markets for baseball-related titles.
    Used if series discovery finds nothing.
    """
    markets: list[dict] = []
    cursor = None

    for _ in range(5):
        params: dict[str, Any] = {"status": "open", "limit": 200}
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

        for m in data.get("markets", []):
            title = m.get("title", "") or m.get("subtitle", "")
            if _is_baseball_related(title) or _detect_prop_type(title):
                markets.append(m)

        cursor = data.get("cursor")
        if not cursor:
            break

    return markets


async def get_todays_props(
    session: aiohttp.ClientSession,
    target_date: date | None = None,
) -> list[PropMarket]:
    """
    Find all active Kalshi MLB player prop markets.
    Auto-discovers series tickers via the events API.
    """
    props: list[PropMarket] = []
    seen: set[str] = set()

    # Step 1: discover MLB series tickers
    mlb_series = await _discover_mlb_series(session)

    raw_markets: list[dict] = []

    if mlb_series:
        # Fetch markets from each discovered series
        import asyncio
        batches = await asyncio.gather(
            *[_fetch_markets_for_series(session, s) for s in mlb_series]
        )
        for batch in batches:
            raw_markets.extend(batch)
    else:
        # Fallback: broad search
        raw_markets = await _broad_market_search(session)

    # Step 2: parse out player prop markets
    for m in raw_markets:
        ticker = m.get("ticker", "")
        if ticker in seen:
            continue

        # Try title fields in order of preference
        title = (
            m.get("title")
            or m.get("subtitle")
            or m.get("market_type")
            or ""
        )

        prop_type = _detect_prop_type(title)
        if not prop_type:
            continue

        player_name = _extract_player_name(title)
        if not player_name:
            continue

        props.append(PropMarket(
            ticker=ticker,
            title=title,
            player_name=player_name,
            prop_type=prop_type,
            yes_ask=m.get("yes_ask"),
            close_time=m.get("close_time", ""),
        ))
        seen.add(ticker)

    return props


async def build_trigger_map(
    props: list[PropMarket],
    player_id_map: dict[str, int],
) -> dict[tuple[int, str], PropMarket]:
    """Map (player_id, prop_type) → PropMarket."""
    from mlb.players import resolve_player

    trigger_map: dict[tuple[int, str], PropMarket] = {}
    for prop in props:
        pid = resolve_player(prop.player_name, player_id_map)
        if pid is None:
            continue
        key = (pid, prop.prop_type)
        if key not in trigger_map or (prop.yes_ask or 100) < (trigger_map[key].yes_ask or 100):
            trigger_map[key] = prop

    return trigger_map


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        connector = aiohttp.TCPConnector(limit=10)
        async with aiohttp.ClientSession(connector=connector) as session:
            print("Discovering MLB series tickers...")
            series = await _discover_mlb_series(session)
            print(f"Found series: {series}")

            print("\nFetching prop markets...")
            props = await get_todays_props(session)
            print(f"Found {len(props)} MLB prop markets\n")
            for p in props[:30]:
                print(f"  [{p.ticker}] {p.title!r}")
                print(f"    player={p.player_name!r}  type={p.prop_type}  ask={p.yes_ask}¢")

    asyncio.run(main())

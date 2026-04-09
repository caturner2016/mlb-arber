"""
Kalshi weather market cache.

Discovers and maps Kalshi weather markets:
  - Temperature HIGH markets: "Will the high temp in NYC be X-Y°F today?"
  - Temperature LOW markets:  "Will the low temp in NYC be X-Y°F today?"
  - Rain markets:             "Will it rain in NYC today?"
  - Monthly/climate markets

Lookup: (city, market_type, threshold) → WeatherMarket

market_type values:
  "high_temp"   — daily high temperature (in °F range or above/below)
  "low_temp"    — daily low temperature
  "rain"        — any precipitation today (YES/NO)
  "rain_inches" — total precipitation amount
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from kalshi.client import KalshiClient

log = logging.getLogger(__name__)

# Known weather series tickers on Kalshi
WEATHER_SERIES_HINTS = [
    "KXHIGHNY",    # NYC high temp
    "KXHIGHLA",    # LA high temp
    "KXHIGHCHI",   # Chicago high temp
    "KXHIGHMIA",   # Miami high temp
    "KXHIGHDEN",   # Denver high temp
    "KXHIGHBOS",   # Boston high temp
    "KXHIGHSEA",   # Seattle high temp
    "KXHIGHPHX",   # Phoenix high temp
    "KXWEATHER",   # general weather
    "KXRAIN",      # rain markets
    "KXTEMP",      # temp markets
    "KXHIGH",      # high temp
    "KXLOW",       # low temp
    "KXNYCHIGH",
    "KXLAHIGH",
    "KXCHIHIGH",
    "KXSFHIGH",
    "KXDCHIGH",
    "KXATL",
    "KXHOU",
    "KXDAL",
    "KXPHX",
]

WEATHER_KEYWORDS = {
    "temperature", "high", "low", "rain", "precipitation",
    "weather", "degrees", "°f", "°c", "humid", "snow", "wind",
    "fahrenheit", "celsius", "forecast",
}

# City names to look for in market titles
KNOWN_CITIES = [
    "los angeles", "new york", "chicago", "miami", "austin",
    "denver", "boston", "atlanta", "philadelphia", "san francisco",
    "minneapolis", "washington", "seattle", "dallas", "oklahoma city",
    "phoenix", "las vegas", "houston", "san antonio", "new orleans",
    # abbreviations that might appear in titles
    "nyc", "la", "sf", "dc", "okc", "philly",
]

CITY_CANONICAL = {
    "nyc": "new york", "new york city": "new york", "ny": "new york",
    "la": "los angeles", "sf": "san francisco", "dc": "washington",
    "d.c.": "washington", "okc": "oklahoma city", "philly": "philadelphia",
}


@dataclass
class WeatherMarket:
    ticker: str
    title: str
    city: str           # lowercase canonical city name
    market_type: str    # "high_temp", "low_temp", "rain", "rain_inches"
    range_low: Optional[int]   # lower bound of temp range (inclusive)
    range_high: Optional[int]  # upper bound of temp range (inclusive), None = "above"
    yes_ask: Optional[int]     # current YES ask in cents


def _extract_city(title: str) -> Optional[str]:
    lower = title.lower()
    # Try longest match first
    for city in sorted(KNOWN_CITIES, key=len, reverse=True):
        if city in lower:
            return CITY_CANONICAL.get(city, city)
    return None


def _extract_temp_range(title: str) -> tuple[Optional[int], Optional[int]]:
    """
    Parse temperature range from title.
    Patterns:
      "70-74°F"  → (70, 74)
      "75-79"    → (75, 79)
      "80 or above" / "80+" → (80, None)
      "below 40" / "under 40" → (None, 39)
      "exactly 72" / "72°" → (72, 72)
    """
    lower = title.lower()

    # Range: "70-74" or "70 to 74"
    m = re.search(r'(\d+)\s*(?:-|to)\s*(\d+)', lower)
    if m:
        return int(m.group(1)), int(m.group(2))

    # Above/over/plus: "80 or above", "80+", "above 80", "over 80"
    m = re.search(r'(\d+)\s*(?:or\s+)?(?:above|over|\+)', lower)
    if m:
        return int(m.group(1)), None
    m = re.search(r'(?:above|over|at least|at or above)\s+(\d+)', lower)
    if m:
        return int(m.group(1)), None

    # Below/under: "below 40", "under 40", "39 or below"
    m = re.search(r'(?:below|under|at most|at or below)\s+(\d+)', lower)
    if m:
        return None, int(m.group(1)) - 1
    m = re.search(r'(\d+)\s*(?:or\s+)?(?:below|under)', lower)
    if m:
        return None, int(m.group(1))

    # Single value: "72°F", "exactly 72"
    m = re.search(r'(?:exactly\s+)?(\d{2,3})\s*°?f?\b', lower)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 130:
            return v, v

    return None, None


def _market_type(title: str) -> Optional[str]:
    lower = title.lower()
    if any(w in lower for w in ("rain", "precipitation", "precip", "wet")):
        if any(w in lower for w in ("inch", "total", "amount", "mm")):
            return "rain_inches"
        return "rain"
    if "high" in lower or "maximum" in lower or "max temp" in lower:
        return "high_temp"
    if "low" in lower or "minimum" in lower or "min temp" in lower:
        return "low_temp"
    if any(w in lower for w in ("temperature", "degrees", "°f", "fahrenheit")):
        # Disambiguate high/low from context
        if re.search(r'\b(high|max)\b', lower):
            return "high_temp"
        if re.search(r'\b(low|min)\b', lower):
            return "low_temp"
        return "high_temp"  # default
    return None


def _is_weather(title: str, category: str, ticker: str) -> bool:
    lower = (title + " " + category + " " + ticker).lower()
    return any(k in lower for k in WEATHER_KEYWORDS)


class WeatherPropCache:
    """
    Discovers and caches all open Kalshi weather markets.
    Maps (city, market_type, range_low, range_high) → WeatherMarket.
    Rebuilt every 30 minutes.
    """

    def __init__(self) -> None:
        self._markets: list[WeatherMarket] = []
        # index: (city, market_type) → list of WeatherMarket
        self._index: dict[tuple[str, str], list[WeatherMarket]] = {}

    async def build(self, kalshi: KalshiClient) -> None:
        log.info("Building Kalshi weather market cache…")

        series_set = set(WEATHER_SERIES_HINTS)
        discovered = await self._discover_weather_series(kalshi)
        series_set.update(discovered)

        import asyncio
        batches = await asyncio.gather(
            *[self._load_series(kalshi, s) for s in series_set],
            return_exceptions=True,
        )

        markets: list[WeatherMarket] = []
        for batch in batches:
            if isinstance(batch, list):
                markets.extend(batch)

        self._markets = markets
        self._index = {}
        for m in markets:
            key = (m.city, m.market_type)
            self._index.setdefault(key, []).append(m)

        log.info(f"Weather market cache ready: {len(self._markets)} markets")

    async def _discover_weather_series(self, kalshi: KalshiClient) -> list[str]:
        series: set[str] = set()
        cursor = None
        import aiohttp
        from kalshi.client import KALSHI_BASE
        for _ in range(10):
            try:
                session = await kalshi._session_()
                params = {"status": "open", "limit": 200}
                if cursor:
                    params["cursor"] = cursor
                async with session.get(
                    KALSHI_BASE + "/events",
                    headers=kalshi._auth_headers("GET", "/events"),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        break
                    data = await resp.json(content_type=None)
            except Exception as e:
                log.debug(f"Weather discover: {e}")
                break

            for event in data.get("events", []):
                title  = event.get("title", "") or ""
                cat    = event.get("category", "") or ""
                ticker = event.get("series_ticker", "") or ""
                if _is_weather(title, cat, ticker):
                    if ticker:
                        series.add(ticker)
                        log.debug(f"  Weather series found: {ticker} — {title}")

            cursor = data.get("cursor")
            if not cursor:
                break

        log.info(f"Discovered {len(series)} weather series via events API")
        return list(series)

    async def _load_series(self, kalshi: KalshiClient, series: str) -> list[WeatherMarket]:
        results: list[WeatherMarket] = []
        cursor = None
        while True:
            try:
                data = await kalshi.get_markets(series, cursor)
            except Exception as e:
                log.debug(f"Weather series {series}: {e}")
                break

            for m in data.get("markets", []):
                ticker = m.get("ticker", "")
                title  = m.get("title", "") or m.get("subtitle", "") or ""

                if not _is_weather(title, "", ticker):
                    continue

                city = _extract_city(title)
                if not city:
                    # Try the event title or subtitle
                    city = _extract_city(ticker)
                if not city:
                    continue

                mtype = _market_type(title)
                if not mtype:
                    continue

                range_low, range_high = _extract_temp_range(title) if "temp" in mtype else (None, None)

                results.append(WeatherMarket(
                    ticker=ticker,
                    title=title,
                    city=city,
                    market_type=mtype,
                    range_low=range_low,
                    range_high=range_high,
                    yes_ask=m.get("yes_ask"),
                ))

            cursor = data.get("cursor")
            if not cursor or not data.get("markets"):
                break

        return results

    def find_temp_market(
        self,
        city: str,
        market_type: str,  # "high_temp" or "low_temp"
        forecast_f: float,
    ) -> Optional[WeatherMarket]:
        """
        Given a city + market_type + forecast temperature,
        find the Kalshi market whose range contains that forecast.
        Returns the best matching market (smallest range that contains the value).
        """
        key = (city, market_type)
        candidates = self._index.get(key, [])
        if not candidates:
            return None

        # Find markets whose range contains the forecast
        matching = []
        for m in candidates:
            if m.range_low is None and m.range_high is None:
                continue  # no range parsed — skip
            low  = m.range_low  if m.range_low  is not None else -999
            high = m.range_high if m.range_high is not None else 999
            if low <= forecast_f <= high:
                matching.append(m)

        if not matching:
            return None

        # Prefer the narrowest range
        def range_width(m: WeatherMarket) -> int:
            lo = m.range_low  if m.range_low  is not None else 0
            hi = m.range_high if m.range_high is not None else 200
            return hi - lo

        return min(matching, key=range_width)

    def find_rain_market(self, city: str) -> Optional[WeatherMarket]:
        key = (city, "rain")
        candidates = self._index.get(key, [])
        return candidates[0] if candidates else None

    def all_markets(self) -> list[WeatherMarket]:
        return list(self._markets)

    def __len__(self) -> int:
        return len(self._markets)


if __name__ == "__main__":
    import asyncio, yaml

    async def main() -> None:
        cfg    = yaml.safe_load(open("config.yaml"))
        kalshi = KalshiClient(
            key_id=cfg.get("kalshi_key_id", ""),
            private_key_path=cfg.get("kalshi_private_key_path", ""),
            paper_mode=True,
        )
        cache = WeatherPropCache()
        await cache.build(kalshi)
        print(f"\nWeather markets: {len(cache)}")
        for m in cache.all_markets():
            rng = (
                f"{m.range_low}-{m.range_high}°F"
                if m.range_low is not None and m.range_high is not None
                else (f"{m.range_low}+°F" if m.range_high is None else f"<{m.range_high}°F")
                if (m.range_low or m.range_high) is not None else "—"
            )
            print(f"  {m.city:20s}  {m.market_type:12s}  {rng:12s}  ask={m.yes_ask}¢  {m.ticker}")
        await kalshi.close()

    asyncio.run(main())

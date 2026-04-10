"""
NOAA/NWS weather data fetcher.

Uses the National Weather Service API (api.weather.gov) — free, no auth.
Updates hourly; point forecasts use the 6-12h gridded model.

For each city we cache:
  - forecast_high_f  (projected daily max)
  - forecast_low_f   (projected daily min)
  - pop_pct          (probability of precipitation 0-100)
  - fetched_at       (monotonic timestamp)

City grid points pre-looked-up to skip the /points lookup latency.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import aiohttp

log = logging.getLogger(__name__)

NWS_BASE = "https://api.weather.gov"

# Pre-resolved NWS grid points for each city.
# Get via: GET https://api.weather.gov/points/{lat},{lon}
# Then read: properties.gridId, properties.gridX, properties.gridY
CITY_GRIDS: dict[str, dict] = {
    "los angeles":     {"office": "LOX", "x": 155, "y": 45,  "zone": "CAZ041"},
    "new york":        {"office": "OKX", "x": 33,  "y": 37,  "zone": "NYZ072"},
    "chicago":         {"office": "LOT", "x": 76,  "y": 73,  "zone": "ILZ011"},
    "miami":           {"office": "MFL", "x": 107, "y": 50,  "zone": "FLZ068"},
    "austin":          {"office": "EWX", "x": 155, "y": 91,  "zone": "TXZ192"},
    "denver":          {"office": "BOU", "x": 57,  "y": 60,  "zone": "COZ040"},
    "boston":          {"office": "BOX", "x": 64,  "y": 53,  "zone": "MAZ015"},
    "atlanta":         {"office": "FFC", "x": 52,  "y": 88,  "zone": "GAZ044"},
    "philadelphia":    {"office": "PHI", "x": 49,  "y": 65,  "zone": "PAZ071"},
    "san francisco":   {"office": "MTR", "x": 84,  "y": 105, "zone": "CAZ006"},
    "minneapolis":     {"office": "MPX", "x": 107, "y": 70,  "zone": "MNZ060"},
    "washington":      {"office": "LWX", "x": 97,  "y": 71,  "zone": "DCZ001"},
    "seattle":         {"office": "SEW", "x": 124, "y": 67,  "zone": "WAZ558"},
    "dallas":          {"office": "FWD", "x": 94,  "y": 85,  "zone": "TXZ231"},
    "oklahoma city":   {"office": "OUN", "x": 103, "y": 90,  "zone": "OKZ063"},
    "phoenix":         {"office": "PSR", "x": 156, "y": 54,  "zone": "AZZ023"},
    "las vegas":       {"office": "VEF", "x": 36,  "y": 52,  "zone": "NVZ023"},
    "houston":         {"office": "HGX", "x": 66,  "y": 98,  "zone": "TXZ163"},
    "san antonio":     {"office": "EWX", "x": 161, "y": 80,  "zone": "TXZ203"},
    "new orleans":     {"office": "LIX", "x": 41,  "y": 89,  "zone": "LAZ063"},
}

# Aliases for matching Kalshi market city names
CITY_ALIASES: dict[str, str] = {
    "nyc": "new york",
    "new york city": "new york",
    "ny": "new york",
    "la": "los angeles",
    "sf": "san francisco",
    "dc": "washington",
    "d.c.": "washington",
    "okc": "oklahoma city",
    "philly": "philadelphia",
}


@dataclass
class CityForecast:
    city: str
    forecast_high_f:  Optional[float]  # daily high °F, None if unavailable
    forecast_low_f:   Optional[float]  # daily low °F
    pop_pct:          Optional[float]  # precipitation probability 0-100
    fetched_at: float = field(default_factory=time.monotonic)


class NOAAClient:
    """
    Fetches today's hourly NWS forecast for each city.
    Parses daily high/low and precip probability from hourly data.
    """

    def __init__(self) -> None:
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "KalshiWeatherBot/1.0 weather-bot@example.com",
                    "Accept": "application/geo+json",
                },
                connector=aiohttp.TCPConnector(ttl_dns_cache=600, keepalive_timeout=60),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_city(self, city: str) -> CityForecast:
        grid = CITY_GRIDS.get(city)
        if not grid:
            return CityForecast(city=city, forecast_high_f=None, forecast_low_f=None, pop_pct=None)

        url = f"{NWS_BASE}/gridpoints/{grid['office']}/{grid['x']},{grid['y']}/forecast/hourly"
        session = await self._get_session()

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    log.warning(f"NWS {city}: HTTP {resp.status}")
                    return CityForecast(city=city, forecast_high_f=None, forecast_low_f=None, pop_pct=None)
                data = await resp.json(content_type=None)
        except Exception as e:
            log.warning(f"NWS {city}: {e}")
            return CityForecast(city=city, forecast_high_f=None, forecast_low_f=None, pop_pct=None)

        periods = data.get("properties", {}).get("periods", [])
        if not periods:
            return CityForecast(city=city, forecast_high_f=None, forecast_low_f=None, pop_pct=None)

        # Gather today's hourly temps (next 24h)
        temps: list[float] = []
        pops:  list[float] = []
        for p in periods[:24]:
            t = p.get("temperature")
            if t is not None:
                temps.append(float(t))
            pop = p.get("probabilityOfPrecipitation", {})
            if isinstance(pop, dict):
                v = pop.get("value")
                if v is not None:
                    pops.append(float(v))

        high = max(temps) if temps else None
        low  = min(temps) if temps else None
        pop  = max(pops)  if pops  else None

        return CityForecast(
            city=city,
            forecast_high_f=high,
            forecast_low_f=low,
            pop_pct=pop,
            fetched_at=time.monotonic(),
        )

    async def fetch_all(self) -> dict[str, CityForecast]:
        """Fetch all cities concurrently. Returns city → CityForecast."""
        cities = list(CITY_GRIDS.keys())
        results = await asyncio.gather(
            *[self.fetch_city(c) for c in cities],
            return_exceptions=True,
        )
        out: dict[str, CityForecast] = {}
        for city, result in zip(cities, results):
            if isinstance(result, CityForecast):
                out[city] = result
            else:
                log.warning(f"NWS {city} error: {result}")
        return out

    def normalize_city(self, name: str) -> str:
        """Normalize a city name from Kalshi market title."""
        lower = name.lower().strip()
        return CITY_ALIASES.get(lower, lower)


if __name__ == "__main__":
    import asyncio

    async def main():
        client = NOAAClient()
        forecasts = await client.fetch_all()
        print(f"\nNWS forecasts for {len(forecasts)} cities:")
        for city, fc in sorted(forecasts.items()):
            if fc.forecast_high_f is not None:
                print(
                    f"  {city:20s}  high={fc.forecast_high_f:.0f}°F  "
                    f"low={fc.forecast_low_f:.0f}°F  "
                    f"pop={fc.pop_pct:.0f}%" if fc.pop_pct is not None
                    else f"  {city:20s}  high={fc.forecast_high_f:.0f}°F  low={fc.forecast_low_f:.0f}°F"
                )
            else:
                print(f"  {city:20s}  no data")
        await client.close()

    asyncio.run(main())

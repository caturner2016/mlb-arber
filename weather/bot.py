"""
Kalshi weather prop trading bot.

Strategy:
  NOAA/NWS updates forecasts roughly every hour. Kalshi weather markets
  often lag 15-60 minutes behind NWS updates. When NWS strongly predicts
  a temperature that falls into a specific Kalshi range, but Kalshi is
  still pricing that range low, we buy YES before the market catches up.

  Example:
    NWS forecasts NYC high = 78°F with high confidence
    Kalshi market "NYC high 75-79°F today" is priced at 40¢ YES
    → Buy YES — it should reprice to 90-99¢

Edge conditions:
  - Temperature: NWS forecast must be within the range AND at least
    MIN_CONFIDENCE_MARGIN degrees from the range boundary
  - Rain: NWS PoP must be ≥ RAIN_BUY_THRESHOLD to buy YES rain market,
    or ≤ RAIN_SELL_THRESHOLD to buy NO (not implemented yet — just YES)
  - Only fire when Kalshi YES ask is ≤ MAX_BUY_CENTS

Polls NWS every POLL_INTERVAL_SEC seconds (NWS updates ~every hour,
so 60s polling is plenty — we're fast either way).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from kalshi.client import KalshiClient
from kalshi.weather_props import WeatherMarket, WeatherPropCache
from weather.noaa import CityForecast, NOAAClient

log = logging.getLogger(__name__)

# Minimum °F distance from range boundary for high-confidence bets
MIN_CONFIDENCE_MARGIN = 1.0

# Rain PoP thresholds
RAIN_BUY_YES_THRESHOLD  = 80   # PoP ≥ 80% → buy YES rain
RAIN_BUY_NO_THRESHOLD   = 15   # PoP ≤ 15% → buy NO rain (= buy YES "no rain")

# Only buy if YES ask ≤ this (market hasn't fully corrected yet)
DEFAULT_MAX_BUY_CENTS = 80  # aggressive — if NWS is near-certain, should hit 95+


@dataclass
class WeatherSignal:
    city: str
    market: WeatherMarket
    forecast: CityForecast
    signal_type: str          # "high_temp_yes", "low_temp_yes", "rain_yes"
    confidence: str           # "high", "medium"
    forecast_value: float     # the actual forecast number
    yes_ask: int              # Kalshi ask at signal time
    detected_at: float = field(default_factory=time.monotonic)


class WeatherBot:
    """
    Main weather trading loop.
    Polls NWS every poll_interval seconds.
    On each poll, compare forecasts to Kalshi markets.
    Fires when NWS gives a strong signal that Kalshi hasn't priced in.
    """

    def __init__(
        self,
        kalshi: KalshiClient,
        weather_cache: WeatherPropCache,
        noaa: NOAAClient,
        cfg: dict,
        tg_fn,
        dry_run: bool = False,
    ) -> None:
        self.kalshi       = kalshi
        self.weather_cache = weather_cache
        self.noaa         = noaa
        self.cfg          = cfg
        self.tg_fn        = tg_fn
        self.dry_run      = dry_run

        wcfg = cfg.get("weather", {})
        self.max_buy_cents = int(wcfg.get("max_buy_cents", DEFAULT_MAX_BUY_CENTS))
        self.max_bet       = float(wcfg.get("max_bet", 10.0))
        self.poll_interval = float(wcfg.get("poll_interval_sec", 60.0))
        self.min_margin    = float(wcfg.get("min_confidence_margin_f", MIN_CONFIDENCE_MARGIN))
        self.rain_yes_thr  = float(wcfg.get("rain_buy_yes_threshold_pct", RAIN_BUY_YES_THRESHOLD))
        self.rain_no_thr   = float(wcfg.get("rain_buy_no_threshold_pct", RAIN_BUY_NO_THRESHOLD))

        self._fired: set[str] = set()  # date:ticker

    def _fire_key(self, ticker: str) -> str:
        return f"{date.today()}:WEATHER:{ticker}"

    def _already_fired(self, ticker: str) -> bool:
        return self._fire_key(ticker) in self._fired

    def _mark_fired(self, ticker: str) -> None:
        self._fired.add(self._fire_key(ticker))

    def _find_signals(self, forecasts: dict[str, CityForecast]) -> list[WeatherSignal]:
        signals: list[WeatherSignal] = []

        for city, fc in forecasts.items():
            # ── HIGH TEMP ────────────────────────────────────────────────────
            if fc.forecast_high_f is not None:
                market = self.weather_cache.find_temp_market(city, "high_temp", fc.forecast_high_f)
                if market and market.yes_ask is not None and market.yes_ask <= self.max_buy_cents:
                    if not self._already_fired(market.ticker):
                        # Check confidence: is forecast comfortably inside the range?
                        lo = market.range_low  if market.range_low  is not None else -999
                        hi = market.range_high if market.range_high is not None else 999
                        margin = min(fc.forecast_high_f - lo, hi - fc.forecast_high_f)
                        confidence = "high" if margin >= self.min_margin * 2 else "medium"
                        signals.append(WeatherSignal(
                            city=city,
                            market=market,
                            forecast=fc,
                            signal_type="high_temp_yes",
                            confidence=confidence,
                            forecast_value=fc.forecast_high_f,
                            yes_ask=market.yes_ask,
                        ))

            # ── LOW TEMP ─────────────────────────────────────────────────────
            if fc.forecast_low_f is not None:
                market = self.weather_cache.find_temp_market(city, "low_temp", fc.forecast_low_f)
                if market and market.yes_ask is not None and market.yes_ask <= self.max_buy_cents:
                    if not self._already_fired(market.ticker):
                        lo = market.range_low  if market.range_low  is not None else -999
                        hi = market.range_high if market.range_high is not None else 999
                        margin = min(fc.forecast_low_f - lo, hi - fc.forecast_low_f)
                        confidence = "high" if margin >= self.min_margin * 2 else "medium"
                        signals.append(WeatherSignal(
                            city=city,
                            market=market,
                            forecast=fc,
                            signal_type="low_temp_yes",
                            confidence=confidence,
                            forecast_value=fc.forecast_low_f,
                            yes_ask=market.yes_ask,
                        ))

            # ── RAIN ─────────────────────────────────────────────────────────
            if fc.pop_pct is not None:
                rain_market = self.weather_cache.find_rain_market(city)
                if rain_market and rain_market.yes_ask is not None:
                    if fc.pop_pct >= self.rain_yes_thr and rain_market.yes_ask <= self.max_buy_cents:
                        if not self._already_fired(rain_market.ticker):
                            signals.append(WeatherSignal(
                                city=city,
                                market=rain_market,
                                forecast=fc,
                                signal_type="rain_yes",
                                confidence="high" if fc.pop_pct >= 90 else "medium",
                                forecast_value=fc.pop_pct,
                                yes_ask=rain_market.yes_ask,
                            ))

        return signals

    async def _execute_signal(self, signal: WeatherSignal) -> None:
        ticker  = signal.market.ticker
        title   = signal.market.title
        city    = signal.city.title()
        yes_ask = signal.yes_ask

        if yes_ask > self.max_buy_cents:
            log.info(f"  [WEATHER] {city} {signal.signal_type} — ask {yes_ask}¢ > max {self.max_buy_cents}¢")
            return

        # Re-fetch live ask price
        result = await self.kalshi.get_yes_ask(ticker, max_cents=self.max_buy_cents)
        if result is None:
            log.info(f"  [WEATHER] {city} {signal.signal_type} — market already repriced")
            self.tg_fn(
                f"MISS [WEATHER {city}]: {title}\n"
                f"NWS: {signal.signal_type} = {signal.forecast_value:.1f}\n"
                f"Market already repriced above {self.max_buy_cents}¢\n{ticker}"
            )
            return

        live_ask, qty = result
        balance = await self.kalshi.get_balance() if not self.dry_run else self.max_bet
        spend   = min(balance, self.max_bet)
        count   = min(qty, int(spend / (live_ask / 100)))

        if count < 1:
            log.info(f"  [WEATHER] {city} — insufficient balance ${balance:.2f}")
            return

        cost       = count * live_ask / 100
        expected   = count * (100 - live_ask) / 100
        latency_ms = (time.monotonic() - signal.detected_at) * 1000

        label = f"WEATHER {signal.signal_type.upper()}"
        log.info(
            f"  BUY [{label}] {city}  {count} @ {live_ask}¢  "
            f"cost=${cost:.2f}  conf={signal.confidence}  latency={latency_ms:.0f}ms"
        )

        if not self.dry_run:
            await self.kalshi.place_order(ticker, "yes", live_ask, count, "limit")

        self._mark_fired(ticker)

        self.tg_fn(
            f"BUY [{label}] — {city}\n"
            f"Market: {title}\n"
            f"NWS forecast: {signal.signal_type} = {signal.forecast_value:.1f}"
            f"{'°F' if 'temp' in signal.signal_type else '%'}\n"
            f"Confidence: {signal.confidence}\n"
            f"Contracts: {count} @ {live_ask}¢\n"
            f"Cost: ${cost:.2f}  |  Expected: +${expected:.2f}\n"
            f"Hold to $1.00 settlement\n"
            f"{'[DRY RUN]' if self.dry_run else '[LIVE]'}"
        )

    async def run_once(self) -> None:
        """Single poll cycle: fetch NWS, find signals, execute."""
        log.debug("Weather bot: fetching NWS forecasts…")
        forecasts = await self.noaa.fetch_all()

        signals = self._find_signals(forecasts)
        if signals:
            log.info(f"Weather bot: {len(signals)} signal(s) found")
        for s in signals:
            log.info(
                f"  Signal: {s.city.title():20s}  {s.signal_type:15s}  "
                f"NWS={s.forecast_value:.1f}  Kalshi={s.yes_ask}¢  conf={s.confidence}"
            )
            await self._execute_signal(s)

    async def run(self) -> None:
        """Continuous polling loop."""
        log.info(
            f"Weather bot started  poll={self.poll_interval}s  "
            f"max_buy={self.max_buy_cents}¢  max_bet=${self.max_bet}"
        )
        while True:
            cycle_start = time.monotonic()
            try:
                await self.run_once()
            except Exception as e:
                log.warning(f"Weather bot cycle error: {e}")

            elapsed   = time.monotonic() - cycle_start
            sleep_for = max(0.0, self.poll_interval - elapsed)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    import asyncio, logging, yaml
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s")

    async def main():
        cfg    = yaml.safe_load(open("config.yaml"))
        kalshi = KalshiClient(
            key_id=cfg.get("kalshi_key_id", ""),
            private_key_path=cfg.get("kalshi_private_key_path", ""),
            paper_mode=True,
        )
        weather_cache = WeatherPropCache()
        noaa          = NOAAClient()

        await weather_cache.build(kalshi)
        bot = WeatherBot(kalshi, weather_cache, noaa, cfg, lambda msg: print(f"TG: {msg}"), dry_run=True)

        print(f"\nWeather markets loaded: {len(weather_cache)}")
        print("Running single forecast cycle…\n")
        await bot.run_once()

        await kalshi.close()
        await noaa.close()

    asyncio.run(main())

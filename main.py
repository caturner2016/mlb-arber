"""
MLB + NBA + Weather Arber — Kalshi prop latency bot.

MLB:     HR and hits markets — fires when event confirmed in MLB Stats API feed
NBA:     Points markets — fires when stat confirmed in NBA CDN feed
Weather: Temperature/rain markets — fires when NWS forecast diverges from Kalshi price

Run:
  py main.py                  # paper mode (MLB + NBA + Weather)
  py main.py --live           # live trading
  py main.py --dry-run        # orders skipped (logs + Telegram only)
  py main.py --nba-only       # skip MLB and Weather
  py main.py --mlb-only       # skip NBA and Weather
  py main.py --weather-only   # skip MLB and NBA
  py main.py --no-weather     # MLB + NBA, no weather
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import date, datetime, timedelta

import requests
import yaml

from execution.paper import PaperTrader
from kalshi.client import KalshiClient
from kalshi.nba_props import NBAPropCache
from kalshi.props import PropCache
from kalshi.weather_props import WeatherPropCache
from mlb.feed import PropEvent, stream_all_games
from mlb.games import get_todays_games
from nba.feed import NBAStatEvent, stream_nba_games
from nba.games import get_todays_nba_games
from weather.bot import WeatherBot
from weather.noaa import NOAAClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("kalshi_bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("arber")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Telegram ───────────────────────────────────────────────────────────────────

def tg(token: str, chat_id: str, msg: str) -> None:
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg},
            timeout=5,
        )
    except Exception as e:
        log.warning(f"Telegram: {e}")


# ── Generic buy + hold ─────────────────────────────────────────────────────────

async def buy_and_hold(
    ticker: str,
    player_name: str,
    label: str,
    max_buy_cents: int,
    max_bet: float,
    kalshi: KalshiClient,
    tg_fn,
    dry_run: bool,
    detected_at: float,
) -> None:
    """
    Buy YES contracts up to max_bet, hold to $1.00 settlement.
    Used for all NBA props (stat corrections make flipping risky).
    """
    result = await kalshi.get_yes_ask(ticker, max_cents=max_buy_cents)
    if result is None:
        log.info(f"  [{label}] market above {max_buy_cents}¢ — already repriced")
        tg_fn(f"MISS [{label}]: {player_name}\nMarket already repriced\n{ticker}")
        return

    yes_cents, qty = result
    balance = await kalshi.get_balance() if not dry_run else max_bet
    spend   = min(balance, max_bet)
    count   = min(qty, int(spend / (yes_cents / 100)))

    if count < 1:
        log.info(f"  [{label}] insufficient balance ${balance:.2f}")
        return

    cost       = count * yes_cents / 100
    expected   = count * (100 - yes_cents) / 100
    latency_ms = (time.monotonic() - detected_at) * 1000
    ts         = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    log.info(f"  BUY [{label}] {count} @ {yes_cents}¢  cost=${cost:.2f}  latency={latency_ms:.0f}ms")

    if not dry_run:
        await kalshi.place_order(ticker, "yes", yes_cents, count, "limit")

    tg_fn(
        f"BUY [{label}] — {player_name}\n"
        f"Ticker: {ticker}\n"
        f"Contracts: {count} @ {yes_cents}¢\n"
        f"Cost: ${cost:.2f}  |  Expected: +${expected:.2f}\n"
        f"Hold to $1.00 settlement\n"
        f"Latency: {latency_ms:.0f}ms\n"
        f"{'[DRY RUN]' if dry_run else '[LIVE]'}"
    )


# ── MLB execution ──────────────────────────────────────────────────────────────

async def execute_mlb(
    event: PropEvent,
    cache: PropCache,
    kalshi: KalshiClient,
    cfg: dict,
    tg_fn,
    fired: set,
    dry_run: bool,
) -> None:
    home_cfg = cfg.get("home", cfg)
    max_buy  = home_cfg.get("max_buy_cents", 99)
    max_bet  = float(home_cfg.get("max_hr_bet", 5.0))

    if event.event_type == "home_run":
        key = f"{date.today()}:HR:1:{event.player_name}"
        if key in fired:
            return
        fired.add(key)
        prop = cache.get_hr_ticker(event.player_name)
        if not prop:
            log.info(f"[MLB HR] {event.player_name} — no market")
            return
        log.info(f"[MLB HR] {event.player_name}  inn={event.inning}{event.half[0].upper()}")
        await buy_and_hold(prop.ticker, event.player_name, f"MLB HR",
                           max_buy, max_bet, kalshi, tg_fn, dry_run, event.feed_detected_at)

    elif event.event_type == "hit":
        for threshold in (1, 2, 3):
            key = f"{date.today()}:HIT:{threshold}:{event.player_name}"
            if key in fired:
                continue
            prop = cache.get_hit_ticker(event.player_name, threshold)
            if not prop:
                fired.add(key)
                continue
            log.info(f"[MLB HIT {threshold}+] {event.player_name}  inn={event.inning}{event.half[0].upper()}")
            max_hit_bet = float(home_cfg.get("max_hits_bet", 5.0))
            await buy_and_hold(prop.ticker, event.player_name, f"MLB HIT {threshold}+",
                               max_buy, max_hit_bet, kalshi, tg_fn, dry_run, event.feed_detected_at)
            fired.add(key)


# ── NBA execution ──────────────────────────────────────────────────────────────

async def execute_nba(
    event: NBAStatEvent,
    cache: NBAPropCache,
    kalshi: KalshiClient,
    cfg: dict,
    tg_fn,
    fired: set,
    dry_run: bool,
) -> None:
    key = f"{date.today()}:NBA:{event.stat_type}:{event.threshold}:{event.player_name}"
    if key in fired:
        return
    fired.add(key)

    nba_cfg = cfg.get("nba", cfg.get("home", cfg))
    max_buy = int(nba_cfg.get("max_buy_cents", 99))
    max_bet = float(nba_cfg.get("max_bet", nba_cfg.get("max_hr_bet", 5.0)))

    prop = cache.lookup(event.player_name, event.stat_type, event.threshold)
    if not prop:
        log.info(f"[NBA] {event.player_name} {event.stat_type} {event.threshold}+ — no Kalshi market")
        return

    label = f"NBA {event.stat_type.upper()} {event.threshold}+"
    log.info(
        f"[NBA] {event.player_name}  {event.stat_type} {event.threshold}+  "
        f"(has {event.current_value})  Q{event.period} {event.clock}"
    )
    await buy_and_hold(prop.ticker, event.player_name, label,
                       max_buy, max_bet, kalshi, tg_fn, dry_run, event.feed_detected_at)


# ── Background loops ───────────────────────────────────────────────────────────

async def cache_refresh_loop(mlb_cache: PropCache, nba_cache: NBAPropCache,
                              weather_cache: WeatherPropCache,
                              kalshi: KalshiClient,
                              mlb: bool, nba: bool, weather: bool) -> None:
    while True:
        await asyncio.sleep(1800)
        try:
            await asyncio.gather(
                mlb_cache.build(kalshi)     if mlb     else asyncio.sleep(0),
                nba_cache.build(kalshi)     if nba     else asyncio.sleep(0),
                weather_cache.build(kalshi) if weather else asyncio.sleep(0),
            )
            log.info("Caches refreshed")
        except Exception as e:
            log.warning(f"Cache refresh error: {e}")


async def daily_summary_loop(tg_fn, kalshi: KalshiClient) -> None:
    while True:
        now    = datetime.now()
        target = now.replace(hour=23, minute=59, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        balance = await kalshi.get_balance()
        tg_fn(f"Daily Summary {date.today()}\nBalance: ${balance:.2f}")


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(cfg: dict, dry_run: bool, mlb: bool, nba: bool, weather: bool) -> None:
    mode  = cfg.get("mode", "paper")
    paper = (mode != "live") or dry_run

    tg_token = cfg.get("telegram_token", "")
    tg_chat  = cfg.get("telegram_chat_id", "")
    tg_fn    = lambda msg: tg(tg_token, tg_chat, msg)

    log.info(f"Arber  mode={'DRY RUN' if dry_run else mode}  MLB={mlb}  NBA={nba}  Weather={weather}")

    kalshi = KalshiClient(
        key_id=cfg.get("kalshi_key_id", ""),
        private_key_path=cfg.get("kalshi_private_key_path", ""),
        paper_mode=paper,
    )

    # ── Build caches ──────────────────────────────────────────────────────────
    mlb_cache     = PropCache()
    nba_cache     = NBAPropCache()
    weather_cache = WeatherPropCache()
    noaa          = NOAAClient()

    await asyncio.gather(
        mlb_cache.build(kalshi)     if mlb     else asyncio.sleep(0),
        nba_cache.build(kalshi)     if nba     else asyncio.sleep(0),
        weather_cache.build(kalshi) if weather else asyncio.sleep(0),
    )

    nba_cfg     = cfg.get("nba", {})
    weather_cfg = cfg.get("weather", {})
    tg_fn(
        f"Bot started ({'DRY RUN' if dry_run else mode.upper()})\n"
        f"MLB: {len(mlb_cache.hr_cache)} HR + {len(mlb_cache.hits_cache)} hits markets\n"
        f"NBA: {len(nba_cache)} prop markets\n"
        f"Weather: {len(weather_cache)} markets\n"
        f"NBA max bet: ${nba_cfg.get('max_bet', 5)}  "
        f"Weather max bet: ${weather_cfg.get('max_bet', 10)}"
    )

    asyncio.create_task(cache_refresh_loop(mlb_cache, nba_cache, weather_cache, kalshi, mlb, nba, weather))
    asyncio.create_task(daily_summary_loop(tg_fn, kalshi))

    fired: set[str] = set()

    # ── MLB stream ────────────────────────────────────────────────────────────
    async def run_mlb():
        games    = get_todays_games()
        game_pks = [g.game_pk for g in games]
        log.info(f"MLB: watching {len(games)} games")
        for g in games:
            log.info(f"  [{g.game_pk}] {g.away_team} @ {g.home_team}  ({g.status})")
        poll = float(cfg.get("poll_interval_sec", 0.5))
        async for event in stream_all_games(game_pks, poll_interval=poll):
            await execute_mlb(event, mlb_cache, kalshi, cfg, tg_fn, fired, dry_run)

    # ── NBA stream ────────────────────────────────────────────────────────────
    async def run_nba():
        games    = get_todays_nba_games()
        game_ids = [g.game_id for g in games]
        log.info(f"NBA: watching {len(games)} games")
        for g in games:
            log.info(f"  [{g.game_id}] {g.away_team} @ {g.home_team}  {g.status}")
        async for event in stream_nba_games(game_ids, poll_interval=1.0):
            await execute_nba(event, nba_cache, kalshi, cfg, tg_fn, fired, dry_run)

    # ── Weather stream ────────────────────────────────────────────────────────
    async def run_weather():
        weather_bot = WeatherBot(
            kalshi=kalshi,
            weather_cache=weather_cache,
            noaa=noaa,
            cfg=cfg,
            tg_fn=tg_fn,
            dry_run=dry_run,
        )
        await weather_bot.run()

    tasks = []
    if mlb:
        tasks.append(asyncio.create_task(run_mlb()))
    if nba:
        tasks.append(asyncio.create_task(run_nba()))
    if weather:
        tasks.append(asyncio.create_task(run_weather()))

    if not tasks:
        log.error("No sport/market selected — specify at least one of --mlb-only/--nba-only/--weather-only")
        return

    await asyncio.gather(*tasks)
    await kalshi.close()
    await noaa.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB + NBA + Weather Kalshi prop latency bot")
    parser.add_argument("--live",          action="store_true")
    parser.add_argument("--dry-run",       action="store_true")
    parser.add_argument("--mlb-only",      action="store_true")
    parser.add_argument("--nba-only",      action="store_true")
    parser.add_argument("--weather-only",  action="store_true")
    parser.add_argument("--no-weather",    action="store_true")
    parser.add_argument("--config",        default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.live:
        cfg["mode"] = "live"

    if args.weather_only:
        mlb, nba, weather = False, False, True
    elif args.mlb_only:
        mlb, nba, weather = True, False, False
    elif args.nba_only:
        mlb, nba, weather = False, True, False
    else:
        mlb     = True
        nba     = True
        weather = not args.no_weather

    try:
        asyncio.run(run(cfg, dry_run=args.dry_run, mlb=mlb, nba=nba, weather=weather))
    except KeyboardInterrupt:
        log.info("Stopped")


if __name__ == "__main__":
    main()

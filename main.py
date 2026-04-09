"""
MLB Arber — Kalshi HR + Hits prop latency bot.

Strategy: confirm event via MLB Stats API feed → buy YES on Kalshi before
the market reprices to ~$1.00.

  HR markets  (KXMLBHR):  buy at ask <= 96c, flip at 98c
  Hits markets (KXMLBHIT): buy at 99c IOC, hold to $1.00 settlement

Run:
  py main.py              # paper mode
  py main.py --live       # live trading
  py main.py --dry-run    # live mode but orders are skipped (logs only)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import date, datetime, timedelta

import aiohttp
import requests
import yaml

from execution.paper import PaperTrader
from kalshi.client import KalshiClient
from kalshi.props import PropCache
from mlb.feed import PropEvent, stream_all_games
from mlb.games import get_todays_games


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("kalshi_bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("mlb-arber")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Telegram ──────────────────────────────────────────────────────────────────

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


# ── Trade execution ────────────────────────────────────────────────────────────

async def execute_hr(
    event: PropEvent,
    prop,
    kalshi: KalshiClient,
    cfg: dict,
    tg_fn,
    dry_run: bool = False,
) -> None:
    """Buy HR market YES, then flip at sell_cents."""
    max_buy = cfg["max_buy_cents"]
    sell_at = cfg["sell_cents"]
    max_bet = cfg["max_hr_bet"]

    result = await kalshi.get_yes_ask(prop.ticker, max_cents=max_buy)
    if result is None:
        log.info(f"  HR: no asks <= {max_buy}c for {prop.ticker} — market already repriced")
        tg_fn(f"HR MISS: {event.player_name}\nMarket already repriced\n{prop.ticker}")
        return

    yes_cents, qty = result
    balance = await kalshi.get_balance() if not dry_run else max_bet
    spend   = min(balance, max_bet)
    count   = min(qty, int(spend / (yes_cents / 100)))

    if count < 1:
        log.info(f"  HR: insufficient balance (${balance:.2f})")
        return

    cost           = count * yes_cents / 100
    detect_str     = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    latency_ms     = (time.monotonic() - event.feed_detected_at) * 1000

    log.info(
        f"  HR BUY: {count} contracts @ {yes_cents}¢  "
        f"cost=${cost:.2f}  latency={latency_ms:.0f}ms"
    )

    if not dry_run:
        order = await kalshi.place_order(prop.ticker, "yes", yes_cents, count, "limit")
        filled_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    else:
        order = {"dry_run": True}
        filled_str = detect_str

    expected = count * (sell_at - yes_cents) / 100
    tg_fn(
        f"BUY HR — {event.player_name}\n"
        f"Ticker: {prop.ticker}\n"
        f"Contracts: {count} @ {yes_cents}¢\n"
        f"Cost: ${cost:.2f}\n"
        f"Flip target: {sell_at}¢  (+${expected:.2f})\n"
        f"Detected: {detect_str}\n"
        f"Filled:   {filled_str}\n"
        f"Latency: {latency_ms:.0f}ms\n"
        f"{'[DRY RUN]' if dry_run else ''}"
    )

    if dry_run:
        return

    # Flip: sell at sell_cents in background
    async def flip() -> None:
        deadline = asyncio.get_event_loop().time() + 300
        while asyncio.get_event_loop().time() < deadline:
            try:
                await kalshi.place_order(prop.ticker, "yes", sell_at, count, "limit")
                profit = count * (sell_at - yes_cents) / 100
                log.info(f"  HR SOLD @ {sell_at}¢  profit=${profit:.2f}")
                tg_fn(
                    f"SOLD HR — {event.player_name}\n"
                    f"Sell: {sell_at}¢  profit=${profit:.2f}"
                )
                return
            except Exception as e:
                log.warning(f"  Sell retry: {e}")
                await asyncio.sleep(3)
        log.warning("  Sell timed out — holding to settlement")

    asyncio.create_task(flip())


async def execute_hit(
    event: PropEvent,
    prop,
    kalshi: KalshiClient,
    cfg: dict,
    tg_fn,
    dry_run: bool = False,
) -> None:
    """Buy hits market YES at 99c IOC — hold to $1.00 settlement."""
    buy_cents = cfg["hits_buy_cents"]
    max_bet   = cfg["max_hits_bet"]

    # Cancel any resting orders first
    if not dry_run:
        await kalshi.cancel_resting_orders()

    balance = await kalshi.get_balance() if not dry_run else max_bet
    spend   = min(balance, max_bet)
    count   = int(spend / (buy_cents / 100))

    if count < 1:
        log.info(f"  HIT: insufficient balance (${balance:.2f})")
        return

    cost       = count * buy_cents / 100
    detect_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    latency_ms = (time.monotonic() - event.feed_detected_at) * 1000

    log.info(
        f"  HIT BUY ({prop.threshold}+): {count} @ {buy_cents}¢  "
        f"cost=${cost:.2f}  latency={latency_ms:.0f}ms"
    )

    if not dry_run:
        await kalshi.place_order(prop.ticker, "yes", buy_cents, count, "ioc")

    expected = count * (100 - buy_cents) / 100
    tg_fn(
        f"BUY HIT {prop.threshold}+ — {event.player_name}\n"
        f"Ticker: {prop.ticker}\n"
        f"Contracts: {count} @ {buy_cents}¢\n"
        f"Cost: ${cost:.2f}  |  Expected: +${expected:.2f}\n"
        f"Detected: {detect_str}\n"
        f"Latency: {latency_ms:.0f}ms\n"
        f"{'[DRY RUN]' if dry_run else ''}"
    )


# ── Cache refresh loop ─────────────────────────────────────────────────────────

async def cache_refresh_loop(cache: PropCache, kalshi: KalshiClient) -> None:
    """Rebuild prop cache every 30 minutes."""
    while True:
        await asyncio.sleep(1800)
        try:
            await cache.build(kalshi)
            log.info("Cache refreshed")
        except Exception as e:
            log.warning(f"Cache refresh error: {e}")


# ── Daily summary ──────────────────────────────────────────────────────────────

async def daily_summary_loop(tg_fn, kalshi: KalshiClient, trader: PaperTrader) -> None:
    while True:
        now    = datetime.now()
        target = now.replace(hour=23, minute=59, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        await asyncio.sleep((target - now).total_seconds())
        balance = await kalshi.get_balance()
        trader.summary()
        tg_fn(
            f"Daily Summary {date.today()}\n"
            f"Balance: ${balance:.2f}\n"
            f"Trades: {len(trader._trades)}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

async def run(cfg: dict, dry_run: bool = False) -> None:
    mode  = cfg.get("mode", "paper")
    live  = (mode == "live") and not dry_run
    paper = not live

    tg_token   = cfg.get("telegram_token", "")
    tg_chat    = cfg.get("telegram_chat_id", "")
    tg_fn      = lambda msg: tg(tg_token, tg_chat, msg)
    poll_secs  = float(cfg.get("poll_interval_sec", 0.5))

    log.info(f"MLB Arber  mode={'DRY RUN' if dry_run else mode}  poll={poll_secs}s")

    kalshi = KalshiClient(
        key_id=cfg.get("kalshi_key_id", ""),
        private_key_path=cfg.get("kalshi_private_key_path", ""),
        paper_mode=paper,
    )
    trader = PaperTrader(
        max_trade_usd=max(cfg.get("max_hr_bet", 2), cfg.get("max_hits_bet", 2)),
        min_edge_cents=100 - cfg.get("max_buy_cents", 96),
    )

    # ── Build prop cache ──────────────────────────────────────────────────────
    cache = PropCache()
    await cache.build(kalshi)

    if not cache.hr_cache and not cache.hits_cache:
        log.warning("No Kalshi prop markets found — check credentials or series tickers")
        tg_fn("WARNING: No Kalshi prop markets found")
    else:
        msg = (
            f"Bot started ({'DRY RUN' if dry_run else mode.upper()})\n"
            f"HR markets: {len(cache.hr_cache)}\n"
            f"Hits markets: {len(cache.hits_cache)}\n"
            f"Poll: {poll_secs}s"
        )
        log.info(msg)
        tg_fn(msg)

    # ── Background tasks ──────────────────────────────────────────────────────
    asyncio.create_task(cache_refresh_loop(cache, kalshi))
    asyncio.create_task(daily_summary_loop(tg_fn, kalshi, trader))

    # ── Today's games ─────────────────────────────────────────────────────────
    games = get_todays_games()
    game_pks = [g.game_pk for g in games]
    log.info(f"Watching {len(games)} games today")
    for g in games:
        log.info(f"  [{g.game_pk}] {g.away_team} @ {g.home_team}  ({g.status})")

    # ── Event loop ────────────────────────────────────────────────────────────
    fired: set[str] = set()   # "DATE:TYPE:THRESHOLD:PLAYER" — prevent double-fire

    async for event in stream_all_games(game_pks, poll_interval=poll_secs):
        detected_ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        # ── Home run ─────────────────────────────────────────────────────────
        if event.event_type == "home_run":
            key = f"{date.today()}:HR:1:{event.player_name}"
            if key in fired:
                continue
            fired.add(key)

            prop = cache.get_hr_ticker(event.player_name)
            if not prop:
                log.info(f"[HR] {event.player_name} — no Kalshi market found")
                tg_fn(f"HR (no market): {event.player_name}\nInning {event.inning}")
                continue

            log.info(
                f"[HR] {event.player_name}  inning={event.inning}{event.half[0].upper()}  "
                f"score={event.away_score}-{event.home_score}"
            )
            await execute_hr(event, prop, kalshi, cfg, tg_fn, dry_run)

        # ── Hit — fire at 1+, 2+, 3+ thresholds ─────────────────────────────
        elif event.event_type == "hit":
            for threshold in (1, 2, 3):
                key = f"{date.today()}:HIT:{threshold}:{event.player_name}"
                if key in fired:
                    continue

                prop = cache.get_hit_ticker(event.player_name, threshold)
                if not prop:
                    fired.add(key)   # no market — skip permanently
                    continue

                log.info(
                    f"[HIT {threshold}+] {event.player_name}  "
                    f"inning={event.inning}{event.half[0].upper()}"
                )
                await execute_hit(event, prop, kalshi, cfg, tg_fn, dry_run)
                fired.add(key)

        else:
            log.debug(
                f"[{event.game_pk}] {event.raw_event:20s} {event.player_name} "
                f"inn={event.inning}{event.half[0].upper()}"
            )

    await kalshi.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB Arber — Kalshi prop latency bot")
    parser.add_argument("--live",    action="store_true", help="Live trading mode")
    parser.add_argument("--dry-run", action="store_true", help="Live mode but skip order submission")
    parser.add_argument("--config",  default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.live:
        cfg["mode"] = "live"

    try:
        asyncio.run(run(cfg, dry_run=args.dry_run))
    except KeyboardInterrupt:
        log.info("Stopped")


if __name__ == "__main__":
    main()

"""
MLB Arber — Kalshi prop settlement latency bot.

Strategy:
  1. Load today's Kalshi MLB player prop markets (HR, hits, strikeouts, etc.)
  2. Poll every live MLB game as fast as possible via MLB Stats API
  3. The moment a confirmed event fires (e.g. "Home Run" by Aaron Judge):
       a. Immediately fetch the Kalshi orderbook for that prop
       b. If yes_ask is still << 99¢ (market hasn't caught up), execute
  4. In paper mode: log to trades.csv. In live mode: submit order.

Run:
  python main.py              # paper mode (default)
  python main.py --live       # live trading (requires Kalshi auth)
  python main.py --date 2026-04-09  # specific date (for testing)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from datetime import date
from pathlib import Path

import aiohttp
import yaml

from execution.live import LiveTrader
from execution.paper import PaperTrader
from kalshi.client import KalshiClient
from kalshi.props import build_trigger_map, get_todays_props
from mlb.feed import PropEvent, stream_all_games
from mlb.games import get_todays_games
from mlb.players import get_player_id_map


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mlb-arber")


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


async def run(cfg: dict, target_date: date | None = None) -> None:
    mode = cfg.get("mode", "paper")
    poll_interval = float(cfg.get("poll_interval_sec", 1.0))
    max_trade_usd = float(cfg.get("max_trade_usd", 500))
    min_edge_cents = int(cfg.get("min_edge_cents", 5))
    daily_loss_limit = float(cfg.get("daily_loss_limit", 100))

    log.info(f"Starting MLB Arber  mode={mode}  poll={poll_interval}s  max_trade=${max_trade_usd}")

    # ── Kalshi client ────────────────────────────────────────────────────────
    kalshi = KalshiClient(
        key_id=cfg.get("kalshi_key_id", ""),
        private_key_path=cfg.get("kalshi_private_key_path", ""),
        paper_mode=(mode != "live"),
    )

    # ── Execution layer ──────────────────────────────────────────────────────
    if mode == "live":
        trader: PaperTrader = LiveTrader(kalshi, max_trade_usd, min_edge_cents)
        log.warning("LIVE MODE — real orders will be submitted to Kalshi")
    else:
        trader = PaperTrader(max_trade_usd, min_edge_cents)
        log.info("Paper mode — no real orders will be placed")

    # ── Today's games ────────────────────────────────────────────────────────
    log.info("Fetching today's MLB schedule…")
    games = get_todays_games(target_date)
    if not games:
        log.error("No games found today. Exiting.")
        return

    game_pks = [g.game_pk for g in games]
    log.info(f"Found {len(games)} games: {[g.game_pk for g in games]}")
    for g in games:
        log.info(f"  [{g.game_pk}] {g.away_team} @ {g.home_team}  ({g.status})")

    # ── Player ID map ────────────────────────────────────────────────────────
    log.info("Loading MLB player roster…")
    player_id_map = get_player_id_map()
    log.info(f"Loaded {len(player_id_map)} player name entries")

    # ── Kalshi prop markets ──────────────────────────────────────────────────
    log.info("Fetching Kalshi MLB prop markets…")
    connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        props = await get_todays_props(session, target_date)

    if not props:
        log.warning("No Kalshi MLB prop markets found today. Check Kalshi series tickers.")
    else:
        log.info(f"Found {len(props)} Kalshi prop markets")
        for p in props[:10]:
            log.info(f"  [{p.ticker}] {p.player_name} — {p.prop_type} (ask={p.yes_ask}¢)")
        if len(props) > 10:
            log.info(f"  … and {len(props) - 10} more")

    # ── Build trigger map ────────────────────────────────────────────────────
    trigger_map = await build_trigger_map(props, player_id_map)
    log.info(f"Trigger map: {len(trigger_map)} (player_id, prop_type) pairs armed")

    if not trigger_map:
        log.warning("No trigger mappings resolved. Check player name matching.")

    # ── Main event loop ──────────────────────────────────────────────────────
    log.info(f"Streaming live events at {poll_interval}s interval — waiting for triggers…\n")
    fired: set[tuple[int, str]] = set()  # prevent double-firing

    async for event in stream_all_games(game_pks, poll_interval=poll_interval):
        key = (event.player_id, event.event_type)

        if key not in trigger_map:
            # Log non-prop events at DEBUG level for visibility
            log.debug(
                f"[{event.game_pk}] {event.raw_event:20s} "
                f"batter={event.player_id} inning={event.inning}{event.half[0].upper()}"
            )
            continue

        if key in fired:
            continue  # already executed this prop today

        prop = trigger_map[key]
        log.info(
            f"[TRIGGER] {prop.player_name} — {event.raw_event}  "
            f"inning={event.inning}{event.half[0].upper()}  "
            f"score={event.away_score}-{event.home_score}"
        )

        # ── HOT PATH: fetch orderbook as fast as possible ────────────────
        orderbook = await kalshi.get_orderbook(prop.ticker)
        latency_ms = (orderbook.fetched_at - event.feed_detected_at) * 1000
        log.info(
            f"  Orderbook: yes_ask={orderbook.yes_ask}¢  latency={latency_ms:.0f}ms"
        )

        # ── Execute (paper or live) ───────────────────────────────────────
        if mode == "live":
            record = await trader.execute_live(event, prop, orderbook)  # type: ignore[attr-defined]
        else:
            record = trader.execute(event, prop, orderbook)

        if record:
            fired.add(key)  # each prop fires once (event already happened)
        else:
            log.info("  Skipped — market already repriced (no edge)")

        # ── Daily loss guard ─────────────────────────────────────────────
        # (Paper mode: no real losses, but track virtual)
        if len(fired) >= 1 and -sum(
            t.cost_usd for t in trader._trades
        ) < -daily_loss_limit:
            log.warning("Daily loss limit hit. Halting.")
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="MLB Arber — Kalshi prop latency bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.live:
        cfg["mode"] = "live"

    target_date = None
    if args.date:
        target_date = date.fromisoformat(args.date)

    # Graceful shutdown on Ctrl+C
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run() -> None:
        try:
            await run(cfg, target_date)
        except KeyboardInterrupt:
            pass

    try:
        loop.run_until_complete(_run())
    finally:
        # Print summary on exit
        loop.close()


if __name__ == "__main__":
    main()

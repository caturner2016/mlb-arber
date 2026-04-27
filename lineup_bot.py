"""
Lineup Arb Bot

Strategy:
  1. Watch live batting order for all games (innings 1-5 only)
  2. Identify batters 3-4 spots away from batting
  3. Buy YES on their 1+ hit prop at the current low price (~65-78¢)
  4. Place an immediate limit sell at the target price (default 91¢)
  5. When the batter steps in, market jumps to ~93¢ and sell fills

Why it works:
  The Kalshi hit market reprices upward the moment a batter steps
  to the plate. Being 3-4 spots ahead gives ~1-2 minutes to get in
  at the pre-at-bat price.

Usage:
  py lineup_bot.py              paper mode (no real trades)
  py lineup_bot.py --live       live mode (real money)
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import date

import aiohttp
import yaml

from kalshi.client import KalshiClient
from kalshi.props import PropCache
from mlb.games import get_todays_games
from mlb.lineup import get_batting_state

log = logging.getLogger("lineup_bot")


# ── Config ─────────────────────────────────────────────────────────────────────

MAX_BUY_CENTS  = 94   # don't buy if less than 5¢ room to 99¢
MIN_PROFIT_CENTS = 5   # sell when price rises ≥ 5¢ from buy price
MAX_INNING     = 5    # stop buying after this inning
LOOKAHEAD      = [3, 4]  # spots ahead to target
POLL_SEC       = 3    # how often to poll each game feed
MAX_SPEND_USD  = 5.0  # max $ per trade


# ── Position tracker ───────────────────────────────────────────────────────────

FILL_DELAY_SEC = 3   # simulate real order latency in paper mode


@dataclass
class OpenPosition:
    ticker:          str
    player:          str
    player_id:       int
    contracts:       int
    buy_price:       int          # price we saw when signal fired
    actual_fill:     int  = 0    # price available after fill delay (paper) or real fill (live)
    ab_started:      bool = False
    sold:            bool = False
    fill_confirmed:  bool = False # paper: waiting for delayed fill check


class LineupBot:
    def __init__(self, kalshi: KalshiClient, cache: PropCache, live: bool):
        self.kalshi    = kalshi
        self.cache     = cache
        self.live      = live
        self.paper     = not live
        self.positions: dict[str, OpenPosition] = {}   # ticker → position
        self.bought:    set[str] = set()               # tickers we've ever bought today
        self._id_name:  dict[int, str] = {}            # player_id → full name (shared cache)
        self._game_states: dict[int, int] = {}         # game_pk → current_batter_id

    async def run(self) -> None:
        log.info(f"Lineup bot started — {'LIVE' if self.live else 'PAPER'} mode")
        log.info(f"  Buy ≤ {MAX_BUY_CENTS}¢ | min profit {MIN_PROFIT_CENTS}¢ | Innings 1-{MAX_INNING}")

        async with aiohttp.ClientSession() as session:
            while True:
                games = get_todays_games()
                live_games = [g for g in games if g.status == "Live"]

                states = await asyncio.gather(
                    *[get_batting_state(g.game_pk, session, self._id_name) for g in live_games],
                    return_exceptions=True,
                )

                current_batters: set[int] = set()
                for state in states:
                    if isinstance(state, Exception) or state is None:
                        continue
                    self._game_states[state.game_pk] = state.current_batter_id
                    current_batters.add(state.current_batter_id)
                    if state.inning <= MAX_INNING:
                        await self._process_state(state)

                await self._check_sells(current_batters)
                await asyncio.sleep(POLL_SEC)

    async def _process_state(self, state) -> None:
        balance = await self.kalshi.get_balance() if not self.paper else 500.0

        for offset in LOOKAHEAD:
            player_id = state.upcoming(offset)
            if not player_id:
                continue

            player_name = state.name(player_id)
            if not player_name:
                continue

            current_hits = state.player_hits.get(player_id, 0)
            hit_threshold = current_hits + 1  # next hit they need
            if hit_threshold > 3:
                hit_threshold = None  # already has 3+ hits, no market

            # Collect props to attempt: hit market + HR market
            props_to_try = []
            if hit_threshold:
                hit_prop = self.cache.get_hit_ticker(player_name.lower(), threshold=hit_threshold)
                if hit_prop and hit_prop.ticker not in self.bought:
                    props_to_try.append(("hit", hit_prop))

            hr_prop = self.cache.get_hr_ticker(player_name.lower())
            if hr_prop and hr_prop.ticker not in self.bought:
                props_to_try.append(("hr", hr_prop))

            for prop_type, prop in props_to_try:
                result = await self.kalshi.get_yes_ask(prop.ticker, max_cents=MAX_BUY_CENTS)
                if result is None:
                    log.debug(f"  {player_name} {prop_type}: above {MAX_BUY_CENTS}¢, skipping")
                    continue

                yes_cents, qty = result
                spend = min(MAX_SPEND_USD, balance)
                count = min(qty, int(spend / (yes_cents / 100)))

                if count < 1:
                    continue

                label = f"{prop_type.upper()} ({hit_threshold}+)" if prop_type == "hit" else "HR"
                log.info(f"  BUY {player_name} {label} | {prop.ticker} | {count}x{yes_cents}¢ "
                         f"(inning {state.inning}, {offset} ahead)")

                try:
                    pos = OpenPosition(
                        ticker=prop.ticker,
                        player=player_name,
                        player_id=player_id,
                        contracts=count,
                        buy_price=yes_cents,
                        fill_confirmed=not self.paper,  # live = confirmed immediately
                    )

                    if self.paper:
                        # Don't place order — wait FILL_DELAY_SEC then check real price
                        log.info(f"  [PAPER] SIGNAL {player_name} {label} @ {yes_cents}¢ "
                                 f"— checking fill in {FILL_DELAY_SEC}s")
                        asyncio.create_task(self._confirm_paper_fill(pos))
                    else:
                        await self.kalshi.place_order(prop.ticker, "yes", yes_cents, count, "limit")
                        pos.actual_fill = yes_cents
                        log.info(f"  [LIVE] BUY {player_name} {label} {count}x{yes_cents}¢")

                    self.positions[prop.ticker] = pos
                    self.bought.add(prop.ticker)

                    cost = count * yes_cents / 100
                    log.info(f"  → cost ${cost:.2f} | sell triggers when bid ≥ {yes_cents + MIN_PROFIT_CENTS}¢")

                except Exception as e:
                    log.error(f"  Order failed for {player_name} {prop_type}: {e}")

    async def _confirm_paper_fill(self, pos: OpenPosition) -> None:
        """
        Paper mode: wait FILL_DELAY_SEC, then check the actual ask.
        This simulates real order latency — the price may have already moved.
        """
        await asyncio.sleep(FILL_DELAY_SEC)
        result = await self.kalshi.get_yes_ask(pos.ticker, max_cents=100)
        if result is None:
            log.info(f"  [PAPER] MISSED {pos.player} — no ask available after {FILL_DELAY_SEC}s delay")
            pos.sold = True  # treat as missed, skip tracking
            return

        delayed_price, _ = result
        slippage = delayed_price - pos.buy_price
        pos.actual_fill   = delayed_price
        pos.fill_confirmed = True

        if slippage > 0:
            log.info(f"  [PAPER] SLIPPAGE {pos.player}: signal={pos.buy_price}¢ "
                     f"actual={delayed_price}¢ (+{slippage}¢ slippage) — price already moved")
        elif slippage == 0:
            log.info(f"  [PAPER] FILLED {pos.player} @ {delayed_price}¢ — no slippage")
        else:
            log.info(f"  [PAPER] FILLED {pos.player} @ {delayed_price}¢ ({slippage}¢ better)")

    async def _check_sells(self, current_batters: set[int]) -> None:
        """Poll open positions — sell on profit target or force-sell after at-bat ends."""
        for ticker, pos in list(self.positions.items()):
            if pos.sold or not pos.fill_confirmed:
                continue

            # Track when the at-bat starts
            if pos.player_id in current_batters and not pos.ab_started:
                pos.ab_started = True
                log.info(f"  AT BAT: {pos.player} now batting — watching for sell")

            target = pos.actual_fill + MIN_PROFIT_CENTS
            bid    = await self.kalshi.get_yes_bid(ticker)
            if bid is None:
                continue

            # Profit target hit — sell at current bid
            if bid >= target:
                await self._sell(pos, bid, "profit target")
                continue

            # At-bat ended without hitting profit target — force sell at whatever bid is
            if pos.ab_started and pos.player_id not in current_batters:
                log.info(f"  AT BAT ENDED: {pos.player} — profit target not met, selling at {bid}¢")
                await self._sell(pos, bid, "at-bat ended")

    async def _sell(self, pos: OpenPosition, bid: int, reason: str) -> None:
        pnl = pos.contracts * (bid - pos.actual_fill) / 100
        fill_note = f"signal={pos.buy_price}¢ fill={pos.actual_fill}¢" if self.paper else f"fill={pos.actual_fill}¢"
        log.info(f"  {'[PAPER] ' if self.paper else ''}SELL {pos.player} @ {bid}¢ "
                 f"({fill_note}) | {reason} | {'+' if pnl >= 0 else ''}${pnl:.2f}")
        try:
            await self.kalshi.sell_position(pos.ticker, pos.contracts, bid)
            pos.sold = True
        except Exception as e:
            log.error(f"  Sell failed for {pos.player}: {e}")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",     action="store_true", help="Live mode (real orders)")
    parser.add_argument("--max-bet",  type=float, default=None, help="Max $ per trade (overrides default)")
    args = parser.parse_args()

    if args.max_bet is not None:
        global MAX_SPEND_USD
        MAX_SPEND_USD = args.max_bet

    cfg = yaml.safe_load(open("config.yaml"))
    kalshi = KalshiClient(
        key_id=cfg.get("kalshi_key_id", ""),
        private_key_path=cfg.get("kalshi_private_key_path", ""),
        paper_mode=not args.live,
    )
    cache = PropCache()
    await cache.build(kalshi)
    log.info(f"Cache: {len(cache.hr_cache)} HR, {len(cache.hits_cache)} hits markets")
    log.info(f"Max bet: ${MAX_SPEND_USD} | {'LIVE' if args.live else 'PAPER'}")

    bot = LineupBot(kalshi, cache, live=args.live)
    try:
        await bot.run()
    finally:
        await kalshi.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    asyncio.run(main())

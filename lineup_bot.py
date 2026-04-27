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

MAX_BUY_CENTS  = 78   # only buy if hit prop YES ask is ≤ this
MIN_PROFIT_CENTS = 10  # sell when price rises ≥ 10¢ from buy price
MAX_INNING     = 5    # stop buying after this inning
LOOKAHEAD      = [3, 4]  # spots ahead to target
POLL_SEC       = 3    # how often to poll each game feed
MAX_SPEND_USD  = 5.0  # max $ per trade


# ── Position tracker ───────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    ticker:    str
    player:    str
    contracts: int
    buy_price: int
    sold:      bool = False


class LineupBot:
    def __init__(self, kalshi: KalshiClient, cache: PropCache, live: bool):
        self.kalshi    = kalshi
        self.cache     = cache
        self.live      = live
        self.paper     = not live
        self.positions: dict[str, OpenPosition] = {}   # ticker → position
        self.bought:    set[str] = set()               # tickers we've ever bought today
        self._id_name:  dict[int, str] = {}            # player_id → full name (shared cache)

    async def run(self) -> None:
        log.info(f"Lineup bot started — {'LIVE' if self.live else 'PAPER'} mode")
        log.info(f"  Buy ≤ {MAX_BUY_CENTS}¢ | Sell @ {SELL_CENTS}¢ | Innings 1-{MAX_INNING}")

        async with aiohttp.ClientSession() as session:
            while True:
                games = get_todays_games()
                live_games = [g for g in games if g.status == "Live"]

                tasks = [
                    self._process_game(g.game_pk, session)
                    for g in live_games
                ]
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                await self._check_sells()
                await asyncio.sleep(POLL_SEC)

    async def _process_game(self, game_pk: int, session: aiohttp.ClientSession) -> None:
        state = await get_batting_state(game_pk, session, self._id_name)
        if state is None:
            return

        if state.inning > MAX_INNING:
            return

        for offset in LOOKAHEAD:
            player_id = state.upcoming(offset)
            if not player_id:
                continue

            player_name = state.name(player_id)
            if not player_name:
                continue

            prop = self.cache.get_hit_ticker(player_name.lower(), threshold=1)
            if prop is None:
                log.debug(f"  No hit market for {player_name}")
                continue

            if prop.ticker in self.bought:
                continue  # already in or done for this player today

            # Check current price
            result = await self.kalshi.get_yes_ask(prop.ticker, max_cents=MAX_BUY_CENTS)
            if result is None:
                log.debug(f"  {player_name}: price above {MAX_BUY_CENTS}¢, skipping")
                continue

            yes_cents, qty = result
            balance    = await self.kalshi.get_balance() if not self.paper else 500.0
            spend      = min(MAX_SPEND_USD, balance)
            count      = min(qty, int(spend / (yes_cents / 100)))

            if count < 1:
                log.warning(f"  {player_name}: not enough balance")
                continue

            log.info(f"  BUY {player_name} | {prop.ticker} | {count}x{yes_cents}¢ "
                     f"(inning {state.inning}, {offset} ahead) — will sell when +{MIN_PROFIT_CENTS}¢")

            try:
                await self.kalshi.place_order(prop.ticker, "yes", yes_cents, count, "limit")

                pos = OpenPosition(
                    ticker=prop.ticker,
                    player=player_name,
                    contracts=count,
                    buy_price=yes_cents,
                )
                self.positions[prop.ticker] = pos
                self.bought.add(prop.ticker)

                cost = count * yes_cents / 100
                log.info(f"  → cost ${cost:.2f} | holding, sell triggers when bid ≥ {yes_cents + MIN_PROFIT_CENTS}¢")

            except Exception as e:
                log.error(f"  Order failed for {player_name}: {e}")

    async def _check_sells(self) -> None:
        """Poll open positions and sell when price has risen by MIN_PROFIT_CENTS."""
        for ticker, pos in list(self.positions.items()):
            if pos.sold:
                continue
            target = pos.buy_price + MIN_PROFIT_CENTS
            bid = await self.kalshi.get_yes_bid(ticker)
            if bid is None:
                continue
            log.debug(f"  {pos.player} bid={bid}¢ (trigger @ {target}¢)")
            if bid >= target:
                # Sell at the current bid — capture the full move, not just 10¢
                profit = pos.contracts * (bid - pos.buy_price) / 100
                log.info(f"  SELL {pos.player} @ {bid}¢ (bought {pos.buy_price}¢) "
                         f"| +{bid - pos.buy_price}¢/contract | profit +${profit:.2f}")
                try:
                    await self.kalshi.sell_position(ticker, pos.contracts, bid)
                    pos.sold = True
                except Exception as e:
                    log.error(f"  Sell failed for {pos.player}: {e}")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    live_mode = "--live" in sys.argv

    cfg = yaml.safe_load(open("config.yaml"))
    kalshi = KalshiClient(
        key_id=cfg.get("kalshi_key_id", ""),
        private_key_path=cfg.get("kalshi_private_key_path", ""),
        paper_mode=not live_mode,
    )
    cache = PropCache()
    await cache.build(kalshi)
    log.info(f"Cache: {len(cache.hr_cache)} HR, {len(cache.hits_cache)} hits markets")

    bot = LineupBot(kalshi, cache, live=live_mode)
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

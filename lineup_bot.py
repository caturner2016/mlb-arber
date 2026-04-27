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
from datetime import date, datetime
from pathlib import Path

import aiohttp
import yaml

from kalshi.client import KalshiClient
from kalshi.props import PropCache
from mlb.games import get_todays_games
from mlb.lineup import get_batting_state

log = logging.getLogger("lineup_bot")


# ── Config ─────────────────────────────────────────────────────────────────────

MAX_BUY_CENTS    = 78   # buy early while price is still low, before casual money pumps it
MIN_PROFIT_CENTS = 5    # sell when price rises ≥ 5¢ from buy price
MAX_INNING       = 5    # stop buying after this inning
LOOKAHEAD        = [3, 4]  # spots ahead to target
POLL_SEC         = 3    # how often to poll each game feed
MAX_SPEND_USD    = 5.0  # max $ per trade
MAX_OPEN_USD          = 20.0 # pause buying when total open position value exceeds this
DAILY_STOP_LOSS       = 20.0 # stop all trading if realized losses hit this amount
AT_BAT_END_DELAY_SEC  = 15   # wait after detecting at-bat end before force selling


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
    ab_started:         bool  = False
    sold:               bool  = False
    fill_confirmed:     bool  = False
    ab_end_detected_at: float = 0.0  # monotonic time when at-bat end was first detected


class LineupBot:
    def __init__(self, kalshi: KalshiClient, cache: PropCache, live: bool):
        self.kalshi    = kalshi
        self.cache     = cache
        self.live      = live
        self.paper       = not live
        self.game_filter: str | None = None            # only watch games matching this team name
        self.positions: dict[str, OpenPosition] = {}   # ticker → position
        self.bought:    set[str] = set()               # tickers we've ever bought today
        self._id_name:  dict[int, str] = {}            # player_id → full name (shared cache)
        self._game_states: dict[int, int] = {}         # game_pk → current_batter_id
        self.realized_pnl: float = 0.0                 # cumulative P&L for today
        self._price_log   = self._open_price_log()

    def _open_price_log(self):
        path   = Path("price_tracking.csv")
        exists = path.exists()
        fh     = open(path, "a", newline="")
        import csv
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(["timestamp", "player", "prop_type", "batters_away",
                             "inning", "price_cents", "current_batter", "game"])
            fh.flush()
        return (fh, writer)

    def _log_price(self, player: str, prop_type: str, batters_away: int,
                   inning: int, price: int, current_batter: str, game: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._price_log[1].writerow([ts, player, prop_type, batters_away,
                                     inning, price, current_batter, game])
        self._price_log[0].flush()

    def open_exposure(self) -> float:
        """Total USD currently at risk in unsold positions."""
        return sum(
            p.contracts * (p.actual_fill or p.buy_price) / 100
            for p in self.positions.values()
            if not p.sold and p.fill_confirmed
        )

    async def run(self) -> None:
        log.info(f"Lineup bot started — {'LIVE' if self.live else 'PAPER'} mode")
        log.info(f"  Buy ≤ {MAX_BUY_CENTS}¢ | min profit {MIN_PROFIT_CENTS}¢ | Innings 1-{MAX_INNING}")
        if self.game_filter:
            log.info(f"  Game filter: {self.game_filter}")

        _poll_count = 0
        async with aiohttp.ClientSession() as session:
            while True:
                _poll_count += 1
                games     = get_todays_games()
                all_live  = [g for g in games if g.status == "Live"]

                # Log status every 10 polls (~30s) so user can see the bot is running
                if _poll_count % 10 == 1:
                    game_names = [f"{g.away_team} @ {g.home_team}" for g in all_live]
                    log.info(f"[poll #{_poll_count}] {len(all_live)} live game(s): {game_names or 'none'}")

                # All games → price tracking
                # Filtered games → buying
                if self.game_filter:
                    f = self.game_filter.lower()
                    buy_games   = [g for g in all_live if f in g.home_team.lower() or f in g.away_team.lower()]
                    track_games = all_live
                    if _poll_count % 10 == 1 and all_live:
                        log.info(f"  → filter '{self.game_filter}' matched {len(buy_games)} buy game(s)")
                else:
                    buy_games = track_games = all_live

                all_states = await asyncio.gather(
                    *[get_batting_state(g.game_pk, session, self._id_name) for g in track_games],
                    return_exceptions=True,
                )
                buy_pks = {g.game_pk for g in buy_games}

                current_batters: set[int] = set()
                for state in all_states:
                    if isinstance(state, Exception) or state is None:
                        continue
                    self._game_states[state.game_pk] = state.current_batter_id
                    current_batters.add(state.current_batter_id)
                    if state.inning <= MAX_INNING:
                        # Price tracking: all games
                        # Buying: only filtered games
                        await self._process_state(state, buy=state.game_pk in buy_pks)
                    elif _poll_count % 10 == 1:
                        log.info(f"  Inning {state.inning} > max {MAX_INNING} — not buying")

                await self._check_sells(current_batters)
                await asyncio.sleep(POLL_SEC)

    async def _process_state(self, state, buy: bool = True) -> None:
        balance         = await self.kalshi.get_balance() if not self.paper else 500.0
        game_label      = f"game_{state.game_pk}"
        current_batter  = state.name(state.current_batter_id) if state.current_batter_id else ""

        # ── Price tracking: observe all batters 1-8 away (all games) ──────────
        for offset in range(1, 9):
            pid = state.upcoming(offset)
            if not pid:
                continue
            pname = state.name(pid)
            if not pname:
                continue
            hits = state.player_hits.get(pid, 0)
            thr  = min(hits + 1, 3)
            hit_prop = self.cache.get_hit_ticker(pname.lower(), threshold=thr)
            hr_prop  = self.cache.get_hr_ticker(pname.lower())
            if hit_prop:
                bid = await self.kalshi.get_yes_bid(hit_prop.ticker)
                if bid:
                    self._log_price(pname, f"hit{thr}+", offset, state.inning, bid, current_batter, game_label)
            if hr_prop:
                bid = await self.kalshi.get_yes_bid(hr_prop.ticker)
                if bid:
                    self._log_price(pname, "hr", offset, state.inning, bid, current_batter, game_label)

        # ── Buy logic: only for filtered game, only at LOOKAHEAD offsets ────────
        if not buy:
            return

        if self.realized_pnl <= -DAILY_STOP_LOSS:
            log.warning(f"  STOP LOSS HIT — realized P&L ${self.realized_pnl:.2f} — no more buys today")
            return

        exposure = self.open_exposure()
        if exposure >= MAX_OPEN_USD:
            log.debug(f"  Open exposure ${exposure:.2f} ≥ ${MAX_OPEN_USD} — pausing buys")
            return

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

            target = pos.actual_fill + MIN_PROFIT_CENTS
            bid    = await self.kalshi.get_yes_bid(ticker)
            if bid is None:
                continue

            # Profit target hit — sell immediately
            if bid >= target:
                await self._sell(pos, bid, "profit target")
                continue

            # API shows player is now batting — in reality they stepped up ~15s ago,
            # pump has already happened. Sell now at the peak rather than waiting
            # for the at-bat result which arrives another 15s too late.
            if pos.player_id in current_batters and not pos.ab_started:
                pos.ab_started = True
                log.info(f"  AT BAT DETECTED: {pos.player} — selling now (API ~15s behind real life)")
                await self._sell(pos, bid, "at-bat started (API lag sell)")

    async def _sell(self, pos: OpenPosition, bid: int, reason: str) -> None:
        pnl = pos.contracts * (bid - pos.actual_fill) / 100
        self.realized_pnl += pnl
        fill_note = f"signal={pos.buy_price}¢ fill={pos.actual_fill}¢" if self.paper else f"fill={pos.actual_fill}¢"
        log.info(f"  {'[PAPER] ' if self.paper else ''}SELL {pos.player} @ {bid}¢ "
                 f"({fill_note}) | {reason} | {'+' if pnl >= 0 else ''}${pnl:.2f} "
                 f"| day P&L: {'+' if self.realized_pnl >= 0 else ''}${self.realized_pnl:.2f}")
        try:
            await self.kalshi.sell_position(pos.ticker, pos.contracts, bid)
            pos.sold = True
        except Exception as e:
            log.error(f"  Sell failed for {pos.player}: {e}")


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze() -> None:
    import pandas as pd
    path = Path("price_tracking.csv")
    if not path.exists():
        print("No price_tracking.csv yet — run the bot during a game first.")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("No data yet.")
        return

    print(f"\nRecords: {len(df)}  |  Players: {df['player'].nunique()}  |  Games: {df['game'].nunique()}")

    print("\n── Average price by batters away (hit props) ──")
    hits = df[df["prop_type"].str.startswith("hit")]
    if not hits.empty:
        by_dist = hits.groupby("batters_away")["price_cents"].mean().sort_index()
        for dist, avg in by_dist.items():
            bar = "█" * int(avg / 5)
            print(f"  {dist} away:  {avg:5.1f}¢  {bar}")

    print("\n── Average price by batters away (HR props) ──")
    hrs = df[df["prop_type"] == "hr"]
    if not hrs.empty:
        by_dist = hrs.groupby("batters_away")["price_cents"].mean().sort_index()
        for dist, avg in by_dist.items():
            bar = "█" * int(avg / 2)
            print(f"  {dist} away:  {avg:5.1f}¢  {bar}")

    print("\n── Price jump: 4 away → 1 away (per player, top movers) ──")
    for prop_type in ["hit1+", "hr"]:
        sub = df[df["prop_type"] == prop_type]
        if sub.empty:
            continue
        at4   = sub[sub["batters_away"] == 4].groupby("player")["price_cents"].mean()
        at1   = sub[sub["batters_away"] == 1].groupby("player")["price_cents"].mean()
        delta = (at1 - at4).dropna().sort_values(ascending=False).head(10)
        if not delta.empty:
            print(f"\n  {prop_type}:")
            for player, jump in delta.items():
                print(f"    {player:25s}  +{jump:.1f}¢ from 4-away to 1-away")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",     action="store_true", help="Live mode (real orders)")
    parser.add_argument("--max-bet",  type=float, default=None, help="Max $ per trade")
    parser.add_argument("--max-open", type=float, default=None, help="Max total $ open at once")
    parser.add_argument("--game",     type=str,   default=None, help="Only watch games with this team name e.g. 'rays'")
    parser.add_argument("--analyze",  action="store_true", help="Analyze price_tracking.csv")
    args = parser.parse_args()

    if args.max_bet is not None:
        global MAX_SPEND_USD
        MAX_SPEND_USD = args.max_bet
    if args.max_open is not None:
        global MAX_OPEN_USD
        MAX_OPEN_USD = args.max_open

    if args.analyze:
        analyze()
        return

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
    bot.game_filter = args.game
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

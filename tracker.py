"""
Market Tracker — correlates Kalshi prop price movements with live game state.

Polls every 5 seconds:
  - All Kalshi MLB player prop markets (hit + HR)
  - ESPN scoreboard for current batter, inning, score per live game

Logs every price change to market_log.csv with full game context:
  timestamp, ticker, player, prop_type, price, change,
  current_batter, batter_is_player, inning, half, balls, strikes, outs,
  home_team, away_team, home_score, away_score

Usage:
  py tracker.py             start tracking all live games
  py tracker.py --analyze   print summary of recorded movements
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import aiohttp
import yaml

from kalshi.client import KalshiClient
from kalshi.props import PropCache

log = logging.getLogger("tracker")

LOG_FILE   = Path("market_log.csv")
POLL_SEC   = 5
ESPN_URL   = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"

CSV_FIELDS = [
    "timestamp", "ticker", "player", "prop_type", "threshold",
    "price_cents", "price_change_cents",
    "current_batter", "batter_is_player",
    "inning", "inning_half", "balls", "strikes", "outs",
    "home_team", "away_team", "home_score", "away_score",
]


# ── Game state from ESPN ───────────────────────────────────────────────────────

@dataclass
class GameState:
    home_team:   str
    away_team:   str
    home_score:  int
    away_score:  int
    inning:      int
    inning_half: str   # "Top" / "Bottom"
    balls:       int
    strikes:     int
    outs:        int
    batter:      str   # display name, lowercase
    pitcher:     str


async def fetch_espn_states(session: aiohttp.ClientSession) -> list[GameState]:
    """Fetch all live game states from ESPN."""
    try:
        async with session.get(ESPN_URL, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return []
            data = await resp.json(content_type=None)
    except Exception as e:
        log.debug(f"ESPN fetch error: {e}")
        return []

    states = []
    for event in data.get("events", []):
        comp   = event.get("competitions", [{}])[0]
        status = event.get("status", {}).get("type", {}).get("description", "")
        if status not in ("In Progress",):
            continue

        sit     = comp.get("situation", {})
        batter  = sit.get("batter",  {}).get("athlete", {}).get("displayName", "")
        pitcher = sit.get("pitcher", {}).get("athlete", {}).get("displayName", "")
        teams   = comp.get("competitors", [])
        home    = next((t for t in teams if t.get("homeAway") == "home"), {})
        away    = next((t for t in teams if t.get("homeAway") == "away"), {})

        states.append(GameState(
            home_team   = home.get("team", {}).get("abbreviation", ""),
            away_team   = away.get("team", {}).get("abbreviation", ""),
            home_score  = int(home.get("score", 0) or 0),
            away_score  = int(away.get("score", 0) or 0),
            inning      = int(sit.get("inning", 0) or 0),
            inning_half = sit.get("inningHalf", ""),
            balls       = int(sit.get("balls", 0) or 0),
            strikes     = int(sit.get("strikes", 0) or 0),
            outs        = int(sit.get("outs", 0) or 0),
            batter      = batter.lower().strip(),
            pitcher     = pitcher.lower().strip(),
        ))

    return states


def find_game_state(player_name: str, states: list[GameState]) -> GameState | None:
    """Match a player name to a live game state."""
    lower = player_name.lower().strip()
    for s in states:
        if lower in s.batter or lower in s.pitcher:
            return s
        # Try last-name match
        last = lower.split()[-1] if lower.split() else ""
        if last and len(last) > 3 and (last in s.batter or last in s.pitcher):
            return s
    return None


# ── Price tracker ──────────────────────────────────────────────────────────────

class MarketTracker:
    def __init__(self, kalshi: KalshiClient, cache: PropCache):
        self.kalshi      = kalshi
        self.cache       = cache
        self.last_prices: dict[str, int] = {}   # ticker → last known price in cents
        self._writer: csv.DictWriter | None = None
        self._fh = None

    def _open_log(self) -> None:
        exists = LOG_FILE.exists()
        self._fh     = open(LOG_FILE, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        if not exists:
            self._writer.writeheader()
            self._fh.flush()
        log.info(f"Logging to {LOG_FILE}")

    def _log_row(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    async def run(self) -> None:
        self._open_log()
        log.info(f"Tracker started — polling every {POLL_SEC}s")

        async with aiohttp.ClientSession() as session:
            while True:
                # Fetch ESPN game states and Kalshi prices concurrently
                espn_task    = fetch_espn_states(session)
                prices_task  = self._fetch_all_prices()
                game_states, price_snapshot = await asyncio.gather(espn_task, prices_task)

                await self._process(price_snapshot, game_states)
                await asyncio.sleep(POLL_SEC)

    async def _fetch_all_prices(self) -> dict[str, tuple[str, str, int, int]]:
        """
        Returns {ticker: (player_name, prop_type, threshold, price_cents)}
        for every market currently in the cache.
        """
        tasks   = {}
        entries = {}

        for name, prop in self.cache.hr_cache.items():
            entries[prop.ticker] = (name, "hr", 1)
        for key, prop in self.cache.hits_cache.items():
            name, thr = key.split(":")[0], int(key.split(":")[1])
            entries[prop.ticker] = (name, "hit", thr)

        async def fetch_one(ticker):
            return ticker, await self.kalshi.get_yes_bid(ticker)

        results = await asyncio.gather(*[fetch_one(t) for t in entries], return_exceptions=True)

        snapshot = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            ticker, bid = item
            if bid is not None and ticker in entries:
                name, ptype, thr = entries[ticker]
                snapshot[ticker] = (name, ptype, thr, bid)
        return snapshot

    async def _process(
        self,
        snapshot:     dict[str, tuple],
        game_states:  list[GameState],
    ) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        changes = 0

        for ticker, (player, ptype, threshold, price) in snapshot.items():
            last = self.last_prices.get(ticker)
            self.last_prices[ticker] = price

            if last is None:
                continue  # first observation, no delta yet

            change = price - last
            if change == 0:
                continue  # no movement

            gs = find_game_state(player, game_states)
            batter_is_player = (
                gs is not None and
                (player.split()[-1] in gs.batter or player in gs.batter)
            ) if gs else False

            row = {
                "timestamp":         now,
                "ticker":            ticker,
                "player":            player.title(),
                "prop_type":         ptype,
                "threshold":         threshold,
                "price_cents":       price,
                "price_change_cents": change,
                "current_batter":    gs.batter.title() if gs else "",
                "batter_is_player":  batter_is_player,
                "inning":            gs.inning if gs else "",
                "inning_half":       gs.inning_half if gs else "",
                "balls":             gs.balls if gs else "",
                "strikes":           gs.strikes if gs else "",
                "outs":              gs.outs if gs else "",
                "home_team":         gs.home_team if gs else "",
                "away_team":         gs.away_team if gs else "",
                "home_score":        gs.home_score if gs else "",
                "away_score":        gs.away_score if gs else "",
            }
            self._log_row(row)

            direction = f"+{change}" if change > 0 else str(change)
            batter_tag = " ← AT BAT" if batter_is_player else ""
            log.info(
                f"  {player.title():22s} {ptype:3s} {threshold}+  "
                f"{last}¢ → {price}¢  ({direction}¢)"
                f"  inning={gs.inning if gs else '?'}  batter={gs.batter.title() if gs else '?'}"
                f"{batter_tag}"
            )
            changes += 1

        if changes:
            log.info(f"  — {changes} price changes this poll")


# ── Analysis ───────────────────────────────────────────────────────────────────

def analyze() -> None:
    if not LOG_FILE.exists():
        print("No market_log.csv found — run tracker first.")
        return

    import pandas as pd

    df = pd.read_csv(LOG_FILE)
    if df.empty:
        print("Log is empty.")
        return

    print(f"\nTotal price changes logged: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Average move when batter steps up vs not
    at_bat     = df[df["batter_is_player"] == True]
    not_at_bat = df[df["batter_is_player"] == False]

    print(f"\n── When player IS the current batter ({len(at_bat)} events) ──")
    if not at_bat.empty:
        print(f"  Avg price change:    {at_bat['price_change_cents'].mean():.1f}¢")
        print(f"  Median price change: {at_bat['price_change_cents'].median():.1f}¢")
        print(f"  Moves UP:            {(at_bat['price_change_cents'] > 0).sum()} "
              f"({(at_bat['price_change_cents'] > 0).mean():.0%})")
        print(f"  Avg up move:         {at_bat[at_bat['price_change_cents']>0]['price_change_cents'].mean():.1f}¢")

    print(f"\n── When player is NOT batting ({len(not_at_bat)} events) ──")
    if not not_at_bat.empty:
        print(f"  Avg price change:    {not_at_bat['price_change_cents'].mean():.1f}¢")
        print(f"  Moves UP:            {(not_at_bat['price_change_cents'] > 0).sum()} "
              f"({(not_at_bat['price_change_cents'] > 0).mean():.0%})")

    # Top players by avg step-up move
    print(f"\n── Top players by step-up price jump ──")
    if not at_bat.empty:
        by_player = (
            at_bat[at_bat["price_change_cents"] > 0]
            .groupby("player")["price_change_cents"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .head(15)
        )
        for player, row in by_player.iterrows():
            print(f"  {player:25s}  avg +{row['mean']:.1f}¢  (n={int(row['count'])})")

    # By inning
    print(f"\n── Average move by inning (at-bat only) ──")
    if not at_bat.empty:
        by_inning = (
            at_bat[at_bat["price_change_cents"] > 0]
            .groupby("inning")["price_change_cents"]
            .mean()
            .sort_index()
        )
        for inning, avg in by_inning.items():
            print(f"  Inning {inning}: +{avg:.1f}¢")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    cfg = yaml.safe_load(open("config.yaml"))
    kalshi = KalshiClient(
        key_id=cfg.get("kalshi_key_id", ""),
        private_key_path=cfg.get("kalshi_private_key_path", ""),
        paper_mode=True,  # read-only, no orders
    )
    cache = PropCache()
    await cache.build(kalshi)
    log.info(f"Cache: {len(cache.hr_cache)} HR, {len(cache.hits_cache)} hits markets")

    tracker = MarketTracker(kalshi, cache)
    try:
        await tracker.run()
    finally:
        await kalshi.close()
        if tracker._fh:
            tracker._fh.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if "--analyze" in sys.argv:
        analyze()
    else:
        asyncio.run(main())

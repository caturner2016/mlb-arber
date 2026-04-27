"""
Market Tracker — correlates Kalshi prop price movements with live game state.

Polls every 5 seconds:
  - All Kalshi MLB player prop markets (hit + HR), including orderbook depth
  - ESPN scoreboard for current batter, count, score, runners per live game
  - MLB live feed for batting order (to compute batters_away) and hits today

Logs every price change to market_log.csv:
  timestamp, ticker, player, prop_type, threshold, price_cents, price_change_cents,
  current_batter, batter_is_player, batters_away,
  inning, inning_half, balls, strikes, outs,
  runners_on_base, player_hits_today, ask_depth,
  home_team, away_team, home_score, away_score

Usage:
  py tracker.py             start tracking all live games
  py tracker.py --analyze   print summary of recorded movements
"""

from __future__ import annotations

import asyncio
import csv
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import yaml

from kalshi.client import KalshiClient
from kalshi.props import PropCache
from mlb.games import get_todays_games
from mlb.lineup import get_batting_state, BattingState

log = logging.getLogger("tracker")

LOG_FILE = Path("market_log.csv")
POLL_SEC = 5
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"

CSV_FIELDS = [
    "timestamp", "ticker", "player", "prop_type", "threshold",
    "price_cents", "price_change_cents",
    "current_batter", "batter_is_player", "batters_away",
    "inning", "inning_half", "balls", "strikes", "outs",
    "runners_on_base", "player_hits_today", "ask_depth",
    "home_team", "away_team", "home_score", "away_score",
]


# ── Game state ─────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    home_team:   str
    away_team:   str
    home_score:  int
    away_score:  int
    inning:      int
    inning_half: str
    balls:       int
    strikes:     int
    outs:        int
    batter:      str        # lowercase display name
    pitcher:     str
    on_first:    bool = False
    on_second:   bool = False
    on_third:    bool = False

    @property
    def runners_on_base(self) -> int:
        return sum([self.on_first, self.on_second, self.on_third])


async def fetch_espn_states(session: aiohttp.ClientSession) -> list[GameState]:
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
        if status != "In Progress":
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
            on_first    = bool(sit.get("onFirst")),
            on_second   = bool(sit.get("onSecond")),
            on_third    = bool(sit.get("onThird")),
        ))

    return states


def find_game_state(player_name: str, states: list[GameState]) -> GameState | None:
    lower = player_name.lower().strip()
    last  = lower.split()[-1] if lower.split() else ""
    for s in states:
        if lower in s.batter or lower in s.pitcher:
            return s
        if last and len(last) > 3 and (last in s.batter or last in s.pitcher):
            return s
    return None


# ── Batting order state (MLB live feed) ───────────────────────────────────────

def compute_batters_away(player_name: str, batting_states: list[BattingState]) -> int:
    """
    Returns how many batters until player_name is up, across all live games.
    0 = currently batting, 1 = on deck, etc.
    Returns -1 if player not found in any batting order.
    """
    lower = player_name.lower().strip()
    last  = lower.split()[-1] if lower.split() else ""

    for bs in batting_states:
        # Find this player's ID from id_to_name
        player_id = None
        for pid, name in bs.id_to_name.items():
            if name.lower() == lower or (last and name.lower().endswith(last)):
                player_id = pid
                break
        if player_id is None:
            continue
        if player_id not in bs.batting_order:
            continue

        current_idx = bs.batting_order.index(bs.current_batter_id) if bs.current_batter_id in bs.batting_order else None
        player_idx  = bs.batting_order.index(player_id)
        if current_idx is None:
            continue

        return (player_idx - current_idx) % 9

    return -1


def get_player_hits_today(player_name: str, batting_states: list[BattingState]) -> int:
    lower = player_name.lower().strip()
    last  = lower.split()[-1] if lower.split() else ""

    for bs in batting_states:
        for pid, name in bs.id_to_name.items():
            if name.lower() == lower or (last and name.lower().endswith(last)):
                return bs.player_hits.get(pid, 0)
    return 0


# ── Price tracker ──────────────────────────────────────────────────────────────

class MarketTracker:
    def __init__(self, kalshi: KalshiClient, cache: PropCache):
        self.kalshi      = kalshi
        self.cache       = cache
        self.last_prices: dict[str, int] = {}
        self._id_name:   dict[int, str]  = {}   # shared cache for MLB name lookup
        self._writer: csv.DictWriter | None = None
        self._fh = None

    def _open_log(self) -> None:
        exists   = LOG_FILE.exists()
        self._fh = open(LOG_FILE, "a", newline="")
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
                live_games = [g for g in get_todays_games() if g.status == "Live"]

                espn_task   = fetch_espn_states(session)
                prices_task = self._fetch_all_prices()
                mlb_task    = asyncio.gather(
                    *[get_batting_state(g.game_pk, session, self._id_name) for g in live_games],
                    return_exceptions=True,
                )

                (espn_states, price_snapshot, mlb_results) = await asyncio.gather(
                    espn_task, prices_task, mlb_task
                )

                batting_states = [s for s in mlb_results
                                  if isinstance(s, BattingState)]

                await self._process(price_snapshot, espn_states, batting_states)
                await asyncio.sleep(POLL_SEC)

    async def _fetch_all_prices(self) -> dict[str, tuple]:
        """
        Returns {ticker: (player_name, prop_type, threshold, bid_cents, ask_depth)}
        """
        entries: dict[str, tuple] = {}
        for name, prop in self.cache.hr_cache.items():
            entries[prop.ticker] = (name, "hr", 1)
        for key, prop in self.cache.hits_cache.items():
            name, thr = key.split(":")[0], int(key.split(":")[1])
            entries[prop.ticker] = (name, "hit", thr)

        async def fetch_one(ticker):
            bid   = await self.kalshi.get_yes_bid(ticker)
            depth_result = await self.kalshi.get_yes_ask(ticker, max_cents=99)
            depth = depth_result[1] if depth_result else 0
            return ticker, bid, depth

        results = await asyncio.gather(*[fetch_one(t) for t in entries], return_exceptions=True)

        snapshot = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            ticker, bid, depth = item
            if bid is not None and ticker in entries:
                name, ptype, thr = entries[ticker]
                snapshot[ticker] = (name, ptype, thr, bid, depth)
        return snapshot

    async def _process(
        self,
        snapshot:       dict[str, tuple],
        espn_states:    list[GameState],
        batting_states: list[BattingState],
    ) -> None:
        now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        changes = 0

        for ticker, (player, ptype, threshold, price, ask_depth) in snapshot.items():
            last = self.last_prices.get(ticker)
            self.last_prices[ticker] = price

            if last is None:
                continue
            change = price - last
            if change == 0:
                continue

            gs = find_game_state(player, espn_states)
            batter_is_player = (
                gs is not None and
                (player.split()[-1] in gs.batter or player in gs.batter)
            ) if gs else False

            batters_away     = compute_batters_away(player, batting_states)
            player_hits_today = get_player_hits_today(player, batting_states)

            row = {
                "timestamp":          now,
                "ticker":             ticker,
                "player":             player.title(),
                "prop_type":          ptype,
                "threshold":          threshold,
                "price_cents":        price,
                "price_change_cents": change,
                "current_batter":     gs.batter.title() if gs else "",
                "batter_is_player":   batter_is_player,
                "batters_away":       batters_away,
                "inning":             gs.inning if gs else "",
                "inning_half":        gs.inning_half if gs else "",
                "balls":              gs.balls if gs else "",
                "strikes":            gs.strikes if gs else "",
                "outs":               gs.outs if gs else "",
                "runners_on_base":    gs.runners_on_base if gs else "",
                "player_hits_today":  player_hits_today,
                "ask_depth":          ask_depth,
                "home_team":          gs.home_team if gs else "",
                "away_team":          gs.away_team if gs else "",
                "home_score":         gs.home_score if gs else "",
                "away_score":         gs.away_score if gs else "",
            }
            self._log_row(row)

            direction  = f"+{change}" if change > 0 else str(change)
            batter_tag = " ← AT BAT" if batter_is_player else ""
            away_tag   = f" ({batters_away} away)" if batters_away >= 0 else ""
            log.info(
                f"  {player.title():22s} {ptype:3s} {threshold}+  "
                f"{last}¢ → {price}¢  ({direction}¢)"
                f"  inning={gs.inning if gs else '?'}"
                f"  runners={gs.runners_on_base if gs else '?'}"
                f"{away_tag}{batter_tag}"
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
        print("No data yet.")
        return

    print(f"\nTotal price changes: {len(df)}  |  Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # ── At-bat vs not ──
    at_bat     = df[df["batter_is_player"] == True]
    not_at_bat = df[df["batter_is_player"] == False]

    print(f"\n── At bat ({len(at_bat)} events) ──")
    if not at_bat.empty:
        print(f"  Avg change:  {at_bat['price_change_cents'].mean():.1f}¢")
        print(f"  Moves UP:    {(at_bat['price_change_cents'] > 0).mean():.0%}")
        print(f"  Avg up move: {at_bat[at_bat['price_change_cents']>0]['price_change_cents'].mean():.1f}¢")

    print(f"\n── Not at bat ({len(not_at_bat)} events) ──")
    if not not_at_bat.empty:
        print(f"  Avg change:  {not_at_bat['price_change_cents'].mean():.1f}¢")
        print(f"  Moves UP:    {(not_at_bat['price_change_cents'] > 0).mean():.0%}")

    # ── Price by batters away ──
    if "batters_away" in df.columns:
        print(f"\n── Avg price change by batters away (hit props only) ──")
        hits = df[(df["prop_type"] == "hit") & (df["batters_away"] >= 0)]
        if not hits.empty:
            tbl = hits.groupby("batters_away")["price_change_cents"].agg(["mean", "count"]).sort_index()
            for dist, row in tbl.iterrows():
                bar = "▲" if row["mean"] > 0 else "▼"
                print(f"  {int(dist)} away:  {row['mean']:+.1f}¢  (n={int(row['count'])}) {bar}")

    # ── Runners on base effect ──
    if "runners_on_base" in df.columns:
        print(f"\n── At-bat pump by runners on base ──")
        if not at_bat.empty and "runners_on_base" in at_bat.columns:
            rb = at_bat.groupby("runners_on_base")["price_change_cents"].mean()
            for runners, avg in rb.items():
                print(f"  {int(runners)} runner(s): avg {avg:+.1f}¢")

    # ── By inning ──
    print(f"\n── At-bat avg move by inning ──")
    if not at_bat.empty:
        by_inning = at_bat.groupby("inning")["price_change_cents"].mean().sort_index()
        for inning, avg in by_inning.items():
            print(f"  Inning {int(inning)}: {avg:+.1f}¢")

    # ── Top players ──
    print(f"\n── Top players by at-bat price jump ──")
    if not at_bat.empty:
        top = (
            at_bat[at_bat["price_change_cents"] > 0]
            .groupby("player")["price_change_cents"]
            .agg(["mean", "count"])
            .sort_values("mean", ascending=False)
            .head(15)
        )
        for player, row in top.iterrows():
            print(f"  {player:25s}  avg +{row['mean']:.1f}¢  (n={int(row['count'])})")


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    cfg = yaml.safe_load(open("config.yaml"))
    kalshi = KalshiClient(
        key_id=cfg.get("kalshi_key_id", ""),
        private_key_path=cfg.get("kalshi_private_key_path", ""),
        paper_mode=True,
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

"""
Paper trading logger.

Records every triggered prop signal to trades.csv and tracks virtual P&L.
The key metric to watch: how lagged was the Kalshi price when we fired?
If yes_ask is still << 99 cents after the event, the edge is real.
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kalshi.client import Orderbook
from kalshi.props import PropMarket
from mlb.feed import PropEvent


TRADES_FILE = Path("trades.csv")

FIELDNAMES = [
    "trade_id",
    "timestamp",
    "game_pk",
    "player_name",
    "prop_type",
    "ticker",
    "event_raw",
    "inning",
    "half",
    "score",
    "event_detected_at",       # when MLB feed returned the event (ISO)
    "orderbook_fetched_at",    # when we got the Kalshi price (ISO)
    "latency_ms",              # ms from event detection to orderbook fetch
    "yes_ask_cents",           # Kalshi yes ask AT TIME OF TRIGGER (the key number)
    "contracts",               # how many contracts we would buy
    "cost_usd",                # total cost in USD
    "expected_pnl_usd",        # if settles YES: (100 - yes_ask) * contracts / 100
    "mode",                    # "paper"
]


@dataclass
class TradeRecord:
    trade_id: int
    event: PropEvent
    prop: PropMarket
    orderbook: Orderbook
    contracts: int
    cost_usd: float
    expected_pnl_usd: float


class PaperTrader:
    def __init__(self, max_trade_usd: float = 500.0, min_edge_cents: int = 5) -> None:
        self.max_trade_usd = max_trade_usd
        self.min_edge_cents = min_edge_cents
        self._trade_counter = 0
        self._virtual_pnl = 0.0
        self._trades: list[TradeRecord] = []
        self._ensure_csv()

    def _ensure_csv(self) -> None:
        if not TRADES_FILE.exists():
            with open(TRADES_FILE, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    def should_execute(self, orderbook: Orderbook) -> bool:
        """Return True if the Kalshi price is still lagged (edge exists)."""
        if orderbook.yes_ask is None:
            return False
        # If yes_ask > 95 cents, market already repriced — no edge
        return orderbook.yes_ask <= (100 - self.min_edge_cents)

    def execute(
        self,
        event: PropEvent,
        prop: PropMarket,
        orderbook: Orderbook,
    ) -> TradeRecord | None:
        """Log a paper trade. Returns the record, or None if no edge."""
        if not self.should_execute(orderbook):
            return None

        yes_ask = orderbook.yes_ask  # cents
        # Contracts = floor(max_trade_usd / (yes_ask / 100))
        contracts = max(1, int(self.max_trade_usd / (yes_ask / 100)))
        cost_usd = contracts * yes_ask / 100
        # Each contract settles at $1.00 if YES wins
        expected_pnl_usd = contracts * (100 - yes_ask) / 100

        self._trade_counter += 1
        record = TradeRecord(
            trade_id=self._trade_counter,
            event=event,
            prop=prop,
            orderbook=orderbook,
            contracts=contracts,
            cost_usd=cost_usd,
            expected_pnl_usd=expected_pnl_usd,
        )
        self._trades.append(record)
        self._append_csv(record)
        self._print_signal(record)
        return record

    def _append_csv(self, r: TradeRecord) -> None:
        ob = r.orderbook
        ev = r.event

        event_ts = datetime.now().isoformat()
        ob_ts = datetime.now().isoformat()
        latency_ms = (ob.fetched_at - ev.feed_detected_at) * 1000

        row = {
            "trade_id": r.trade_id,
            "timestamp": event_ts,
            "game_pk": ev.game_pk,
            "player_name": r.prop.player_name,
            "prop_type": ev.event_type,
            "ticker": r.prop.ticker,
            "event_raw": ev.raw_event,
            "inning": ev.inning,
            "half": ev.half,
            "score": f"{ev.away_score}-{ev.home_score}",
            "event_detected_at": event_ts,
            "orderbook_fetched_at": ob_ts,
            "latency_ms": f"{latency_ms:.1f}",
            "yes_ask_cents": ob.yes_ask,
            "contracts": r.contracts,
            "cost_usd": f"{r.cost_usd:.2f}",
            "expected_pnl_usd": f"{r.expected_pnl_usd:.2f}",
            "mode": "paper",
        }
        with open(TRADES_FILE, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    def _print_signal(self, r: TradeRecord) -> None:
        ob = r.orderbook
        latency_ms = (ob.fetched_at - r.event.feed_detected_at) * 1000
        print(
            f"\n[PAPER TRADE #{r.trade_id}] {r.prop.player_name} — {r.event.raw_event}\n"
            f"  Ticker:       {r.prop.ticker}\n"
            f"  YES ask:      {ob.yes_ask}¢  (should be ~99¢ if settled)\n"
            f"  Contracts:    {r.contracts}  @ ${ob.yes_ask / 100:.2f} each\n"
            f"  Cost:         ${r.cost_usd:.2f}\n"
            f"  Expected P&L: +${r.expected_pnl_usd:.2f}\n"
            f"  Latency:      {latency_ms:.0f}ms (event→orderbook fetch)\n"
        )

    def summary(self) -> None:
        print(f"\n=== Paper Trading Summary ===")
        print(f"  Trades logged: {len(self._trades)}")
        total_expected = sum(t.expected_pnl_usd for t in self._trades)
        total_cost = sum(t.cost_usd for t in self._trades)
        print(f"  Total deployed: ${total_cost:.2f}")
        print(f"  Expected P&L (if all settle YES): +${total_expected:.2f}")
        if self._trades:
            avg_ask = sum(t.orderbook.yes_ask or 0 for t in self._trades) / len(self._trades)
            print(f"  Avg YES ask at trigger: {avg_ask:.1f}¢")
        print(f"  Trades file: {TRADES_FILE.resolve()}")

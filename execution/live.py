"""
Live order execution — only active when config.mode == "live".

Wraps the paper trader but actually submits orders to Kalshi.
"""

from __future__ import annotations

from kalshi.client import KalshiClient, Orderbook
from kalshi.props import PropMarket
from mlb.feed import PropEvent
from execution.paper import PaperTrader, TradeRecord


class LiveTrader(PaperTrader):
    def __init__(
        self,
        kalshi: KalshiClient,
        max_trade_usd: float = 500.0,
        min_edge_cents: int = 5,
    ) -> None:
        super().__init__(max_trade_usd=max_trade_usd, min_edge_cents=min_edge_cents)
        self._kalshi = kalshi

    async def execute_live(
        self,
        event: PropEvent,
        prop: PropMarket,
        orderbook: Orderbook,
    ) -> TradeRecord | None:
        """Log the paper record AND submit the real order."""
        record = self.execute(event, prop, orderbook)
        if record is None:
            return None

        yes_ask = orderbook.yes_ask
        order = await self._kalshi.place_order(
            ticker=prop.ticker,
            side="yes",
            price_cents=yes_ask,
            count=record.contracts,
        )
        print(f"  [LIVE ORDER] submitted: {order}")
        return record

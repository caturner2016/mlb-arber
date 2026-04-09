"""
Kalshi API v2 client — thin async wrapper using aiohttp for speed.

Auth: RSA-PSS (Kalshi v2 requires signing each request with a private key).
In paper mode the client still reads orderbook prices but never submits orders.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


def _load_crypto():
    """Lazy import — only needed for live mode RSA signing."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    return hashes, serialization, padding


KALSHI_API = "https://trading-api.kalshi.com/trade-api/v2"
KALSHI_DEMO_API = "https://demo-api.kalshi.co/trade-api/v2"


@dataclass
class Orderbook:
    ticker: str
    yes_ask: int | None   # best YES ask in cents (what you pay to buy YES)
    no_ask: int | None    # best NO ask in cents
    yes_bid: int | None   # best YES bid
    fetched_at: float     # monotonic time for latency tracking


class KalshiClient:
    """
    Async Kalshi client. Designed to be reused across many requests.
    In paper mode: orderbook reads work; place_order is a no-op.
    """

    def __init__(
        self,
        key_id: str = "",
        private_key_path: str = "",
        paper_mode: bool = True,
        demo: bool = False,
    ) -> None:
        self.key_id = key_id or os.environ.get("KALSHI_KEY_ID", "")
        self.paper_mode = paper_mode
        self.base_url = KALSHI_DEMO_API if demo else KALSHI_API

        pkey_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        self._private_key = None
        if pkey_path and Path(pkey_path).exists():
            _, serialization, _ = _load_crypto()
            with open(pkey_path, "rb") as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)

        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(
                connector=connector,
                headers={"Accept-Encoding": "gzip", "Content-Type": "application/json"},
            )
        return self._session

    def _sign(self, method: str, path: str) -> dict[str, str]:
        """Generate RSA-PSS auth headers for Kalshi v2."""
        if not self._private_key or not self.key_id:
            return {}
        hashes, _, padding = _load_crypto()
        ts_ms = str(int(time.time() * 1000))
        msg = ts_ms + method.upper() + path
        signature = self._private_key.sign(
            msg.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

    async def get_orderbook(self, ticker: str) -> Orderbook:
        """
        Fetch the current orderbook for a market.
        Returns best YES ask/bid prices in cents.
        This is the hot path — called immediately after an event fires.
        """
        session = await self._get_session()
        path = f"/markets/{ticker}/orderbook"
        url = self.base_url + path
        headers = self._sign("GET", f"/trade-api/v2/markets/{ticker}/orderbook")

        fetched_at = time.monotonic()
        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status != 200:
                    return Orderbook(ticker=ticker, yes_ask=None, no_ask=None, yes_bid=None, fetched_at=fetched_at)
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return Orderbook(ticker=ticker, yes_ask=None, no_ask=None, yes_bid=None, fetched_at=fetched_at)

        ob = data.get("orderbook", {})
        yes_asks = ob.get("yes", [])  # list of [price_cents, quantity]
        no_asks = ob.get("no", [])
        yes_bids = [x for x in ob.get("yes", []) if x]  # bid side

        # Kalshi orderbook: "yes" field lists YES asks sorted ascending
        best_yes_ask = yes_asks[0][0] if yes_asks else None
        best_no_ask = no_asks[0][0] if no_asks else None
        best_yes_bid = yes_bids[-1][0] if yes_bids else None

        return Orderbook(
            ticker=ticker,
            yes_ask=best_yes_ask,
            no_ask=best_no_ask,
            yes_bid=best_yes_bid,
            fetched_at=fetched_at,
        )

    async def get_market(self, ticker: str) -> dict[str, Any]:
        """Fetch full market details (used for prop discovery, not hot path)."""
        session = await self._get_session()
        path = f"/markets/{ticker}"
        url = self.base_url + path
        headers = self._sign("GET", f"/trade-api/v2/markets/{ticker}")
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)

    async def place_order(
        self,
        ticker: str,
        side: str,           # "yes" or "no"
        price_cents: int,    # limit price (what you pay)
        count: int,          # number of contracts
    ) -> dict[str, Any]:
        """
        Place a limit buy order. No-op in paper mode.
        Returns order dict (or mock dict in paper mode).
        """
        if self.paper_mode:
            return {
                "paper": True,
                "ticker": ticker,
                "side": side,
                "price_cents": price_cents,
                "count": count,
            }

        if not self._private_key or not self.key_id:
            raise RuntimeError("Live mode requires KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH")

        session = await self._get_session()
        path = "/portfolio/orders"
        url = self.base_url + path
        headers = self._sign("POST", "/trade-api/v2/portfolio/orders")

        body = {
            "ticker": ticker,
            "action": "buy",
            "type": "limit",
            "side": side,
            "count": count,
            "yes_price": price_cents if side == "yes" else 100 - price_cents,
            "no_price": price_cents if side == "no" else 100 - price_cents,
        }

        async with session.post(url, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=3)) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

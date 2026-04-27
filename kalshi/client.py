"""
Kalshi API v2 async client.

Base URL: api.elections.kalshi.com (confirmed working from prior bot).
Auth: RSA-PSS — signature covers timestamp + method + "/trade-api/v2" + path.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class Orderbook:
    ticker: str
    yes_ask: int | None   # best YES ask in cents
    no_ask: int | None
    yes_bid: int | None
    fetched_at: float     # monotonic time


def _load_private_key(path: str):
    from cryptography.hazmat.primitives import serialization
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)


def _sign(private_key, key_id: str, method: str, path: str) -> dict[str, str]:
    """RSA-PSS signature. Message = timestamp_ms + METHOD + /trade-api/v2 + path."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding as crypto_padding

    ts = str(int(time.time() * 1000))
    msg = (ts + method.upper() + "/trade-api/v2" + path).encode()
    sig = private_key.sign(
        msg,
        crypto_padding.PSS(
            mgf=crypto_padding.MGF1(hashes.SHA256()),
            salt_length=crypto_padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY":       key_id,
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "Content-Type":            "application/json",
    }


class KalshiClient:
    def __init__(
        self,
        key_id: str = "",
        private_key_path: str = "",
        paper_mode: bool = True,
    ) -> None:
        self.key_id = key_id or os.environ.get("KALSHI_KEY_ID", "")
        self.paper_mode = paper_mode
        self._private_key = None

        pkey_path = private_key_path or os.environ.get("KALSHI_PRIVATE_KEY_PATH", "")
        if pkey_path and Path(pkey_path).exists():
            self._private_key = _load_private_key(pkey_path)

        self._session: aiohttp.ClientSession | None = None

    def _can_sign(self) -> bool:
        return bool(self._private_key and self.key_id)

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if not self._can_sign():
            return {"Content-Type": "application/json"}
        return _sign(self._private_key, self.key_id, method, path)

    async def _session_(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=20, ttl_dns_cache=300),
                headers={"Accept-Encoding": "gzip"},
            )
        return self._session

    async def get_market(self, ticker: str) -> dict[str, Any]:
        """Get full market data — includes yes_ask price."""
        session = await self._session_()
        path = f"/markets/{ticker}"
        headers = self._auth_headers("GET", path)
        async with session.get(
            KALSHI_BASE + path, headers=headers, timeout=aiohttp.ClientTimeout(total=3)
        ) as resp:
            if resp.status != 200:
                return {}
            data = await resp.json(content_type=None)
            return data.get("market", data)

    async def get_yes_ask(self, ticker: str, max_cents: int = 96) -> tuple[int, int] | None:
        """
        Sweeps the full orderbook up to max_cents.
        Returns (best_ask_cents, total_qty_available) across all levels <= max_cents.
        """
        try:
            session = await self._session_()
            path = f"/markets/{ticker}/orderbook"
            headers = self._auth_headers("GET", path)
            async with session.get(
                KALSHI_BASE + path, headers=headers, timeout=aiohttp.ClientTimeout(total=3)
            ) as resp:
                if resp.status != 200:
                    # Fall back to market snapshot
                    return await self._get_yes_ask_snapshot(ticker, max_cents)
                data = await resp.json(content_type=None)

            ob = data.get("orderbook", data)
            # YES asks are in the "yes" key as [[price_cents, qty], ...]
            # or sometimes "asks" / "yes_asks"
            levels = ob.get("yes") or ob.get("asks") or ob.get("yes_asks") or []

            if not levels:
                return await self._get_yes_ask_snapshot(ticker, max_cents)

            best_ask  = None
            total_qty = 0
            for level in levels:
                price = level[0] if isinstance(level, (list, tuple)) else level.get("price", 0)
                qty   = level[1] if isinstance(level, (list, tuple)) else level.get("quantity", 0)
                p = int(round(float(price) * 100)) if float(price) <= 1.0 else int(price)
                if 1 <= p <= max_cents:
                    if best_ask is None:
                        best_ask = p
                    total_qty += int(qty)

            if best_ask is None or total_qty == 0:
                return None
            return (best_ask, total_qty)

        except (aiohttp.ClientError, asyncio.TimeoutError):
            return await self._get_yes_ask_snapshot(ticker, max_cents)

    async def _get_yes_ask_snapshot(self, ticker: str, max_cents: int) -> tuple[int, int] | None:
        """Fallback: single best ask from market snapshot."""
        try:
            market = await self.get_market(ticker)
            if market.get("status") != "active":
                return None
            yes_ask  = market.get("yes_ask_dollars") or market.get("yes_ask")
            yes_size = int(float(market.get("yes_ask_size_fp") or 100))
            if yes_ask is None:
                return None
            v = float(yes_ask)
            yes_cents = int(round(v * 100)) if v <= 1.0 else int(v)
            if yes_cents < 1 or yes_cents > max_cents:
                return None
            return (yes_cents, yes_size)
        except Exception:
            return None

    async def get_markets(self, series_ticker: str, cursor: str | None = None) -> dict[str, Any]:
        """Paginated market listing for a series."""
        session = await self._session_()
        path = "/markets"
        params: dict[str, Any] = {"series_ticker": series_ticker, "status": "open", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        headers = self._auth_headers("GET", path)
        async with session.get(
            KALSHI_BASE + path, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status != 200:
                return {}
            return await resp.json(content_type=None)

    async def get_markets_search(self, query: str) -> dict:
        """Search markets by title keyword — used for debugging series tickers."""
        session = await self._session_()
        path = "/markets"
        params = {"status": "open", "limit": 10}
        headers = self._auth_headers("GET", path)
        async with session.get(
            KALSHI_BASE + path, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            if resp.status != 200:
                return {"status": resp.status}
            data = await resp.json(content_type=None)
            markets = data.get("markets", [])
            mlb = [{"ticker": m.get("ticker"), "title": m.get("title"), "series": m.get("series_ticker")}
                   for m in markets if query.lower() in (m.get("title") or "").lower()
                   or query.lower() in (m.get("series_ticker") or "").lower()]
            all_series = list({m.get("series_ticker") for m in markets if m.get("series_ticker")})
            return {"mlb_matches": mlb, "all_series_sample": all_series[:20], "total_returned": len(markets)}

    async def get_balance(self) -> float:
        """Account balance in USD."""
        session = await self._session_()
        path = "/portfolio/balance"
        headers = self._auth_headers("GET", path)
        try:
            async with session.get(
                KALSHI_BASE + path, headers=headers, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                data = await resp.json(content_type=None)
                return data.get("balance", 0) / 100.0
        except Exception:
            return 0.0

    async def place_order(
        self,
        ticker: str,
        side: str,        # "yes" or "no"
        price_cents: int,
        count: int,
        order_type: str = "limit",  # "limit" or "ioc"
    ) -> dict[str, Any]:
        if self.paper_mode:
            return {"paper": True, "ticker": ticker, "side": side,
                    "price_cents": price_cents, "count": count}

        if not self._can_sign():
            raise RuntimeError("Live mode requires KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH")

        session = await self._session_()
        path = "/portfolio/orders"
        headers = self._auth_headers("POST", path)
        body = {
            "ticker":    ticker,
            "action":    "buy",
            "side":      side,
            "type":      order_type,
            "count":     count,
            "yes_price": price_cents if side == "yes" else 100 - price_cents,
        }
        async with session.post(
            KALSHI_BASE + path, headers=headers, json=body, timeout=aiohttp.ClientTimeout(total=3)
        ) as resp:
            resp.raise_for_status()
            return await resp.json(content_type=None)

    async def cancel_resting_orders(self) -> None:
        """Cancel all open resting orders (used before hits market buys)."""
        session = await self._session_()
        path = "/portfolio/orders"
        headers = self._auth_headers("GET", path)
        try:
            async with session.get(
                KALSHI_BASE + path, headers=headers,
                params={"status": "resting"}, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                data = await resp.json(content_type=None)
            for order in data.get("orders", []):
                oid = order.get("order_id")
                if not oid:
                    continue
                del_path = f"/portfolio/orders/{oid}"
                del_headers = self._auth_headers("DELETE", del_path)
                try:
                    async with session.delete(
                        KALSHI_BASE + del_path, headers=del_headers, timeout=aiohttp.ClientTimeout(total=3)
                    ) as r:
                        pass
                except Exception:
                    pass
        except Exception:
            pass

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

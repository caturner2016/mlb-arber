import time
import base64
import logging
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.padding import PSS, MGF1
from config import KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH

log = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com"
_PORTFOLIO = "/trade-api/v2/portfolio"
_MARKETS   = "/trade-api/v2/markets"

_RETRY_DELAYS = [2, 4, 8, 16]


class KalshiClient:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        with open(KALSHI_PRIVATE_KEY_PATH, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        log.info("Kalshi API key loaded (id: %s...)", KALSHI_KEY_ID[:8])

    def _sign(self, method: str, path: str) -> dict:
        # path must be the full path e.g. /trade-api/v2/portfolio/balance
        # strip query string before signing
        sign_path = path.split("?")[0]
        ts = str(int(time.time() * 1000))
        msg = f"{ts}{method.upper()}{sign_path}".encode("utf-8")
        sig = self._private_key.sign(
            msg,
            PSS(mgf=MGF1(hashes.SHA256()), salt_length=PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "Content-Type":            "application/json",
            "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = BASE_URL + path
        for attempt, delay in enumerate([-1] + _RETRY_DELAYS):
            if delay >= 0:
                log.warning("Retrying %s %s in %ds", method, path, delay)
                time.sleep(delay)
            try:
                resp = self._session.request(
                    method, url, headers=self._sign(method, path), timeout=20, **kwargs
                )
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt == len(_RETRY_DELAYS) - 1:
                    raise
                log.warning("Request error: %s", e)
        raise RuntimeError(f"All retries failed for {method} {path}")

    def _get(self, path: str, params: dict | None = None) -> dict:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, json=body)

    # ------------------------------------------------------------------
    # Markets
    # ------------------------------------------------------------------

    def get_markets_by_series(self, series_ticker: str, limit: int = 200) -> list[dict]:
        params = {"status": "open", "limit": limit, "series_ticker": series_ticker}
        data = self._get(_MARKETS, params=params)
        return data.get("markets", [])

    def get_tennis_markets(self) -> list[dict]:
        results = []
        seen = set()
        for series in ("KXATPMATCH", "KXWTAMATCH", "KXITFMATCH"):
            for m in self.get_markets_by_series(series):
                if m["ticker"] not in seen:
                    seen.add(m["ticker"])
                    results.append(m)
        return results

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        data = self._get(_PORTFOLIO + "/balance")
        return data.get("balance", 0) / 100.0

    def get_positions(self) -> list[dict]:
        data = self._get(_PORTFOLIO + "/positions")
        return data.get("market_positions", [])

    def place_order(self, ticker: str, side: str, count: int, price: int, client_order_id: str) -> dict:
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "type": "market",
            "action": "buy",
            "side": side,
            "count": count,
        }
        log.info("Placing order: %s %s x%d (market)", side.upper(), ticker, count)
        return self._post(_PORTFOLIO + "/orders", body)

    def get_orders(self) -> list[dict]:
        data = self._get(_PORTFOLIO + "/orders")
        return data.get("orders", [])

    def cancel_resting_orders(self):
        try:
            orders = self._get(_PORTFOLIO + "/orders", params={"status": "resting"}).get("orders", [])
            for order in orders:
                oid = order.get("order_id")
                if not oid:
                    continue
                path = f"{_PORTFOLIO}/orders/{oid}"
                try:
                    self._request("DELETE", path)
                    log.info("Cancelled resting order %s", oid)
                except Exception as e:
                    log.warning("Failed to cancel order %s: %s", oid, e)
            if orders:
                log.info("Cancelled %d resting orders", len(orders))
        except Exception as e:
            log.warning("cancel_resting_orders failed: %s", e)

    def login(self):
        pass

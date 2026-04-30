import time
import base64
import logging
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from config import KALSHI_BASE_URL, KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH

log = logging.getLogger(__name__)

_RETRY_DELAYS = [2, 4, 8, 16]


class KalshiClient:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        with open(KALSHI_PRIVATE_KEY_PATH, "rb") as f:
            self._private_key = serialization.load_pem_private_key(f.read(), password=None)
        log.info("Kalshi API key loaded (id: %s)", KALSHI_KEY_ID[:8] + "...")

    # ------------------------------------------------------------------
    # Auth — RSA signature on every request
    # ------------------------------------------------------------------

    def _auth_headers(self, method: str, path: str) -> dict:
        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method.upper()}{path}".encode()
        sig = self._private_key.sign(message, asym_padding.PKCS1v15(), hashes.SHA256())
        return {
            "KALSHI-ACCESS-KEY":       KALSHI_KEY_ID,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = KALSHI_BASE_URL + path
        for attempt, delay in enumerate([-1] + _RETRY_DELAYS):
            if delay >= 0:
                log.warning("Retrying %s %s in %ds", method, path, delay)
                time.sleep(delay)
            try:
                headers = self._auth_headers(method, path)
                resp = self._session.request(method, url, headers=headers, timeout=20, **kwargs)
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

    def get_markets(self, search: str = "", limit: int = 200) -> list[dict]:
        params = {"status": "open", "limit": limit}
        if search:
            params["search"] = search
        data = self._get("/markets", params=params)
        return data.get("markets", [])

    def get_tennis_markets(self) -> list[dict]:
        results = []
        seen = set()
        for term in ("tennis", "atp", "wta", "wimbledon", "us open", "french open", "australian open"):
            for m in self.get_markets(search=term):
                if m["ticker"] not in seen:
                    seen.add(m["ticker"])
                    results.append(m)
        return results

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        data = self._get("/portfolio/balance")
        return data.get("balance", 0) / 100.0

    def get_positions(self) -> list[dict]:
        data = self._get("/portfolio/positions")
        return data.get("market_positions", [])

    def place_order(
        self,
        ticker: str,
        side: str,
        count: int,
        price: int,
        client_order_id: str,
    ) -> dict:
        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "type": "limit",
            "action": "buy",
            "side": side,
            "count": count,
            f"{side}_price": price,
        }
        log.info("Placing order: %s %s x%d @ %dc", side.upper(), ticker, count, price)
        return self._post("/portfolio/orders", body)

    def get_orders(self) -> list[dict]:
        data = self._get("/portfolio/orders")
        return data.get("orders", [])

    def login(self):
        pass  # no-op, kept for compatibility

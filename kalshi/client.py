import time
import logging
import requests
from config import KALSHI_BASE_URL, KALSHI_EMAIL, KALSHI_PASSWORD

log = logging.getLogger(__name__)

_RETRY_DELAYS = [2, 4, 8, 16]


class KalshiClient:
    def __init__(self):
        self._token: str | None = None
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self):
        resp = self._post("/login", {"email": KALSHI_EMAIL, "password": KALSHI_PASSWORD}, auth=False)
        self._token = resp["token"]
        self._session.headers.update({"Authorization": f"Bearer {self._token}"})
        log.info("Kalshi login OK")

    def _ensure_auth(self):
        if not self._token:
            self.login()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, auth: bool = True, **kwargs) -> dict:
        if auth:
            self._ensure_auth()
        url = KALSHI_BASE_URL + path
        for attempt, delay in enumerate([-1] + _RETRY_DELAYS):
            if delay >= 0:
                log.warning("Retrying %s %s in %ds", method, path, delay)
                time.sleep(delay)
            try:
                resp = self._session.request(method, url, timeout=20, **kwargs)
                if resp.status_code == 401 and auth:
                    log.warning("Token expired, re-logging in")
                    self._token = None
                    self.login()
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                if attempt == len(_RETRY_DELAYS) - 1:
                    raise
                log.warning("Request error: %s", e)
        raise RuntimeError(f"All retries failed for {method} {path}")

    def _get(self, path: str, params: dict | None = None) -> dict:
        return self._request("GET", path, params=params)

    def _post(self, path: str, body: dict, auth: bool = True) -> dict:
        return self._request("POST", path, auth=auth, json=body)

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
        for term in ("tennis", "atp", "wta", "wimbledon", "us open", "french open", "australian open"):
            markets = self.get_markets(search=term)
            for m in markets:
                if m["ticker"] not in {r["ticker"] for r in results}:
                    results.append(m)
        return results

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------

    def get_balance(self) -> float:
        """Returns balance in dollars."""
        data = self._get("/portfolio/balance")
        return data.get("balance", 0) / 100.0

    def get_positions(self) -> list[dict]:
        data = self._get("/portfolio/positions")
        return data.get("market_positions", [])

    def place_order(
        self,
        ticker: str,
        side: str,       # "yes" or "no"
        count: int,      # number of contracts
        price: int,      # cents (1-99)
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

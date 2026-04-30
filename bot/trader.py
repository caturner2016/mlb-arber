import uuid
import logging
from kalshi.client import KalshiClient
from kalshi.markets import parse_all_tennis_markets
from tennis.predictor import TennisPredictor
from bot.risk import should_bet, size_bet, daily_budget_remaining
from bot.state import already_bet, record_bet

log = logging.getLogger(__name__)


class TennisTrader:
    def __init__(self):
        self.kalshi = KalshiClient()
        self.predictor = TennisPredictor()
        self._initialized = False

    def initialize(self):
        log.info("Loading tennis data and building Elo ratings...")
        self.predictor.load()
        self.kalshi.login()
        self.kalshi.cancel_resting_orders()
        self._initialized = True
        balance = self.kalshi.get_balance()
        log.info("Kalshi balance: $%.2f", balance)

    def _make_order_id(self, ticker: str, side: str) -> str:
        short = ticker[-16:] if len(ticker) > 16 else ticker
        return f"{short}-{side}-{uuid.uuid4().hex[:8]}"

    def run_cycle(self):
        if not self._initialized:
            self.initialize()

        budget = daily_budget_remaining()
        if budget <= 0:
            log.info("Daily stop-loss reached ($%.2f), skipping cycle", budget)
            return

        log.info("Running trade cycle — daily budget remaining: $%.2f", budget)

        try:
            raw_markets = self.kalshi.get_tennis_markets()
        except Exception as e:
            log.error("Failed to fetch markets: %s", e)
            return

        markets = parse_all_tennis_markets(raw_markets)
        log.info("Found %d parseable tennis match markets", len(markets))

        placed = 0
        skipped_no_data = 0
        skipped_no_edge = 0
        skipped_already_bet = 0

        for m in markets:
            if already_bet(m["ticker"]):
                skipped_already_bet += 1
                continue

            market_prob = m["yes_price_cents"] / 100.0
            model_prob = self.predictor.predict(
                m["player_yes"], m["player_no"], m["surface"]
            )

            if model_prob is None:
                skipped_no_data += 1
                log.debug("No Elo data for %s vs %s", m["player_yes"], m["player_no"])
                continue

            decision = should_bet(model_prob, market_prob)
            if decision is None:
                skipped_no_edge += 1
                log.debug(
                    "%s vs %s — model=%.1f%% market=%.1f%% — no edge",
                    m["player_yes"], m["player_no"],
                    model_prob * 100, market_prob * 100,
                )
                continue

            side, edge = decision

            # Use the actual ask price so the limit order fills immediately
            ask_cents = m["yes_ask_cents"] if side == "yes" else m["no_ask_cents"]
            ask_cents = max(1, min(99, ask_cents))

            sizing = size_bet(model_prob, market_prob, side)
            if sizing is None:
                continue

            amount, contracts, _ = sizing
            # Recalculate amount using actual ask price
            amount = round(contracts * ask_cents / 100, 2)

            log.info(
                "BET %s | %s | model=%.1f%% mkt=%.1f%% edge=%.1f%% | %s x%d @ %dc = $%.2f",
                side.upper(), m["title"],
                model_prob * 100, market_prob * 100, edge * 100,
                side.upper(), contracts, ask_cents, amount,
            )

            try:
                order_id = self._make_order_id(m["ticker"], side)
                resp = self.kalshi.place_order(
                    ticker=m["ticker"],
                    side=side,
                    count=contracts,
                    price=ask_cents,
                    client_order_id=order_id,
                )
                kalshi_id = resp.get("order", {}).get("order_id", order_id)
                record_bet(
                    ticker=m["ticker"],
                    title=m["title"],
                    player_yes=m["player_yes"],
                    player_no=m["player_no"],
                    side=side,
                    contracts=contracts,
                    price_cents=ask_cents,
                    amount_usd=amount,
                    model_prob=model_prob,
                    market_prob=market_prob,
                    edge=edge,
                    kalshi_order_id=kalshi_id,
                )
                placed += 1
            except Exception as e:
                log.error("Order failed for %s: %s", m["ticker"], e)

        log.info(
            "Cycle done — placed=%d no_data=%d no_edge=%d already_bet=%d",
            placed, skipped_no_data, skipped_no_edge, skipped_already_bet,
        )

    def refresh_data(self):
        """Re-download current year data to capture recent results."""
        log.info("Refreshing tennis data...")
        self.predictor.refresh()
        log.info("Data refresh complete")

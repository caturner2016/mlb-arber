import logging
from datetime import date
from config import DAILY_STOP_LOSS, MAX_BET, MIN_EDGE, KELLY_FRACTION, MIN_CONTRACTS, MAX_FADE_PROB, MIN_YES_PRICE
from bot.state import get_daily_spent

log = logging.getLogger(__name__)


def daily_budget_remaining() -> float:
    spent = get_daily_spent()
    remaining = DAILY_STOP_LOSS - spent
    return max(0.0, remaining)


def kelly_bet_size(model_prob: float, market_prob: float, side: str) -> float:
    """
    Returns optimal bet size as a fraction of the daily stop-loss budget.
    side: 'yes' or 'no'
    """
    if side == "yes":
        p = model_prob
        q = market_prob  # implied prob
    else:
        p = 1 - model_prob
        q = 1 - market_prob

    edge = p - q
    if edge <= 0:
        return 0.0

    # Kelly: f = edge / (1 - q)  [net odds form for binary contract]
    denominator = 1 - q
    if denominator <= 0:
        return 0.0

    f = edge / denominator
    fractional_f = f * KELLY_FRACTION
    return min(fractional_f, 1.0)


def size_bet(model_prob: float, market_prob: float, side: str) -> tuple[float, int, int] | None:
    """
    Returns (amount_usd, contracts, price_cents) or None if bet should be skipped.
    """
    edge = (model_prob - market_prob) if side == "yes" else (market_prob - model_prob) - (1 - model_prob)
    # simpler edge check
    yes_edge = model_prob - market_prob
    no_edge = (1 - market_prob) - (1 - model_prob)  # simplifies to market_prob - model_prob... wait
    # edge for YES: model says higher prob than market
    # edge for NO: model says lower prob than market implies (i.e. NO is underpriced)

    if side == "yes":
        actual_edge = model_prob - market_prob
    else:
        actual_edge = (1 - model_prob) - (1 - market_prob)  # = market_prob - model_prob

    if actual_edge < MIN_EDGE:
        return None

    remaining = daily_budget_remaining()
    if remaining <= 0:
        log.info("Daily stop-loss reached, skipping bet")
        return None

    fraction = kelly_bet_size(model_prob, market_prob, side)
    if fraction <= 0:
        return None

    # size against stop-loss budget, not full bankroll
    raw_amount = fraction * DAILY_STOP_LOSS
    amount = min(raw_amount, MAX_BET, remaining)

    if side == "yes":
        price_cents = round(market_prob * 100)
    else:
        price_cents = round((1 - market_prob) * 100)

    price_cents = max(1, min(99, price_cents))

    cost_per_contract = price_cents / 100.0
    contracts = int(amount / cost_per_contract)
    contracts = max(MIN_CONTRACTS, contracts)

    actual_amount = contracts * cost_per_contract
    if actual_amount > remaining:
        contracts = max(MIN_CONTRACTS, int(remaining / cost_per_contract))
        actual_amount = contracts * cost_per_contract

    if contracts < MIN_CONTRACTS:
        return None

    return actual_amount, contracts, price_cents


def should_bet(model_prob: float, market_prob: float) -> tuple[str, float] | None:
    """
    Decide whether to bet YES or NO and return (side, edge).
    Returns None if no edge.

    Guards:
    - Don't bet YES on a heavy underdog (market < MIN_YES_PRICE): model
      compression makes underdogs look systematically attractive.
    - Don't bet NO against a heavy favorite (market > MAX_FADE_PROB): the
      model regresses to 50% more than efficient markets do for dominant
      players, generating false edges.
    """
    yes_edge = model_prob - market_prob
    no_edge = market_prob - model_prob

    if yes_edge >= MIN_EDGE and market_prob >= MIN_YES_PRICE:
        return "yes", yes_edge
    if no_edge >= MIN_EDGE and market_prob <= MAX_FADE_PROB:
        return "no", no_edge
    return None

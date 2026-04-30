import re
import logging
from datetime import datetime, timezone
from config import MIN_HOURS_TO_CLOSE

log = logging.getLogger(__name__)

_SURFACE_MAP = {
    "wimbledon": "grass",
    "roland garros": "clay",
    "french open": "clay",
    "monte carlo": "clay",
    "rome": "clay",
    "madrid": "clay",
    "barcelona": "clay",
    "australian open": "hard",
    "us open": "hard",
    "miami": "hard",
    "indian wells": "hard",
    "canada": "hard",
    "cincinnati": "hard",
    "shanghai": "hard",
    "paris": "hard",
}

_SURFACE_RE = re.compile(r"\b(clay|grass|hard)\b", re.IGNORECASE)

# Extracts opponent from "... the Sinner vs Fils: ..." pattern
_OPPONENT_RE = re.compile(
    r"\bthe\s+(\S+)\s+vs\s+(\S+)\s*:",
    re.IGNORECASE,
)


def _infer_surface(text: str) -> str:
    text_lower = text.lower()
    m = _SURFACE_RE.search(text_lower)
    if m:
        return m.group(1).lower()
    for keyword, surface in _SURFACE_MAP.items():
        if keyword in text_lower:
            return surface
    return "hard"


def _dollars_to_cents(val) -> int:
    try:
        return round(float(val) * 100)
    except (TypeError, ValueError):
        return 0


def _is_pre_match(close_time_str: str) -> bool:
    if not close_time_str:
        return True
    try:
        close_dt = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        hours_left = (close_dt - datetime.now(timezone.utc)).total_seconds() / 3600
        return hours_left >= MIN_HOURS_TO_CLOSE
    except ValueError:
        return True


def parse_match_market(market: dict) -> dict | None:
    """
    Parse a single Kalshi tennis match market.
    Each market represents one player winning (yes_sub_title = that player).
    Returns None if market is not a valid pre-match 1v1 market.
    """
    player_yes = (market.get("yes_sub_title") or "").strip()
    if not player_yes:
        return None

    title = market.get("title", "") or ""
    full_text = f"{title} {market.get('rules_primary', '')}"

    # Extract opponent from "the X vs Y:" pattern in title
    m = _OPPONENT_RE.search(title)
    if not m:
        return None

    name_a, name_b = m.group(1).strip(), m.group(2).strip()
    # player_no is whichever name in the VS pair doesn't match player_yes's last name
    yes_last = player_yes.split()[-1].lower()
    if name_a.lower() == yes_last or name_a.lower() in player_yes.lower():
        player_no = name_b
    elif name_b.lower() == yes_last or name_b.lower() in player_yes.lower():
        player_no = name_a
    else:
        # fallback: pick the one that doesn't appear in player_yes
        player_no = name_b if name_a.lower() in player_yes.lower() else name_a

    if not player_no:
        return None

    if not _is_pre_match(market.get("close_time", "")):
        log.debug("Skipping likely live market: %s", title)
        return None

    yes_ask = _dollars_to_cents(market.get("yes_ask_dollars", 0))
    yes_bid = _dollars_to_cents(market.get("yes_bid_dollars", 0))
    no_ask  = _dollars_to_cents(market.get("no_ask_dollars", 0))

    yes_mid = (yes_bid + yes_ask) / 2 if yes_ask else yes_bid
    if yes_mid <= 0 or yes_mid >= 100:
        return None

    surface = _infer_surface(full_text)

    return {
        "ticker":          market["ticker"],
        "event_ticker":    market.get("event_ticker", market["ticker"]),
        "title":           title,
        "player_yes":      player_yes,
        "player_no":       player_no,
        "surface":         surface,
        "yes_price_cents": round(yes_mid),
        "yes_ask_cents":   yes_ask,
        "no_ask_cents":    no_ask,
        "close_time":      market.get("close_time", ""),
    }


def parse_all_tennis_markets(markets: list[dict]) -> list[dict]:
    parsed = []
    seen_events = set()
    for m in markets:
        result = parse_match_market(m)
        if not result:
            continue
        # One market per matchup — skip the complementary (opponent) market
        ev = result["event_ticker"]
        if ev in seen_events:
            continue
        seen_events.add(ev)
        parsed.append(result)

    log.info("Parsed %d match markets from %d raw markets", len(parsed), len(markets))
    return parsed

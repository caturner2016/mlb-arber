import re
import logging

log = logging.getLogger(__name__)

# Patterns like "Djokovic vs Alcaraz", "Swiatek v Sabalenka", "Murray beat Federer"
_VS_RE = re.compile(
    r"([A-Z][a-zA-Z\-']+(?:\s[A-Z][a-zA-Z\-']+)*)"
    r"\s+(?:vs\.?|v\.?|beat|beats|def\.?)\s+"
    r"([A-Z][a-zA-Z\-']+(?:\s[A-Z][a-zA-Z\-']+)*)",
    re.IGNORECASE,
)

# Surface keywords in title/subtitle
_SURFACE_RE = re.compile(r"\b(clay|grass|hard|carpet|indoor)\b", re.IGNORECASE)

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


def _infer_surface(text: str) -> str:
    text_lower = text.lower()
    m = _SURFACE_RE.search(text_lower)
    if m:
        s = m.group(1).lower()
        return "hard" if s == "carpet" or s == "indoor" else s
    for keyword, surface in _SURFACE_MAP.items():
        if keyword in text_lower:
            return surface
    return "hard"  # default


def parse_match_market(market: dict) -> dict | None:
    """
    Given a Kalshi market dict, try to extract:
      - player_yes: name of player that YES resolves to
      - player_no:  opponent
      - surface:    inferred surface
    Returns None if market doesn't look like a 1v1 match winner market.
    """
    title = market.get("title", "") or ""
    subtitle = market.get("subtitle", "") or ""
    full_text = f"{title} {subtitle}"

    m = _VS_RE.search(full_text)
    if not m:
        return None

    player_yes, player_no = m.group(1).strip(), m.group(2).strip()

    # Skip if either "player" is a generic word
    generic = {"match", "game", "set", "round", "final", "winner", "player"}
    if player_yes.lower() in generic or player_no.lower() in generic:
        return None

    surface = _infer_surface(full_text)

    yes_ask = market.get("yes_ask", 0) or 0
    no_ask = market.get("no_ask", 0) or 0
    yes_bid = market.get("yes_bid", 0) or 0

    # Use mid-price as our reference
    yes_mid = (yes_bid + yes_ask) / 2 if yes_ask else yes_bid

    if yes_mid <= 0 or yes_mid >= 100:
        return None

    return {
        "ticker": market["ticker"],
        "title": title,
        "player_yes": player_yes,
        "player_no": player_no,
        "surface": surface,
        "yes_price_cents": round(yes_mid),
        "yes_ask_cents": yes_ask,
        "no_ask_cents": no_ask,
        "close_time": market.get("close_time", ""),
    }


def parse_all_tennis_markets(markets: list[dict]) -> list[dict]:
    parsed = []
    for m in markets:
        result = parse_match_market(m)
        if result:
            parsed.append(result)
    log.info("Parsed %d match markets from %d raw markets", len(parsed), len(markets))
    return parsed

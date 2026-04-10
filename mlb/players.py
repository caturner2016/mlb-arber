"""Build a player name → MLB player ID mapping (pre-loaded at startup)."""

from __future__ import annotations

import requests
from datetime import date


MLB_API = "https://statsapi.mlb.com/api/v1"


def get_player_id_map(season: int | None = None) -> dict[str, int]:
    """
    Return {full_name: player_id} for all active MLB players.

    Also indexes by last name and "First Last" variants to handle
    minor Kalshi name formatting differences.
    """
    yr = season or date.today().year
    url = f"{MLB_API}/sports/1/players"
    params = {"season": yr, "gameType": "R"}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()

    mapping: dict[str, int] = {}
    for player in resp.json().get("people", []):
        pid = player["id"]
        full = player.get("fullName", "")
        first = player.get("firstName", "")
        last = player.get("lastName", "")

        if full:
            mapping[full.lower()] = pid
        if first and last:
            mapping[f"{first} {last}".lower()] = pid
            # Some Kalshi markets use "F. Last" format
            mapping[f"{first[0]}. {last}".lower()] = pid

    return mapping


def resolve_player(name: str, mapping: dict[str, int]) -> int | None:
    """Case-insensitive lookup, stripping common suffixes (Jr., III, etc.)."""
    clean = name.lower().strip()
    if clean in mapping:
        return mapping[clean]
    # Strip suffixes
    for suffix in (" jr.", " sr.", " ii", " iii", " iv"):
        if clean.endswith(suffix):
            trimmed = clean[: -len(suffix)].strip()
            if trimmed in mapping:
                return mapping[trimmed]
    return None


if __name__ == "__main__":
    m = get_player_id_map()
    print(f"Loaded {len(m)} player name entries")
    for name in ("Aaron Judge", "Shohei Ohtani", "Freddie Freeman"):
        pid = resolve_player(name, m)
        print(f"  {name} → {pid}")

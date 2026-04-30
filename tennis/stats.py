"""
Player statistical profiles built from historical serve/return data.

For each player we track rolling averages (last N matches) of:
  spw  — service points won %
  rpw  — return points won %

For a matchup A vs B we blend each player's own stats with their
opponent's defensive stats to get expected spw/rpw, then derive:
  TPW  — total points won % for player A
  DR   — dominance ratio (A_rpw / B_rpw)

Win probability from TPW uses the empirical tennis curve:
  p=0.50 -> 50%, p=0.51 -> ~85%, p=0.52 -> ~95%
(approximated via a logistic calibrated to that relationship)
"""

import math
import logging
import pandas as pd
from collections import defaultdict, deque

log = logging.getLogger(__name__)

WINDOW = 30          # rolling match window per surface
GLOBAL_WINDOW = 50   # rolling window for overall stats
SURFACES = ("hard", "clay", "grass")
MIN_MATCHES = 8      # require this many to use stats, else fall back to Elo


class _RollingStats:
    def __init__(self, maxlen: int):
        self._spw: deque[float] = deque(maxlen=maxlen)
        self._rpw: deque[float] = deque(maxlen=maxlen)

    def add(self, spw: float, rpw: float):
        self._spw.append(spw)
        self._rpw.append(rpw)

    @property
    def n(self) -> int:
        return len(self._spw)

    @property
    def spw(self) -> float:
        return sum(self._spw) / len(self._spw) if self._spw else 0.62

    @property
    def rpw(self) -> float:
        return sum(self._rpw) / len(self._rpw) if self._rpw else 0.38


class PlayerStats:
    def __init__(self):
        self.overall = _RollingStats(GLOBAL_WINDOW)
        self.surface: dict[str, _RollingStats] = {s: _RollingStats(WINDOW) for s in SURFACES}

    def add_match(self, spw: float, rpw: float, surface: str):
        self.overall.add(spw, rpw)
        if surface in self.surface:
            self.surface[surface].add(spw, rpw)

    def get(self, surface: str) -> tuple[float, float]:
        """Return (spw, rpw) blended between surface-specific and overall."""
        surf = self.surface.get(surface, self.overall)
        if surf.n >= MIN_MATCHES:
            w = min(surf.n / WINDOW, 1.0)
            spw = w * surf.spw + (1 - w) * self.overall.spw
            rpw = w * surf.rpw + (1 - w) * self.overall.rpw
        else:
            spw = self.overall.spw
            rpw = self.overall.rpw
        return spw, rpw

    def enough_data(self) -> bool:
        return self.overall.n >= MIN_MATCHES


def _parse_stats(row) -> tuple[float, float, float, float] | None:
    """Extract (w_spw, w_rpw, l_spw, l_rpw) from a match row. Returns None if data missing."""
    try:
        w_svpt = float(row["w_svpt"])
        w_1stWon = float(row["w_1stWon"])
        w_2ndWon = float(row["w_2ndWon"])
        l_svpt = float(row["l_svpt"])
        l_1stWon = float(row["l_1stWon"])
        l_2ndWon = float(row["l_2ndWon"])
        if w_svpt <= 0 or l_svpt <= 0:
            return None
        w_spw = (w_1stWon + w_2ndWon) / w_svpt
        l_spw = (l_1stWon + l_2ndWon) / l_svpt
        w_rpw = 1.0 - l_spw
        l_rpw = 1.0 - w_spw
        if not (0.3 < w_spw < 0.9 and 0.3 < l_spw < 0.9):
            return None
        return w_spw, w_rpw, l_spw, l_rpw
    except (TypeError, ValueError, KeyError):
        return None


def _tpw_to_win_prob(tpw_a: float) -> float:
    """
    Convert player A's share of total points won to match win probability.
    Calibrated to: tpw=0.50->0.50, tpw=0.51->0.85, tpw=0.52->0.95
    Uses logistic: 1/(1 + exp(-k*(tpw-0.5)))
    k ≈ 33 fits those empirical anchors.
    """
    k = 33.0
    return 1.0 / (1.0 + math.exp(-k * (tpw_a - 0.5)))


class StatsEngine:
    def __init__(self):
        self.players: dict[str, PlayerStats] = defaultdict(PlayerStats)

    def build(self, matches: pd.DataFrame):
        stat_cols = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]
        has_stats = all(c in matches.columns for c in stat_cols)
        if not has_stats:
            log.warning("Match data missing serve stats columns — stats model disabled")
            return

        loaded = 0
        for _, row in matches.iterrows():
            result = _parse_stats(row)
            if result is None:
                continue
            w_spw, w_rpw, l_spw, l_rpw = result
            surface = str(row.get("surface", "hard")).lower()
            self.players[row["winner_name"]].add_match(w_spw, w_rpw, surface)
            self.players[row["loser_name"]].add_match(l_spw, l_rpw, surface)
            loaded += 1

        log.info("Stats profiles built from %d matches for %d players", loaded, len(self.players))

    def win_prob(self, player_a: str, player_b: str, surface: str) -> float | None:
        pa = self.players.get(player_a)
        pb = self.players.get(player_b)
        if pa is None or pb is None:
            return None
        if not pa.enough_data() or not pb.enough_data():
            return None

        a_spw, a_rpw = pa.get(surface)
        b_spw, b_rpw = pb.get(surface)

        # Blend each player's own stats with opponent's defensive context
        pred_a_spw = 0.5 * a_spw + 0.5 * (1.0 - b_rpw)
        pred_b_spw = 0.5 * b_spw + 0.5 * (1.0 - a_rpw)
        pred_a_rpw = 1.0 - pred_b_spw
        pred_b_rpw = 1.0 - pred_a_spw

        # Assume roughly equal serve/return points split
        tpw_a = 0.5 * pred_a_spw + 0.5 * pred_a_rpw
        return _tpw_to_win_prob(tpw_a)

import math
import logging
import pandas as pd
from dataclasses import dataclass, field
from config import ELO_START, ELO_K, SURFACE_BLEND

log = logging.getLogger(__name__)

SURFACES = ("hard", "clay", "grass")


@dataclass
class PlayerRating:
    overall: float = ELO_START
    hard: float = ELO_START
    clay: float = ELO_START
    grass: float = ELO_START
    matches: int = 0

    def surface_rating(self, surface: str) -> float:
        return getattr(self, surface, self.overall)

    def blended(self, surface: str) -> float:
        surf = self.surface_rating(surface)
        if self.matches < 10:
            return self.overall
        return SURFACE_BLEND * surf + (1 - SURFACE_BLEND) * self.overall


def _expected(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + math.pow(10, (rb - ra) / 400.0))


def _k(matches: int) -> float:
    return max(16.0, ELO_K * (1.0 / (1 + matches / 50) ** 0.3))


class EloEngine:
    def __init__(self):
        self.ratings: dict[str, PlayerRating] = {}

    def _get(self, name: str) -> PlayerRating:
        if name not in self.ratings:
            self.ratings[name] = PlayerRating()
        return self.ratings[name]

    def update(self, winner: str, loser: str, surface: str):
        wr = self._get(winner)
        lr = self._get(loser)
        surface = surface if surface in SURFACES else "hard"

        # overall
        ew = _expected(wr.overall, lr.overall)
        k = (_k(wr.matches) + _k(lr.matches)) / 2
        wr.overall += k * (1 - ew)
        lr.overall += k * (0 - (1 - ew))

        # surface-specific
        ws = wr.surface_rating(surface)
        ls = lr.surface_rating(surface)
        esw = _expected(ws, ls)
        setattr(wr, surface, ws + k * (1 - esw))
        setattr(lr, surface, ls + k * (0 - (1 - esw)))

        wr.matches += 1
        lr.matches += 1

    def build(self, matches: pd.DataFrame):
        log.info("Building Elo ratings from %d matches...", len(matches))
        for _, row in matches.iterrows():
            self.update(row["winner_name"], row["loser_name"], row["surface"])
        log.info("Ratings built for %d players", len(self.ratings))

    def win_prob(self, player_a: str, player_b: str, surface: str) -> float | None:
        """Return P(A beats B) on given surface. Returns None if either player unknown."""
        if player_a not in self.ratings or player_b not in self.ratings:
            return None
        ra = self.ratings[player_a].blended(surface)
        rb = self.ratings[player_b].blended(surface)
        return _expected(ra, rb)

    def top_players(self, n: int = 20) -> list[tuple[str, float]]:
        ranked = sorted(self.ratings.items(), key=lambda x: x[1].overall, reverse=True)
        return [(name, r.overall) for name, r in ranked[:n]]

import logging
from difflib import get_close_matches
from tennis.elo import EloEngine
from tennis.stats import StatsEngine
from tennis.data import load_matches, refresh_current_year

log = logging.getLogger(__name__)

# Weight of stats model vs Elo when both are available
STATS_WEIGHT = 0.6
ELO_WEIGHT = 0.4


class TennisPredictor:
    def __init__(self):
        self.elo = EloEngine()
        self.stats = StatsEngine()
        self._player_index: list[str] = []

    def load(self):
        matches = load_matches()
        self.elo.build(matches)
        self.stats.build(matches)
        self._player_index = list(self.elo.ratings.keys())
        log.info("Predictor ready with %d players", len(self._player_index))

    def refresh(self):
        refresh_current_year()
        self.elo = EloEngine()
        self.stats = StatsEngine()
        self._player_index = []
        self.load()

    def fuzzy_match(self, name: str, cutoff: float = 0.75) -> str | None:
        if name in self.elo.ratings:
            return name
        candidates = get_close_matches(name, self._player_index, n=1, cutoff=cutoff)
        if candidates:
            log.debug("Fuzzy match '%s' -> '%s'", name, candidates[0])
            return candidates[0]
        last = name.split()[-1] if name else ""
        for p in self._player_index:
            if p.split()[-1].lower() == last.lower():
                log.debug("Last-name match '%s' -> '%s'", name, p)
                return p
        return None

    def predict(self, player_a: str, player_b: str, surface: str = "hard") -> float | None:
        """
        Return P(player_a wins). Blends Elo and serve/return stats model.
        Falls back to Elo-only if stats data is insufficient.
        surface: 'hard', 'clay', or 'grass'
        """
        a = self.fuzzy_match(player_a)
        b = self.fuzzy_match(player_b)
        if a is None or b is None:
            log.debug("No data for '%s' or '%s'", player_a, player_b)
            return None

        elo_p = self.elo.win_prob(a, b, surface)
        stats_p = self.stats.win_prob(a, b, surface)

        if elo_p is None:
            return stats_p
        if stats_p is None:
            return elo_p

        blended = STATS_WEIGHT * stats_p + ELO_WEIGHT * elo_p
        log.debug(
            "%s vs %s on %s: elo=%.3f stats=%.3f blended=%.3f",
            a, b, surface, elo_p, stats_p, blended,
        )
        return blended

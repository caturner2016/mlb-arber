import logging
from difflib import get_close_matches
from tennis.elo import EloEngine
from tennis.data import load_matches, refresh_current_year

log = logging.getLogger(__name__)


class TennisPredictor:
    def __init__(self):
        self.engine = EloEngine()
        self._player_index: list[str] = []

    def load(self):
        matches = load_matches()
        self.engine.build(matches)
        self._player_index = list(self.engine.ratings.keys())
        log.info("Predictor ready with %d players", len(self._player_index))

    def refresh(self):
        refresh_current_year()
        self.engine = EloEngine()
        self._player_index = []
        self.load()

    def fuzzy_match(self, name: str, cutoff: float = 0.75) -> str | None:
        """Find closest player name in our database."""
        if name in self.engine.ratings:
            return name
        candidates = get_close_matches(name, self._player_index, n=1, cutoff=cutoff)
        if candidates:
            log.debug("Fuzzy match '%s' -> '%s'", name, candidates[0])
            return candidates[0]
        # try last-name matching
        last = name.split()[-1] if name else ""
        for p in self._player_index:
            if p.split()[-1].lower() == last.lower():
                log.debug("Last-name match '%s' -> '%s'", name, p)
                return p
        return None

    def predict(self, player_a: str, player_b: str, surface: str = "hard") -> float | None:
        """
        Return P(player_a wins) using fuzzy name matching.
        surface: 'hard', 'clay', or 'grass'
        """
        a = self.fuzzy_match(player_a)
        b = self.fuzzy_match(player_b)
        if a is None or b is None:
            log.debug("No Elo data for '%s' or '%s'", player_a, player_b)
            return None
        return self.engine.win_prob(a, b, surface)

"""
Train pregame MLB prop models.

Models trained:
  - HitModel:  P(batter gets ≥1 hit today)
  - HRModel:   P(batter hits ≥1 HR today)
  - K6Model:   P(pitcher records ≥6 Ks today)
  - K7Model:   P(pitcher records ≥7 Ks today)

Each model is a calibrated GradientBoostingClassifier
saved to model/saved/{name}.joblib

Run once (takes 10-20 minutes due to MLB API rate limiting):
  python -m model.train

After that, daily predictions are near-instant.
"""

from __future__ import annotations

import logging
import pickle
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from model.data import pull_training_data, get_all_player_ids, get_player_hand
from model.features import (
    build_hit_features, build_hr_features, build_k_features,
    label_hit, label_hr, label_k, team_k_rate,
)

log = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).parent / "saved"
SAVE_DIR.mkdir(exist_ok=True)

TRAIN_SEASONS = [2022, 2023, 2024]
MIN_AB        = 3   # minimum at-bats in game to include (excludes very short appearances)


def _make_estimator(n_features: int) -> Pipeline:
    """
    Gradient boosting + isotonic calibration.
    Produces well-calibrated probabilities.
    """
    gbm = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    calibrated = CalibratedClassifierCV(gbm, method="isotonic", cv=5)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  calibrated),
    ])


def build_hit_training_data(
    batter_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
    player_hand_map: dict,   # player_id → {'bat': 'L'/'R', 'pitch': 'L'/'R'}
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build (X, y) for hit model.
    One row per batter per game, using only pre-game data as features.
    """
    log.info("Building hit training data…")
    rows_X = []
    rows_y = []

    # Only process batters with sufficient data
    batter_ids = batter_df[batter_df["atBats"] >= MIN_AB]["player_id"].unique()

    for i, pid in enumerate(batter_ids):
        if i % 200 == 0:
            log.info(f"  Hit features: {i}/{len(batter_ids)}")

        player_games = batter_df[batter_df["player_id"] == pid].sort_values("date")
        if len(player_games) < 10:
            continue

        batter_hand = player_hand_map.get(pid, {}).get("bat", "R")

        for _, game_row in player_games.iterrows():
            if game_row["atBats"] < MIN_AB:
                continue

            game_date = game_row["date"].date() if hasattr(game_row["date"], "date") else game_row["date"]
            venue_id  = int(game_row.get("venue_id", 0) or 0)
            is_home   = bool(game_row.get("is_home", 0))
            opp_team  = int(game_row.get("opponent_team_id", 0) or 0)

            # Identify opposing pitcher (approximate: pitcher for opp team with most IP on this date)
            opp_pitcher_id, pitcher_hand = _find_starter(pitcher_df, opp_team, game_date, player_hand_map)

            try:
                feats = build_hit_features(
                    batter_logs=batter_df,
                    pitcher_logs=pitcher_df,
                    batter_id=pid,
                    pitcher_id=opp_pitcher_id,
                    batter_hand=batter_hand,
                    pitcher_hand=pitcher_hand,
                    game_date=game_date,
                    venue_id=venue_id,
                    is_home=is_home,
                )
            except Exception:
                continue

            rows_X.append(feats)
            rows_y.append(label_hit(game_row))

    X = pd.DataFrame(rows_X)
    y = pd.Series(rows_y)
    log.info(f"  Hit dataset: {len(X):,} rows  hit_rate={y.mean():.3f}")
    return X, y


def build_hr_training_data(
    batter_df: pd.DataFrame,
    pitcher_df: pd.DataFrame,
    player_hand_map: dict,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for HR model."""
    log.info("Building HR training data…")
    rows_X = []
    rows_y = []

    batter_ids = batter_df[batter_df["atBats"] >= MIN_AB]["player_id"].unique()

    for i, pid in enumerate(batter_ids):
        if i % 200 == 0:
            log.info(f"  HR features: {i}/{len(batter_ids)}")

        player_games = batter_df[batter_df["player_id"] == pid].sort_values("date")
        if len(player_games) < 10:
            continue

        batter_hand = player_hand_map.get(pid, {}).get("bat", "R")

        for _, game_row in player_games.iterrows():
            if game_row["atBats"] < MIN_AB:
                continue

            game_date = game_row["date"].date() if hasattr(game_row["date"], "date") else game_row["date"]
            venue_id  = int(game_row.get("venue_id", 0) or 0)
            is_home   = bool(game_row.get("is_home", 0))
            opp_team  = int(game_row.get("opponent_team_id", 0) or 0)

            opp_pitcher_id, pitcher_hand = _find_starter(pitcher_df, opp_team, game_date, player_hand_map)

            try:
                feats = build_hr_features(
                    batter_logs=batter_df,
                    pitcher_logs=pitcher_df,
                    batter_id=pid,
                    pitcher_id=opp_pitcher_id,
                    batter_hand=batter_hand,
                    pitcher_hand=pitcher_hand,
                    game_date=game_date,
                    venue_id=venue_id,
                    is_home=is_home,
                )
            except Exception:
                continue

            rows_X.append(feats)
            rows_y.append(label_hr(game_row))

    X = pd.DataFrame(rows_X)
    y = pd.Series(rows_y)
    log.info(f"  HR dataset: {len(X):,} rows  hr_rate={y.mean():.3f}")
    return X, y


def build_k_training_data(
    pitcher_df: pd.DataFrame,
    batter_df: pd.DataFrame,
    player_hand_map: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build (X, y6, y7) for strikeout model."""
    log.info("Building K training data…")
    rows_X = []
    rows_y6 = []
    rows_y7 = []

    # Only pitchers with sufficient starts
    pitcher_ids = pitcher_df[
        (pitcher_df.get("inningsPitched", pd.Series([0])) >= 4) |
        (pitcher_df.get("strikeOuts", pd.Series([0])) >= 4)
    ]["player_id"].unique() if "inningsPitched" in pitcher_df.columns else pitcher_df["player_id"].unique()

    for i, pid in enumerate(pitcher_ids):
        if i % 100 == 0:
            log.info(f"  K features: {i}/{len(pitcher_ids)}")

        pitcher_games = pitcher_df[pitcher_df["player_id"] == pid].sort_values("date")
        # Only starting appearances (>= 4 IP or >= 60 pitches)
        starter_games = pitcher_games[
            pitcher_games.get("inningsPitched", pd.Series([0])) >= 4.0
        ] if "inningsPitched" in pitcher_games.columns else pitcher_games

        if len(starter_games) < 5:
            continue

        pitcher_hand = player_hand_map.get(pid, {}).get("pitch", "R")

        for _, game_row in starter_games.iterrows():
            game_date = game_row["date"].date() if hasattr(game_row["date"], "date") else game_row["date"]
            venue_id  = int(game_row.get("venue_id", 0) or 0)
            is_home   = bool(game_row.get("is_home", 0))
            opp_team  = int(game_row.get("opponent_team_id", 0) or 0)

            opp_k_rate = team_k_rate(batter_df, opp_team, game_date)

            try:
                feats = build_k_features(
                    pitcher_logs=pitcher_df,
                    pitcher_id=pid,
                    opp_team_k_rate=opp_k_rate,
                    pitcher_hand=pitcher_hand,
                    game_date=game_date,
                    venue_id=venue_id,
                    is_home=is_home,
                )
            except Exception:
                continue

            rows_X.append(feats)
            rows_y6.append(label_k(game_row, 6))
            rows_y7.append(label_k(game_row, 7))

    X  = pd.DataFrame(rows_X)
    y6 = pd.Series(rows_y6)
    y7 = pd.Series(rows_y7)
    log.info(f"  K dataset: {len(X):,} rows  6k_rate={y6.mean():.3f}  7k_rate={y7.mean():.3f}")
    return X, y6, y7


def _find_starter(
    pitcher_df: pd.DataFrame,
    opp_team_id: int,
    game_date: date,
    player_hand_map: dict,
) -> tuple[int, str]:
    """
    Find the probable starting pitcher for a team on a given date.
    Uses the pitcher who threw the most innings for that team on that date.
    Returns (pitcher_id, hand).
    """
    game_ts = pd.Timestamp(game_date)
    mask = (
        (pitcher_df["team_id"] == opp_team_id) &
        (pitcher_df["date"] == game_ts)
    )
    sub = pitcher_df[mask]
    if sub.empty or "inningsPitched" not in sub.columns:
        return 0, "R"  # unknown pitcher — fallback

    starter_row = sub.sort_values("inningsPitched", ascending=False).iloc[0]
    pid  = int(starter_row["player_id"])
    hand = player_hand_map.get(pid, {}).get("pitch", "R")
    return pid, hand


def train_all(seasons: list[int] = None) -> None:
    """Full training pipeline. Saves models to model/saved/."""
    if seasons is None:
        seasons = TRAIN_SEASONS

    batter_df, pitcher_df = pull_training_data(seasons)

    # Build player hand map for all players across all seasons
    all_players = pd.concat([get_all_player_ids(s) for s in seasons]).drop_duplicates("player_id")
    player_hand_map = {
        row["player_id"]: {"bat": row["bat_side"], "pitch": row["pitch_hand"]}
        for _, row in all_players.iterrows()
    }

    # ── Hit model ─────────────────────────────────────────────────────────────
    X_hit, y_hit = build_hit_training_data(batter_df, pitcher_df, player_hand_map)
    if len(X_hit) > 0:
        model_hit = _make_estimator(X_hit.shape[1])
        model_hit.fit(X_hit, y_hit)
        joblib.dump({"model": model_hit, "features": list(X_hit.columns)},
                    SAVE_DIR / "hit_model.joblib")
        train_acc = model_hit.predict_proba(X_hit)[:, 1].mean()
        log.info(f"Hit model saved  mean_pred={train_acc:.3f}  actual={y_hit.mean():.3f}")

    # ── HR model ──────────────────────────────────────────────────────────────
    X_hr, y_hr = build_hr_training_data(batter_df, pitcher_df, player_hand_map)
    if len(X_hr) > 0:
        model_hr = _make_estimator(X_hr.shape[1])
        model_hr.fit(X_hr, y_hr)
        joblib.dump({"model": model_hr, "features": list(X_hr.columns)},
                    SAVE_DIR / "hr_model.joblib")
        log.info(f"HR model saved  actual_rate={y_hr.mean():.3f}")

    # ── K models ──────────────────────────────────────────────────────────────
    X_k, y6, y7 = build_k_training_data(pitcher_df, batter_df, player_hand_map)
    if len(X_k) > 0:
        for threshold, y_k, name in [(6, y6, "k6_model"), (7, y7, "k7_model")]:
            model_k = _make_estimator(X_k.shape[1])
            model_k.fit(X_k, y_k)
            joblib.dump({"model": model_k, "features": list(X_k.columns)},
                        SAVE_DIR / f"{name}.joblib")
            log.info(f"K{threshold} model saved  actual_rate={y_k.mean():.3f}")

    log.info("All models saved to model/saved/")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )
    train_all()

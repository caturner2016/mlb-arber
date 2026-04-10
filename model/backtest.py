"""
Backtest the pregame model on held-out data.

Reports:
  1. Overall win rate by model and Kalshi price threshold
  2. The "UFC edges" — simple subgroup rules with 60%+ win rates
  3. ROI simulation: if you bet $10 every time model says P > threshold

Run:
  python -m model.backtest
  python -m model.backtest --season 2024   # test on full 2024 season

The goal: find the specific spots (like UFC age gap, reach advantage)
where you have a systematic, repeatable edge.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from model.data import pull_training_data, get_all_player_ids
from model.features import (
    build_hit_features, build_hr_features, build_k_features,
    label_hit, label_hr, label_k, team_k_rate,
)
from model.train import (
    build_hit_training_data, build_hr_training_data, build_k_training_data,
    _find_starter,
)

log = logging.getLogger(__name__)

SAVE_DIR = Path(__file__).parent / "saved"


def _load_model(name: str):
    path = SAVE_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Run `python -m model.train` first.")
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def find_edge_rules(X: pd.DataFrame, y: pd.Series, probs: np.ndarray, model_name: str) -> pd.DataFrame:
    """
    Find the "UFC edge" rules — subgroups where win rate is consistently 60%+.
    Tests single features and common pairs.

    Returns DataFrame sorted by win rate descending.
    """
    df = X.copy()
    df["_y"]    = y.values
    df["_prob"] = probs

    rules = []

    # ── Single feature thresholds ──────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if not c.startswith("_")]

    for col in numeric_cols:
        for pct in [50, 65, 75, 85]:
            thresh = np.percentile(df[col], pct)
            mask   = df[col] >= thresh
            n      = mask.sum()
            if n < 30:
                continue
            win_rate = df.loc[mask, "_y"].mean()
            rules.append({
                "rule":      f"{col} ≥ {thresh:.3f}",
                "win_rate":  win_rate,
                "n":         int(n),
                "pct":       pct,
                "model":     model_name,
            })

    # ── Model probability threshold (the main signal) ─────────────────────────
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = df["_prob"] >= thresh
        n    = mask.sum()
        if n < 20:
            continue
        win_rate = df.loc[mask, "_y"].mean()
        rules.append({
            "rule":     f"model_prob ≥ {thresh:.0%}",
            "win_rate": win_rate,
            "n":        int(n),
            "pct":      int(thresh * 100),
            "model":    model_name,
        })

    # ── Key compound rules (baseball-specific) ────────────────────────────────
    if "platoon_factor" in df.columns and "batter_hit_rate_30d" in df.columns:
        mask = (df["platoon_factor"] > 0) & (df["batter_hit_rate_30d"] >= 0.270)
        n    = mask.sum()
        if n >= 20:
            rules.append({
                "rule":     "platoon_advantage AND hit_rate_30d ≥ .270",
                "win_rate": df.loc[mask, "_y"].mean(),
                "n":        int(n), "pct": 0, "model": model_name,
            })

    if "pitcher_k9_30d" in df.columns and "opp_team_k_rate_30d" in df.columns:
        mask = (df["pitcher_k9_30d"] >= 9.5) & (df["opp_team_k_rate_30d"] >= 0.23)
        n    = mask.sum()
        if n >= 20:
            rules.append({
                "rule":     "pitcher_K9 ≥ 9.5 AND opp_K% ≥ 23%",
                "win_rate": df.loc[mask, "_y"].mean(),
                "n":        int(n), "pct": 0, "model": model_name,
            })

    if "park_hr_factor" in df.columns and "batter_hr_rate_30d" in df.columns:
        mask = (df["park_hr_factor"] >= 103) & (df["batter_hr_rate_30d"] >= 0.04)
        n    = mask.sum()
        if n >= 20:
            rules.append({
                "rule":     "hitter park (HR factor ≥ 103) AND batter HR rate ≥ 4%",
                "win_rate": df.loc[mask, "_y"].mean(),
                "n":        int(n), "pct": 0, "model": model_name,
            })

    result = pd.DataFrame(rules)
    if result.empty:
        return result
    return result[result["win_rate"] >= 0.55].sort_values("win_rate", ascending=False)


def roi_sim(probs: np.ndarray, y: np.ndarray, bet_amount: float, threshold: float) -> dict:
    """
    Simulate flat betting $bet_amount whenever model_prob >= threshold.
    Assumes Kalshi pays $1.00 on YES contract.
    """
    mask     = probs >= threshold
    n_bets   = mask.sum()
    if n_bets == 0:
        return {"n_bets": 0, "win_rate": 0, "roi_pct": 0, "profit": 0}
    wins      = y[mask].sum()
    win_rate  = wins / n_bets
    # Average cost approximation: model prob ≈ fair value,
    # assume Kalshi prices it at (prob - 0.05) due to lag
    avg_cost  = float((probs[mask] - 0.05).mean())
    avg_cost  = max(0.05, min(0.95, avg_cost))
    profit    = wins * (1.0 - avg_cost) * bet_amount - (n_bets - wins) * avg_cost * bet_amount
    roi_pct   = profit / (n_bets * avg_cost * bet_amount) * 100
    return {
        "n_bets":   int(n_bets),
        "win_rate": float(win_rate),
        "roi_pct":  float(roi_pct),
        "profit":   float(profit),
    }


def run_backtest(test_season: int = 2024) -> None:
    """
    Trains on all seasons < test_season, evaluates on test_season.
    """
    train_seasons = [s for s in [2022, 2023, 2024] if s < test_season]
    if not train_seasons:
        train_seasons = [2022, 2023]

    log.info(f"Backtest: train={train_seasons}  test={test_season}")

    # Pull data
    train_bat, train_pit = pull_training_data(train_seasons)
    test_bat,  test_pit  = pull_training_data([test_season])

    all_players = pd.concat([
        get_all_player_ids(s) for s in train_seasons + [test_season]
    ]).drop_duplicates("player_id")
    hand_map = {
        row["player_id"]: {"bat": row["bat_side"], "pitch": row["pitch_hand"]}
        for _, row in all_players.iterrows()
    }

    # ── Hit model ─────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("HIT MODEL BACKTEST")
    print("═" * 60)

    X_train, y_train = build_hit_training_data(train_bat, train_pit, hand_map)
    X_test,  y_test  = build_hit_training_data(test_bat,  test_pit,  hand_map)

    if len(X_train) > 0 and len(X_test) > 0:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from model.train import _make_estimator

        model = _make_estimator(X_train.shape[1])
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        print(f"  Test size:  {len(X_test):,} games")
        print(f"  Actual hit rate: {y_test.mean():.3f}")
        print(f"  AUC:        {roc_auc_score(y_test, probs):.4f}")
        print(f"  Brier score: {brier_score_loss(y_test, probs):.4f}")

        print("\n  ROI at different confidence thresholds:")
        for thresh in [0.60, 0.65, 0.70, 0.75]:
            r = roi_sim(probs, y_test.values, bet_amount=10, threshold=thresh)
            print(f"    P ≥ {thresh:.0%}:  {r['n_bets']:5d} bets  "
                  f"win={r['win_rate']:.1%}  ROI={r['roi_pct']:+.1f}%  "
                  f"profit=${r['profit']:+.0f}")

        print("\n  Top edge rules (win rate ≥ 55%):")
        rules = find_edge_rules(X_test, y_test, probs, "hit")
        for _, row in rules.head(10).iterrows():
            print(f"    {row['win_rate']:.1%}  (n={row['n']:4d})  {row['rule']}")

    # ── HR model ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("HR MODEL BACKTEST")
    print("═" * 60)

    X_train, y_train = build_hr_training_data(train_bat, train_pit, hand_map)
    X_test,  y_test  = build_hr_training_data(test_bat,  test_pit,  hand_map)

    if len(X_train) > 0 and len(X_test) > 0:
        model = _make_estimator(X_train.shape[1])
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        print(f"  Test size:  {len(X_test):,} games")
        print(f"  Actual HR rate: {y_test.mean():.3f}")
        print(f"  AUC:        {roc_auc_score(y_test, probs):.4f}")

        print("\n  ROI at different confidence thresholds:")
        for thresh in [0.08, 0.12, 0.15, 0.20]:
            r = roi_sim(probs, y_test.values, bet_amount=10, threshold=thresh)
            print(f"    P ≥ {thresh:.0%}:  {r['n_bets']:5d} bets  "
                  f"win={r['win_rate']:.1%}  ROI={r['roi_pct']:+.1f}%  "
                  f"profit=${r['profit']:+.0f}")

        print("\n  Top edge rules:")
        rules = find_edge_rules(X_test, y_test, probs, "hr")
        for _, row in rules.head(10).iterrows():
            print(f"    {row['win_rate']:.1%}  (n={row['n']:4d})  {row['rule']}")

    # ── K model ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("STRIKEOUT MODEL BACKTEST (6+ and 7+ Ks)")
    print("═" * 60)

    X_train, y6_train, y7_train = build_k_training_data(train_pit, train_bat, hand_map)
    X_test,  y6_test,  y7_test  = build_k_training_data(test_pit,  test_bat,  hand_map)

    if len(X_train) > 0 and len(X_test) > 0:
        for thresh_name, y_train, y_test in [("6+K", y6_train, y6_test), ("7+K", y7_train, y7_test)]:
            model = _make_estimator(X_train.shape[1])
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

            print(f"\n  [{thresh_name}]  actual_rate={y_test.mean():.3f}  "
                  f"AUC={roc_auc_score(y_test, probs):.4f}")

            for thresh in [0.50, 0.60, 0.65, 0.70]:
                r = roi_sim(probs, y_test.values, bet_amount=10, threshold=thresh)
                print(f"    P ≥ {thresh:.0%}:  {r['n_bets']:4d} bets  "
                      f"win={r['win_rate']:.1%}  ROI={r['roi_pct']:+.1f}%  "
                      f"profit=${r['profit']:+.0f}")

        print("\n  Top K edge rules:")
        rules = find_edge_rules(X_test, y6_test, probs, "k6")
        for _, row in rules.head(10).iterrows():
            print(f"    {row['win_rate']:.1%}  (n={row['n']:4d})  {row['rule']}")

    print("\n" + "═" * 60)
    print("BACKTEST COMPLETE")
    print("═" * 60)


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2024, help="Season to test on (default: 2024)")
    args = parser.parse_args()
    run_backtest(test_season=args.season)

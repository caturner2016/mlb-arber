"""
Daily pregame predictions — run each morning before games start.

Outputs ranked list of today's best bets with:
  - Player, prop type
  - Model probability
  - Current Kalshi ask price
  - Edge (model prob - Kalshi price)
  - Why: which factors are driving the call

Also integrates with the Kalshi client to place bets automatically
when edge > MIN_EDGE_CENTS and YES ask is available.

Run manually:
  python -m model.predict
  python -m model.predict --dry-run   # log but don't bet
  python -m model.predict --min-edge 8  # only bet when edge >= 8 cents
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import yaml

from model.data import (
    get_all_player_ids, get_player_hand, get_todays_matchups, get_todays_lineups,
    get_batter_game_logs, get_pitcher_game_logs, pull_training_data,
)
from model.features import (
    build_hit_features, build_hr_features, build_k_features,
    team_k_rate,
)

log = logging.getLogger(__name__)

SAVE_DIR  = Path(__file__).parent / "saved"
MIN_EDGE  = 8    # cents — only bet when model is ≥ 8¢ above Kalshi ask
MIN_PROB  = 0.55 # minimum model probability to even consider a bet


def _load_model(name: str):
    path = SAVE_DIR / f"{name}.joblib"
    if not path.exists():
        return None, None
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def _explain_hit(feats: dict) -> str:
    parts = []
    hr = feats.get("batter_hit_rate_30d", 0)
    if hr >= 0.300:
        parts.append(f"hot bat (.{int(hr*1000):03d} L30)")
    elif hr >= 0.270:
        parts.append(f"solid bat (.{int(hr*1000):03d} L30)")
    if feats.get("platoon_factor", 0) > 0:
        parts.append("platoon adv")
    if feats.get("pitcher_whip_30d", 1.3) > 1.35:
        parts.append(f"weak pitcher (WHIP {feats['pitcher_whip_30d']:.2f})")
    if feats.get("park_run_factor", 100) >= 104:
        parts.append("hitter's park")
    return "  |  ".join(parts) if parts else "model signal"


def _explain_hr(feats: dict) -> str:
    parts = []
    if feats.get("park_hr_factor", 100) >= 105:
        parts.append(f"HR park ({feats['park_hr_factor']})")
    if feats.get("wind_boost", 0) >= 0.10:
        parts.append(f"wind out")
    hr = feats.get("batter_hr_rate_30d", 0)
    if hr >= 0.05:
        parts.append(f"hot power ({hr:.1%} HR/PA L30)")
    if feats.get("pitcher_hr9_30d", 1.0) >= 1.4:
        parts.append(f"HR-prone pitcher")
    if feats.get("platoon_factor", 0) > 0:
        parts.append("platoon adv")
    return "  |  ".join(parts) if parts else "model signal"


def _explain_k(feats: dict) -> str:
    parts = []
    k9 = feats.get("pitcher_k9_30d", 0)
    if k9 >= 10:
        parts.append(f"elite K rate ({k9:.1f} K/9)")
    elif k9 >= 9:
        parts.append(f"high K rate ({k9:.1f} K/9)")
    opp_k = feats.get("opp_team_k_rate_30d", 0)
    if opp_k >= 0.25:
        parts.append(f"opp whiffs a lot ({opp_k:.0%} K%)")
    rest = feats.get("pitcher_days_rest", 5)
    if rest < 4:
        parts.append(f"⚠ short rest ({rest}d)")
    return "  |  ".join(parts) if parts else "model signal"


class PregamePredictor:
    """
    Loads trained models and generates today's predictions.
    """

    def __init__(self) -> None:
        self.hit_model,  self.hit_feats  = _load_model("hit_model")
        self.hr_model,   self.hr_feats   = _load_model("hr_model")
        self.k6_model,   self.k6_feats   = _load_model("k6_model")
        self.k7_model,   self.k7_feats   = _load_model("k7_model")

        if self.hit_model is None:
            log.warning("Models not found — run `python -m model.train` first")

    def predict_today(
        self,
        batter_df: pd.DataFrame,
        pitcher_df: pd.DataFrame,
        player_hand_map: dict,
        temp_by_venue: dict[int, float] = None,     # venue_id → temp_f
        wind_by_venue: dict[int, tuple] = None,     # venue_id → (speed, out_bool)
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with today's predictions:
          player_id, player_name, prop_type, threshold, model_prob,
          game_pk, venue_id, features, explanation
        """
        matchups = get_todays_matchups()
        if not matchups:
            log.info("No games today")
            return pd.DataFrame()

        lineups = get_todays_lineups()

        rows = []

        for m in matchups:
            game_pk    = m["game_pk"]
            venue_id   = m["venue_id"]
            temp_f     = (temp_by_venue or {}).get(venue_id, 70.0)
            wind_speed, wind_out = (wind_by_venue or {}).get(venue_id, (5.0, False))

            home_pid   = m["home_pitcher_id"]
            away_pid   = m["away_pitcher_id"]
            home_tid   = m["home_team_id"]
            away_tid   = m["away_team_id"]

            # ── Batter predictions ────────────────────────────────────────────
            game_batters = lineups.get(game_pk, [])
            if not game_batters:
                log.debug(f"  No lineup yet for {m['home_team_name']} vs {m['away_team_name']}")

            for batter_id in game_batters:
                hand   = player_hand_map.get(batter_id, {})
                bhand  = hand.get("bat", "R")
                is_home = batter_id in [
                    # simple heuristic: look at the player's team — not perfect without lineup API
                    # We'll approximate: first half of game_batters = away, second = home
                ]

                # Identify pitcher the batter faces
                # (simplified: use home pitcher for away batters and vice versa)
                batter_team_proxy = "home" if batter_id in game_batters[len(game_batters)//2:] else "away"
                opp_pitcher_id    = home_pid if batter_team_proxy == "away" else away_pid
                if opp_pitcher_id is None:
                    continue

                pitcher_hand = player_hand_map.get(opp_pitcher_id, {}).get("pitch", "R")
                is_home_bat  = batter_team_proxy == "home"

                today = date.today()

                # Hit prediction
                if self.hit_model is not None:
                    try:
                        feats = build_hit_features(
                            batter_logs=batter_df,
                            pitcher_logs=pitcher_df,
                            batter_id=batter_id,
                            pitcher_id=opp_pitcher_id,
                            batter_hand=bhand,
                            pitcher_hand=pitcher_hand,
                            game_date=today,
                            venue_id=venue_id,
                            is_home=is_home_bat,
                            temp_f=temp_f,
                            wind_speed=wind_speed,
                            wind_out=wind_out,
                        )
                        X = pd.DataFrame([feats])[self.hit_feats]
                        prob = float(self.hit_model.predict_proba(X)[0, 1])
                        rows.append({
                            "player_id":    batter_id,
                            "prop_type":    "hit",
                            "threshold":    1,
                            "model_prob":   prob,
                            "game_pk":      game_pk,
                            "venue_id":     venue_id,
                            "home_team":    m["home_team_name"],
                            "away_team":    m["away_team_name"],
                            "features":     feats,
                            "explanation":  _explain_hit(feats),
                        })
                    except Exception as e:
                        log.debug(f"Hit feat error batter {batter_id}: {e}")

                # HR prediction
                if self.hr_model is not None:
                    try:
                        feats = build_hr_features(
                            batter_logs=batter_df,
                            pitcher_logs=pitcher_df,
                            batter_id=batter_id,
                            pitcher_id=opp_pitcher_id,
                            batter_hand=bhand,
                            pitcher_hand=pitcher_hand,
                            game_date=today,
                            venue_id=venue_id,
                            is_home=is_home_bat,
                            temp_f=temp_f,
                            wind_speed=wind_speed,
                            wind_out=wind_out,
                        )
                        X = pd.DataFrame([feats])[self.hr_feats]
                        prob = float(self.hr_model.predict_proba(X)[0, 1])
                        rows.append({
                            "player_id":    batter_id,
                            "prop_type":    "hr",
                            "threshold":    1,
                            "model_prob":   prob,
                            "game_pk":      game_pk,
                            "venue_id":     venue_id,
                            "home_team":    m["home_team_name"],
                            "away_team":    m["away_team_name"],
                            "features":     feats,
                            "explanation":  _explain_hr(feats),
                        })
                    except Exception as e:
                        log.debug(f"HR feat error batter {batter_id}: {e}")

            # ── Pitcher predictions ───────────────────────────────────────────
            for pitcher_id, is_home_p, opp_team_id in [
                (home_pid, True,  away_tid),
                (away_pid, False, home_tid),
            ]:
                if pitcher_id is None:
                    continue
                pitcher_hand = player_hand_map.get(pitcher_id, {}).get("pitch", "R")
                opp_k = team_k_rate(batter_df, opp_team_id, date.today())

                for k_thresh, model, feat_list in [
                    (6, self.k6_model, self.k6_feats),
                    (7, self.k7_model, self.k7_feats),
                ]:
                    if model is None:
                        continue
                    try:
                        feats = build_k_features(
                            pitcher_logs=pitcher_df,
                            pitcher_id=pitcher_id,
                            opp_team_k_rate=opp_k,
                            pitcher_hand=pitcher_hand,
                            game_date=date.today(),
                            venue_id=venue_id,
                            is_home=is_home_p,
                            temp_f=temp_f,
                        )
                        X = pd.DataFrame([feats])[feat_list]
                        prob = float(model.predict_proba(X)[0, 1])
                        rows.append({
                            "player_id":   pitcher_id,
                            "prop_type":   "strikeouts",
                            "threshold":   k_thresh,
                            "model_prob":  prob,
                            "game_pk":     game_pk,
                            "venue_id":    venue_id,
                            "home_team":   m["home_team_name"],
                            "away_team":   m["away_team_name"],
                            "features":    feats,
                            "explanation": _explain_k(feats),
                        })
                    except Exception as e:
                        log.debug(f"K feat error pitcher {pitcher_id}: {e}")

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)


async def run_pregame_bets(
    cfg: dict,
    dry_run: bool = True,
    min_edge: int = MIN_EDGE,
) -> None:
    """
    Load models, generate predictions, match to Kalshi, place bets.
    Called from main.py or standalone.
    """
    from kalshi.client import KalshiClient
    from kalshi.props import PropCache
    import requests

    mode  = cfg.get("mode", "paper")
    paper = (mode != "live") or dry_run

    kalshi = KalshiClient(
        key_id=cfg.get("kalshi_key_id", ""),
        private_key_path=cfg.get("kalshi_private_key_path", ""),
        paper_mode=paper,
    )

    tg_token = cfg.get("telegram_token", "")
    tg_chat  = cfg.get("telegram_chat_id", "")

    def tg(msg: str) -> None:
        if not tg_token or not tg_chat:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                json={"chat_id": tg_chat, "text": msg},
                timeout=5,
            )
        except Exception:
            pass

    # Pull data
    seasons    = [2022, 2023, 2024, 2025]
    batter_df, pitcher_df = pull_training_data([2022, 2023, 2024])
    all_players = pd.concat([get_all_player_ids(s) for s in [2024, 2025]]).drop_duplicates("player_id")
    hand_map = {
        row["player_id"]: {"bat": row["bat_side"], "pitch": row["pitch_hand"]}
        for _, row in all_players.iterrows()
    }
    name_map = {row["player_id"]: row["full_name"] for _, row in all_players.iterrows()}

    predictor = PregamePredictor()
    if predictor.hit_model is None:
        log.warning("No models found. Run `python -m model.train` first.")
        await kalshi.close()
        return

    # Build weather context (optional — falls back to defaults if NOAA fails)
    temp_by_venue = {}
    wind_by_venue = {}
    try:
        from weather.noaa import NOAAClient
        noaa      = NOAAClient()
        forecasts = await noaa.fetch_all()
        await noaa.close()
        # Map city forecasts to venue temp (best-effort by team name matching)
        # For now use league average 70°F; venue-city mapping can be added later
    except Exception as e:
        log.debug(f"Weather fetch skipped: {e}")

    # Run predictions
    log.info("Generating pregame predictions…")
    preds = predictor.predict_today(batter_df, pitcher_df, hand_map, temp_by_venue, wind_by_venue)
    if preds.empty:
        log.info("No predictions generated (no lineups yet?)")
        await kalshi.close()
        return

    # Load Kalshi prop cache for matching
    prop_cache = PropCache()
    await prop_cache.build(kalshi)

    model_cfg  = cfg.get("model", {})
    max_buy    = int(model_cfg.get("max_buy_cents", 70))
    max_bet    = float(model_cfg.get("max_bet", 20.0))

    fired:  set[str] = set()
    bets_placed = 0

    # Sort by edge potential (highest model_prob first)
    preds = preds.sort_values("model_prob", ascending=False)

    print(f"\n{'═'*70}")
    print(f"PREGAME PREDICTIONS — {date.today()}")
    print(f"{'═'*70}")

    for _, row in preds.iterrows():
        pid       = int(row["player_id"])
        name      = name_map.get(pid, f"ID:{pid}")
        prop_type = row["prop_type"]
        threshold = int(row["threshold"])
        model_p   = float(row["model_prob"])

        if model_p < MIN_PROB:
            continue

        # Match to Kalshi market
        if prop_type == "hit":
            prop = prop_cache.get_hit_ticker(name, threshold)
        elif prop_type == "hr":
            prop = prop_cache.get_hr_ticker(name)
        else:
            prop = None   # Strikeout tickers need a separate cache (future work)

        if prop is None:
            continue

        yes_ask = prop.yes_ask
        if yes_ask is None:
            continue

        edge = int(model_p * 100) - yes_ask
        if edge < min_edge:
            continue

        fire_key = f"{date.today()}:MODEL:{prop_type}:{threshold}:{pid}"
        if fire_key in fired:
            continue

        print(f"\n  {name:25s}  {prop_type:10s} {threshold}+")
        print(f"    Model: {model_p:.1%}   Kalshi: {yes_ask}¢   Edge: +{edge}¢")
        print(f"    {row['explanation']}")
        print(f"    Ticker: {prop.ticker}")

        # Place the bet
        result = await kalshi.get_yes_ask(prop.ticker, max_cents=max_buy)
        if result is None:
            log.info(f"    Market repriced above {max_buy}¢ — skip")
            continue

        live_ask, qty = result
        balance = await kalshi.get_balance() if not dry_run else max_bet
        spend   = min(balance, max_bet)
        count   = min(qty, int(spend / (live_ask / 100)))

        if count < 1:
            continue

        cost     = count * live_ask / 100
        expected = count * (100 - live_ask) / 100

        print(f"    → BUY {count} @ {live_ask}¢  cost=${cost:.2f}  "
              f"expected=+${expected:.2f}  {'[DRY RUN]' if dry_run else '[LIVE]'}")

        if not dry_run:
            await kalshi.place_order(prop.ticker, "yes", live_ask, count, "limit")
            bets_placed += 1

        fired.add(fire_key)

        tg(
            f"PREGAME BET — {name}\n"
            f"Prop: {prop_type} {threshold}+\n"
            f"Model: {model_p:.1%}  |  Kalshi: {live_ask}¢  |  Edge: +{edge}¢\n"
            f"Contracts: {count} @ {live_ask}¢\n"
            f"Cost: ${cost:.2f}  Expected: +${expected:.2f}\n"
            f"Reason: {row['explanation']}\n"
            f"{'[DRY RUN]' if dry_run else '[LIVE]'}"
        )

    print(f"\n{'═'*70}")
    print(f"Bets placed: {bets_placed}  ({'DRY RUN' if dry_run else 'LIVE'})")
    print(f"{'═'*70}\n")

    await kalshi.close()


if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="Daily pregame bet predictions")
    parser.add_argument("--live",      action="store_true")
    parser.add_argument("--dry-run",   action="store_true", default=True)
    parser.add_argument("--min-edge",  type=int, default=MIN_EDGE)
    parser.add_argument("--config",    default="config.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.live:
        cfg["mode"] = "live"

    asyncio.run(run_pregame_bets(cfg, dry_run=not args.live, min_edge=args.min_edge))

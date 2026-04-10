"""
Feature engineering for the pregame MLB model.

The "UFC edge" equivalents for baseball:

  HITTING (Hit prop):
    - Platoon advantage (LHB vs RHP, or RHB vs LHP): +.025 BA, persistent
    - Batter recent form (L30 day avg): strongest single predictor
    - Opposing pitcher WHIP (L30): hit-allowing rate
    - Park run factor: Coors +15%, Petco -8%
    - Temperature: <50°F suppresses offense

  HOME RUNS (HR prop):
    - Batter ISO/HR rate vs pitcher handedness (L30)
    - Opposing pitcher HR/9 (L30)
    - Park HR factor (biggest single factor: Coors +20%, Petco -12%)
    - Wind: blowing out >10mph = +15-20% HR rate
    - Temperature: <45°F = -10% HR rate

  STRIKEOUTS (K prop for pitchers):
    - Pitcher K/9 rate (L30): most stable predictor in baseball
    - Opposing team K% (L30): some teams strike out 28%, others 18%
    - Pitcher days rest: short rest (<4 days) = worse performance
    - Pitcher hand: doesn't predict K rate directly, but matters for matchup

Each function returns a flat dict of features ready for sklearn.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ── Park factors ───────────────────────────────────────────────────────────────
# venue_id → (run_factor, hr_factor)  [100 = neutral, 110 = +10%]
# Source: multi-year Fangraphs park factors (2022-2024 average)

PARK_FACTORS: dict[int, tuple[int, int]] = {
    19:   (115, 122),  # Coors Field            (COL)  — extreme hitter park
    2602: (107, 112),  # Great American BP       (CIN)
    5325: (104, 107),  # Globe Life Field        (TEX)
    3:    (103, 101),  # Fenway Park             (BOS)
    17:   (101, 103),  # Wrigley Field           (CHC)  — wind dependent
    3313: (100, 105),  # Yankee Stadium          (NYY)
    2889: (100, 102),  # Citizens Bank Park      (PHI)
    1:    ( 99, 100),  # Oriole Park Camden      (BAL)
    7:    ( 99,  98),  # Rogers Centre           (TOR)  — dome
    4169: ( 99,  97),  # loanDepot Park          (MIA)  — dome
    15:   ( 98,  99),  # American Family Field   (MIL)  — retractable
    4321: ( 98,  97),  # Truist Park             (ATL)
    2392: ( 98,  97),  # Angel Stadium           (LAA)
    5:    ( 97,  96),  # Kauffman Stadium        (KC)
    31:   ( 97,  96),  # T-Mobile Park           (SEA)  — dome
    2500: ( 97,  95),  # Busch Stadium           (STL)
    2518: ( 97,  97),  # Minute Maid Park        (HOU)  — dome
    2395: ( 97,  95),  # Nationals Park          (WSH)
    680:  ( 97,  96),  # Dodger Stadium          (LAD)
    14:   ( 96,  95),  # Progressive Field       (CLE)
    4705: ( 96,  96),  # Citi Field              (NYM)
    2681: ( 95,  93),  # PNC Park                (PIT)
    32:   ( 95,  95),  # Chase Field             (ARI)  — retractable
    4403: ( 95,  93),  # Target Field            (MIN)
    2394: ( 95,  94),  # Guaranteed Rate Field   (CWS)
    2593: ( 94,  92),  # Oracle Park             (SF)   — extreme pitcher park
    12:   ( 94,  92),  # Comerica Park           (DET)
    4140: ( 93,  91),  # Tropicana Field         (TB)   — dome
    2680: ( 93,  91),  # Petco Park              (SD)   — extreme pitcher park
    672:  ( 96,  96),  # Oakland Coliseum        (OAK)
}

PARK_DEFAULT = (97, 96)  # neutral-ish fallback


def park_factors(venue_id: int) -> tuple[int, int]:
    """Returns (run_factor, hr_factor) for a venue. 100 = neutral."""
    return PARK_FACTORS.get(venue_id, PARK_DEFAULT)


# ── Rolling stat helpers ───────────────────────────────────────────────────────

def rolling_batter_stats(
    game_logs: pd.DataFrame,
    player_id: int,
    as_of_date: date,
    window_days: int = 30,
) -> dict:
    """
    Compute batter rolling stats in the N days before as_of_date.
    Returns dict with keys: hit_rate, hr_rate, k_rate, bb_rate, ab, games.
    """
    cutoff = pd.Timestamp(as_of_date - timedelta(days=window_days))
    end    = pd.Timestamp(as_of_date)

    mask = (
        (game_logs["player_id"] == player_id) &
        (game_logs["date"] >= cutoff) &
        (game_logs["date"] < end)
    )
    sub = game_logs[mask]

    if sub.empty or sub["atBats"].sum() < 5:
        return {
            "hit_rate":  0.260,  # league average fallback
            "hr_rate":   0.030,
            "k_rate":    0.220,
            "bb_rate":   0.080,
            "ab_30d":    0,
            "games_30d": 0,
        }

    ab  = sub["atBats"].sum()
    pa  = ab + sub.get("baseOnBalls", pd.Series([0])).sum()
    pa  = max(pa, ab)

    return {
        "hit_rate":  sub["hits"].sum() / ab if ab > 0 else 0.260,
        "hr_rate":   sub["homeRuns"].sum() / pa if pa > 0 else 0.030,
        "k_rate":    sub["strikeOuts"].sum() / pa if pa > 0 else 0.220,
        "bb_rate":   sub.get("baseOnBalls", pd.Series([0])).sum() / pa if pa > 0 else 0.080,
        "ab_30d":    int(ab),
        "games_30d": len(sub),
    }


def rolling_pitcher_stats(
    game_logs: pd.DataFrame,
    player_id: int,
    as_of_date: date,
    window_days: int = 30,
) -> dict:
    """
    Pitcher rolling stats in the N days before as_of_date.
    Returns dict with keys: k9, whip, hr9, era, ip, days_rest, last_ip.
    """
    cutoff = pd.Timestamp(as_of_date - timedelta(days=window_days))
    end    = pd.Timestamp(as_of_date)

    mask = (
        (game_logs["player_id"] == player_id) &
        (game_logs["date"] >= cutoff) &
        (game_logs["date"] < end)
    )
    sub = game_logs[mask]

    # Days rest: days since last appearance
    all_before = game_logs[
        (game_logs["player_id"] == player_id) &
        (game_logs["date"] < end)
    ].sort_values("date")

    days_rest = 5  # default (typical rotation)
    last_ip   = 5.0
    if not all_before.empty:
        last_date = all_before["date"].iloc[-1].date()
        days_rest = (as_of_date - last_date).days
        last_ip   = float(all_before["inningsPitched"].iloc[-1]) if "inningsPitched" in all_before.columns else 5.0

    if sub.empty:
        return {
            "k9":       8.0,    # league average fallback
            "whip":     1.30,
            "hr9":      1.20,
            "era":      4.50,
            "ip_30d":   0.0,
            "days_rest": days_rest,
            "last_ip":   last_ip,
        }

    ip   = float(sub["inningsPitched"].sum()) if "inningsPitched" in sub.columns else 0
    ip   = max(ip, 0.1)
    h    = sub["hits"].sum() if "hits" in sub.columns else 0
    bb   = sub["baseOnBalls"].sum() if "baseOnBalls" in sub.columns else 0
    so   = sub["strikeOuts"].sum() if "strikeOuts" in sub.columns else 0
    er   = sub["earnedRuns"].sum() if "earnedRuns" in sub.columns else 0
    hr   = sub["homeRuns"].sum() if "homeRuns" in sub.columns else 0

    return {
        "k9":        so * 9 / ip,
        "whip":      (h + bb) / ip,
        "hr9":       hr * 9 / ip,
        "era":       er * 9 / ip,
        "ip_30d":    ip,
        "days_rest": days_rest,
        "last_ip":   last_ip,
    }


def team_k_rate(
    pitcher_logs: pd.DataFrame,
    team_id: int,
    as_of_date: date,
    window_days: int = 30,
) -> float:
    """Team strikeout rate allowed in last N days (proxy for batter K tendency)."""
    cutoff = pd.Timestamp(as_of_date - timedelta(days=window_days))
    end    = pd.Timestamp(as_of_date)

    # We need batters' strikeout rate. Use opposing pitcher logs where team_id = opponent.
    # Proxy: find pitchers who pitched FOR the opponent team in this window
    mask = (
        (pitcher_logs["opponent_team_id"] == team_id) &
        (pitcher_logs["date"] >= cutoff) &
        (pitcher_logs["date"] < end)
    )
    sub = pitcher_logs[mask]

    if sub.empty or "strikeOuts" not in sub.columns:
        return 0.220  # league average

    ip  = float(sub["inningsPitched"].sum()) if "inningsPitched" in sub.columns else 0
    so  = sub["strikeOuts"].sum()
    # Approx: PAs ≈ IP * 4.3
    pa  = ip * 4.3
    return float(so / pa) if pa > 0 else 0.220


# ── Platoon splits ─────────────────────────────────────────────────────────────

def platoon_factor(batter_hand: str, pitcher_hand: str) -> float:
    """
    Returns the platoon BA adjustment.
    Same-hand (L vs L, R vs R) = disadvantage for batter.
    Opposite-hand = advantage for batter.

    Historical platoon split: ~.025-.040 BA difference.
    """
    if not batter_hand or not pitcher_hand:
        return 0.0

    bats  = batter_hand.upper()[0]   # L, R, or S (switch)
    pitches = pitcher_hand.upper()[0]  # L or R

    if bats == "S":
        return 0.010  # switch hitter — small advantage vs same, neutralized

    if bats != pitches:
        return 0.025   # opposite hand = batter advantage
    else:
        return -0.015  # same hand = batter disadvantage


# ── Weather helpers ────────────────────────────────────────────────────────────

def wind_hr_boost(wind_speed_mph: float, wind_dir_out: bool) -> float:
    """
    HR rate multiplier from wind.
    Out = ball carries more. In = ball dies.
    Research: ~2% per 5mph when blowing out.
    """
    if wind_dir_out:
        return min(0.25, wind_speed_mph * 0.020)   # max +25% HR rate
    else:
        return max(-0.15, -wind_speed_mph * 0.010)  # max -15% HR rate


def temp_offense_factor(temp_f: float) -> float:
    """
    Offensive rate adjustment for temperature.
    Below 50°F: ball doesn't carry, muscles tight.
    Above 85°F: slight boost.
    Research: ~2.5% per 10°F below 65.
    """
    if temp_f < 50:
        return (temp_f - 65) * 0.0025   # negative adjustment
    elif temp_f > 85:
        return (temp_f - 85) * 0.001    # small positive
    return 0.0


# ── Feature builders ───────────────────────────────────────────────────────────

def build_hit_features(
    batter_logs: pd.DataFrame,
    pitcher_logs: pd.DataFrame,
    batter_id: int,
    pitcher_id: int,
    batter_hand: str,
    pitcher_hand: str,
    game_date: date,
    venue_id: int,
    is_home: bool,
    temp_f: float = 70.0,
    wind_speed: float = 5.0,
    wind_out: bool = False,
) -> dict:
    """Feature vector for hit prop (will batter get ≥1 hit?)."""
    b = rolling_batter_stats(batter_logs, batter_id, game_date)
    p = rolling_pitcher_stats(pitcher_logs, pitcher_id, game_date)
    run_f, hr_f = park_factors(venue_id)
    plat  = platoon_factor(batter_hand, pitcher_hand)
    temp_adj = temp_offense_factor(temp_f)

    # Expected hit rate given all factors
    # Base: batter's recent hit rate, adjusted for pitcher quality and park
    base_hit_rate = b["hit_rate"]
    pitcher_adj   = (1.30 - p["whip"]) * 0.05   # better WHIP = harder to get hits
    park_adj      = (run_f - 100) / 100 * 0.15   # park factor on hit rate (smaller effect than HRs)

    return {
        # Batter form (most predictive)
        "batter_hit_rate_30d":    b["hit_rate"],
        "batter_ab_30d":          b["ab_30d"],
        "batter_games_30d":       b["games_30d"],
        "batter_k_rate_30d":      b["k_rate"],

        # Pitcher quality
        "pitcher_whip_30d":       p["whip"],
        "pitcher_k9_30d":         p["k9"],
        "pitcher_ip_30d":         p["ip_30d"],
        "pitcher_days_rest":      p["days_rest"],

        # Matchup
        "platoon_factor":         plat,
        "is_home":                int(is_home),

        # Context
        "park_run_factor":        run_f,
        "temp_f":                 temp_f,
        "temp_adj":               temp_adj,

        # Composite signals (the "UFC edge" combinations)
        "hit_rate_plus_platoon":  base_hit_rate + plat,
        "pitcher_adj_hit_rate":   base_hit_rate + pitcher_adj + park_adj + plat + temp_adj,
        "favorable_matchup":      int(plat > 0 and b["hit_rate"] > 0.270 and p["whip"] > 1.25),
    }


def build_hr_features(
    batter_logs: pd.DataFrame,
    pitcher_logs: pd.DataFrame,
    batter_id: int,
    pitcher_id: int,
    batter_hand: str,
    pitcher_hand: str,
    game_date: date,
    venue_id: int,
    is_home: bool,
    temp_f: float = 70.0,
    wind_speed: float = 5.0,
    wind_out: bool = False,
) -> dict:
    """Feature vector for HR prop."""
    b = rolling_batter_stats(batter_logs, batter_id, game_date)
    p = rolling_pitcher_stats(pitcher_logs, pitcher_id, game_date)
    run_f, hr_f = park_factors(venue_id)
    plat     = platoon_factor(batter_hand, pitcher_hand)
    wind_b   = wind_hr_boost(wind_speed, wind_out)
    temp_adj = temp_offense_factor(temp_f)

    return {
        "batter_hr_rate_30d":    b["hr_rate"],
        "batter_hit_rate_30d":   b["hit_rate"],
        "batter_k_rate_30d":     b["k_rate"],
        "batter_ab_30d":         b["ab_30d"],

        "pitcher_hr9_30d":       p["hr9"],
        "pitcher_whip_30d":      p["whip"],
        "pitcher_days_rest":     p["days_rest"],

        "park_hr_factor":        hr_f,
        "platoon_factor":        plat,
        "wind_boost":            wind_b,
        "temp_adj":              temp_adj,
        "is_home":               int(is_home),

        # Composites
        "hr_park_wind":          b["hr_rate"] * (hr_f / 100) * (1 + wind_b),
        "favorable_hr":          int(hr_f >= 103 and wind_out and b["hr_rate"] > 0.04),
    }


def build_k_features(
    pitcher_logs: pd.DataFrame,
    pitcher_id: int,
    opp_team_k_rate: float,   # opposing lineup K% in last 30 days
    pitcher_hand: str,
    game_date: date,
    venue_id: int,
    is_home: bool,
    temp_f: float = 70.0,
) -> dict:
    """Feature vector for pitcher strikeout prop."""
    p = rolling_pitcher_stats(pitcher_logs, pitcher_id, game_date)
    run_f, _ = park_factors(venue_id)

    # Expected Ks in a start = K/9 × expected_innings
    # Good starters average ~5.5 innings these days
    expected_ip   = min(p["last_ip"] * 0.95, 6.5)   # slight fatigue from last start
    expected_k    = p["k9"] / 9 * expected_ip

    # Short rest penalty
    rest_penalty  = 0.5 if p["days_rest"] < 4 else 0.0

    return {
        "pitcher_k9_30d":        p["k9"],
        "pitcher_whip_30d":      p["whip"],
        "pitcher_ip_30d":        p["ip_30d"],
        "pitcher_days_rest":     p["days_rest"],
        "pitcher_last_ip":       p["last_ip"],

        "opp_team_k_rate_30d":   opp_team_k_rate,
        "park_run_factor":       run_f,
        "is_home":               int(is_home),
        "temp_f":                temp_f,

        # Composites — key signal
        "expected_k_raw":        expected_k,
        "expected_k_adj":        expected_k - rest_penalty,
        "k_rate_vs_opp":         p["k9"] * opp_team_k_rate / 0.22,   # normalized
        "elite_strikeout_arm":   int(p["k9"] >= 9.5 and opp_team_k_rate >= 0.23),
    }


# ── Target labels ──────────────────────────────────────────────────────────────

def label_hit(row: pd.Series) -> int:
    """1 if batter got at least 1 hit in the game."""
    return int(row.get("hits", 0) >= 1)

def label_hr(row: pd.Series) -> int:
    """1 if batter hit at least 1 HR."""
    return int(row.get("homeRuns", 0) >= 1)

def label_k(row: pd.Series, threshold: int) -> int:
    """1 if pitcher recorded at least `threshold` strikeouts."""
    return int(row.get("strikeOuts", 0) >= threshold)

"""Multi-sport prop evaluation engine with Monte Carlo simulation and devigged market odds."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gamma, nbinom, norm, poisson

from nfl_adjust import injury_layer, tier_splits
from nfl_matchups import matchup_multiplier
from nfl_six_point import canon_stat
from sports_data import clean_name, load_weekly

DEFAULTS = ("Patrick Mahomes", 275.5)
SOURCES = ["Model (EWMA + Matchup)", "My Mean (L10)", "My Median (L10)"]

# Supported Stat Categories (12 Markets)
STAT_CONFIG: Dict[str, Dict[str, Any]] = {
    "PASSING YARDS": {"col": "passing_yards", "dist": "gamma", "default_pos": "QB"},
    "PASS COMPLETIONS": {"col": "completions", "dist": "norm", "default_pos": "QB"},
    "PASS ATTEMPTS": {"col": "attempts", "dist": "norm", "default_pos": "QB"},
    "PASSING TDS": {"col": "passing_tds", "dist": "nbinom", "default_pos": "QB"},
    "INTERCEPTIONS": {"col": "interceptions", "dist": "nbinom", "default_pos": "QB"},
    "RUSHING YARDS": {"col": "rushing_yards", "dist": "gamma", "default_pos": "RB"},
    "RUSHING ATTEMPTS": {"col": "carries", "dist": "norm", "default_pos": "RB"},
    "RECEIVING YARDS": {"col": "receiving_yards", "dist": "gamma", "default_pos": "WR"},
    "RECEPTIONS": {"col": "receptions", "dist": "nbinom", "default_pos": "WR"},
    "KICKING POINTS": {"col": "kicking_points", "dist": "gamma", "default_pos": "K"},
    "FIELD GOALS": {"col": "field_goals", "dist": "nbinom", "default_pos": "K"},
    "TOUCHDOWNS": {"col": "touchdowns", "dist": "nbinom", "default_pos": "RB"},
}


def default_season() -> int:
    return 2026


def _to_prob(american_odds: float) -> float:
    try:
        o = float(american_odds)
        if o > 0:
            return 100.0 / (o + 100.0)
        return (-o) / (-o + 100.0)
    except Exception:
        return 0.5


def devig_two_way(over_odds: float, under_odds: float) -> Tuple[float, float]:
    p_over_raw = _to_prob(over_odds)
    p_under_raw = _to_prob(under_odds)
    total_implied = p_over_raw + p_under_raw
    if total_implied <= 0:
        return 0.5, 0.5
    return p_over_raw / total_implied, p_under_raw / total_implied


def probability_to_american(p: float) -> int:
    try:
        p = float(np.clip(p, 0.001, 0.999))
        if p >= 0.5:
            return int(round(-100.0 * (p / (1.0 - p))))
        return int(round(100.0 * ((1.0 - p) / p)))
    except Exception:
        return -110


def compute_ewma_baseline(vals: np.ndarray, alpha: float = 0.25, prior_weight: float = 4.0, prior_mean: Optional[float] = None) -> float:
    if len(vals) == 0:
        return float(prior_mean) if prior_mean is not None and not np.isnan(prior_mean) else 0.0
    
    weights = (1.0 - alpha) ** np.arange(len(vals))[::-1]
    ewma_val = float(np.sum(vals * weights) / np.sum(weights))
    
    if prior_mean is not None and not np.isnan(prior_mean) and prior_weight > 0:
        total_weight = float(len(vals) + prior_weight)
        return float((ewma_val * len(vals) + float(prior_mean) * prior_weight) / total_weight)
    return float(ewma_val)


def simulate_distribution(mu: float, dist_type: str, n_sims: int = 50000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = max(float(mu), 0.01)
    
    if dist_type == "gamma":
        shape = max(mu / 1.5, 0.5)
        scale = mu / shape
        return rng.gamma(shape=shape, scale=scale, size=n_sims)
    elif dist_type == "norm":
        sd = max(mu * 0.35, 1.0)
        draws = rng.normal(loc=mu, scale=sd, size=n_sims)
        return np.maximum(draws, 0.0)
    elif dist_type == "nbinom":
        r = max(mu * 1.2, 0.5)
        p = float(r / (r + mu))
        return rng.negative_binomial(n=r, p=p, size=n_sims).astype(float)
    
    return rng.poisson(lam=mu, size=n_sims).astype(float)


def resolve_stat_col(df: pd.DataFrame, stat_canon: str) -> str:
    candidates = []
    if stat_canon == "KICKING POINTS":
        candidates = ["kicking_points", "kick_pts", "pat_made"]
    elif stat_canon == "FIELD GOALS":
        candidates = ["field_goals", "fg_made", "field_goals_made"]
    elif stat_canon == "TOUCHDOWNS":
        candidates = ["touchdowns", "tds", "anytime_td"]
    elif stat_canon == "PASSING YARDS":
        candidates = ["passing_yards", "pass_yds"]
    elif stat_canon == "PASS COMPLETIONS":
        candidates = ["completions", "pass_comp"]
    elif stat_canon == "PASS ATTEMPTS":
        candidates = ["attempts", "pass_att"]
    elif stat_canon == "PASSING TDS":
        candidates = ["passing_tds", "pass_tds"]
    elif stat_canon == "INTERCEPTIONS":
        candidates = ["interceptions", "pass_ints"]
    elif stat_canon == "RUSHING YARDS":
        candidates = ["rushing_yards", "rush_yds"]
    elif stat_canon == "RUSHING ATTEMPTS":
        candidates = ["carries", "rushing_attempts", "rush_att"]
    elif stat_canon == "RECEIVING YARDS":
        candidates = ["receiving_yards", "rec_yds"]
    elif stat_canon == "RECEPTIONS":
        candidates = ["receptions", "rec"]

    upper_cols = {c.upper(): c for c in df.columns}
    for col in candidates:
        if col.upper() in upper_cols:
            return upper_cols[col.upper()]
    
    for col in df.columns:
        if stat_canon.split()[0] in col.upper():
            return col
    return df.columns[0]


def evaluate_prop(
    player_name: str,
    stat_category: str,
    line: float,
    over_odds: float,
    under_odds: float,
    projection_source: str = "Model (EWMA + Matchup)",
    season: int = 2026,
    opponent: Optional[str] = None,
    matchup_k: Optional[Dict[str, float]] = None,
    adj_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Strict numeric type coercion
    try:
        line = float(line)
        over_odds = float(over_odds)
        under_odds = float(under_odds)
        season = int(season)
    except (ValueError, TypeError) as exc:
        return {"error": f"Invalid numeric parameter passed to evaluator: {exc}"}

    c_stat = canon_stat(stat_category)
    config = STAT_CONFIG.get(c_stat)
    if config is None:
        return {"error": f"Unknown stat category '{stat_category}'."}
    
    adj_cfg = adj_cfg or {}
    try:
        weekly = load_weekly(season, back=1)
    except Exception as exc:
        return {"error": f"Failed to load player weekly stats: {exc}"}
    
    cn = clean_name(player_name)
    p_df = weekly[weekly["CNAME"] == cn].sort_values(["SEASON", "WEEK"])
    if p_df.empty:
        p_df = weekly[weekly["NAME"].str.lower().str.contains(str(player_name).lower())].sort_values(["SEASON", "WEEK"])
        if p_df.empty:
            return {"error": f"Player '{player_name}' not found in season data."}
    
    actual_name = str(p_df.iloc[-1]["NAME"])
    pid = str(p_df.iloc[-1]["PID"])
    team = str(p_df.iloc[-1]["TEAM"])
    pos = str(p_df.iloc[-1].get("POS", config["default_pos"]))
    
    stat_col = resolve_stat_col(weekly, c_stat)
    cur_df = p_df[p_df["SEASON"] == season]
    prior_df = p_df[p_df["SEASON"] == season - 1]
    
    cur_vals = pd.to_numeric(cur_df[stat_col], errors="coerce").fillna(0.0).to_numpy(dtype=float) if not cur_df.empty and stat_col in cur_df.columns else np.array([], dtype=float)
    prior_vals = pd.to_numeric(prior_df[stat_col], errors="coerce").fillna(0.0).to_numpy(dtype=float) if not prior_df.empty and stat_col in prior_df.columns else np.array([], dtype=float)
    
    prior_mean = float(np.mean(prior_vals)) if len(prior_vals) > 0 else None
    base_proj = float(compute_ewma_baseline(cur_vals, alpha=0.25, prior_weight=4.0, prior_mean=prior_mean))
    
    # Safe Opponent Matchup Adjustment
    m_info = None
    mult = 1.0
    if opponent and str(opponent).upper() != "NONE":
        try:
            m_info = matchup_multiplier(weekly, c_stat, pos, season, str(opponent), matchup_k)
            if isinstance(m_info, dict) and "mult" in m_info and m_info["mult"] is not None:
                mult = float(m_info["mult"])
        except Exception:
            m_info = None
            mult = 1.0
    
    # Usage / Injury Redistribution
    inj_info = None
    if adj_cfg.get("injuries", True):
        try:
            inj_info = injury_layer(actual_name, pid, team, c_stat, weekly, season, adj_cfg)
            if isinstance(inj_info, dict) and inj_info.get("mult") is not None:
                mult *= float(inj_info.get("mult", 1.0))
        except Exception:
            inj_info = None
    
    mult = float(mult)
    if "Mean" in str(projection_source) and len(cur_vals) >= 3:
        mu = float(np.mean(cur_vals[-10:])) * mult
    elif "Median" in str(projection_source) and len(cur_vals) >= 3:
        mu = float(np.median(cur_vals[-10:])) * mult
    else:
        mu = float(base_proj * mult)

    sims = simulate_distribution(mu, config["dist"], n_sims=50000)
    p_over_model = float(np.mean(sims > line)) * 100.0
    p_under_model = float(np.mean(sims < line)) * 100.0
    
    p_over_mkt, p_under_mkt = devig_two_way(over_odds, under_odds)
    p_over_mkt_pct, p_under_mkt_pct = float(p_over_mkt * 100.0), float(p_under_mkt * 100.0)
    
    over_prob_inv = (_to_prob(over_odds) ** -1) if over_odds > 0 else (1.0 + (100.0 / max(abs(over_odds), 1e-5)))
    under_prob_inv = (_to_prob(under_odds) ** -1) if under_odds > 0 else (1.0 + (100.0 / max(abs(under_odds), 1e-5)))
    
    ev_over = (p_over_model / 100.0) * over_prob_inv - 1.0
    ev_under = (p_under_model / 100.0) * under_prob_inv - 1.0
    
    if ev_over >= ev_under:
        best_side, p_model, p_market, ev_val = "Over", p_over_model, p_over_mkt_pct, ev_over
        fair_odds = probability_to_american(p_over_model / 100.0)
        b = (_to_prob(over_odds) ** -1 - 1.0) if over_odds > 0 else (100.0 / max(abs(over_odds), 1e-5))
    else:
        best_side, p_model, p_market, ev_val = "Under", p_under_model, p_under_mkt_pct, ev_under
        fair_odds = probability_to_american(p_under_model / 100.0)
        b = (_to_prob(under_odds) ** -1 - 1.0) if under_odds > 0 else (100.0 / max(abs(under_odds), 1e-5))
    
    edge_pp = float(p_model - p_market)
    ev_pct = float(ev_val * 100.0)
    
    p_win = float(p_model / 100.0)
    q_loss = float(1.0 - p_win)
    full_kelly = max((b * p_win - q_loss) / max(b, 1e-5), 0.0)
    kelly_pct = float(min(full_kelly * 0.25 * 100.0, 5.0) if ev_pct > 0 else 0.0)

    recent_all = p_df.tail(10)
    gamelog = [
        {
            "x_label": f"W{int(r['WEEK'])}",
            "stat_value": float(r[stat_col]),
            "season": "current" if r["SEASON"] == season else "prior",
        }
        for _, r in recent_all.iterrows()
        if stat_col in r and pd.notna(r[stat_col])
    ]

    last5 = cur_vals[-5:] if len(cur_vals) >= 5 else cur_vals
    last10 = cur_vals[-10:] if len(cur_vals) >= 10 else cur_vals

    try:
        t_splits = tier_splits(weekly, pid, pos, c_stat)
    except Exception:
        t_splits = None

    return {
        "player_name": actual_name,
        "team": team,
        "pos": pos,
        "stat": c_stat,
        "n_current": len(cur_vals),
        "n_prior": len(prior_vals),
        "base_proj": float(base_proj),
        "mu": float(mu),
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "best_side": best_side,
        "p_model": float(p_model),
        "p_market": float(p_market),
        "fair_odds": fair_odds,
        "edge_pp": float(edge_pp),
        "ev_pct": float(ev_pct),
        "kelly_pct": float(kelly_pct),
        "raw_avg": float(np.mean(last10)) if len(last10) else 0.0,
        "hit5": int(np.sum(last5 > line)),
        "n5": len(last5),
        "hit10": int(np.sum(last10 > line)),
        "n10": len(last10),
        "gamelog_details": gamelog,
        "matchup": m_info,
        "inj": inj_info,
        "tiers": t_splits,
        "warnings": [],
    }

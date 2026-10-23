"""Multi-Sport Monte Carlo Prop Evaluation Engine with robust stat key mapping."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sports_data import fetch_nfl_player_logs, fetch_nba_player_logs

SPORTS = ["NFL", "NBA"]

DEFAULTS = {
    "NFL": ("Patrick Mahomes", 265.5),
    "NBA": ("Jayson Tatum", 26.5),
}

# Normalize and map user/odds UI dropdown categories directly to DataFrame column keys
STAT_MAP = {
    "NFL": {
        "PASSING YARDS": ["PASSING YARDS", "PASS_YARDS", "YDS", "PASS YDS"],
        "PASS COMPLETIONS": ["PASS COMPLETIONS", "COMPLETIONS", "COMP", "C"],
        "PASS ATTEMPTS": ["PASS ATTEMPTS", "ATTEMPTS", "ATT"],
        "PASSING TDS": ["PASSING TDS", "PASS_TDS", "TD", "PASS TD"],
        "RUSHING YARDS": ["RUSHING YARDS", "RUSH_YARDS", "RUSH YDS"],
        "RUSHING ATTEMPTS": ["RUSHING ATTEMPTS", "CARRIES", "CAR", "RUSH ATT"],
        "RECEIVING YARDS": ["RECEIVING YARDS", "REC_YARDS", "REC YDS"],
        "RECEPTIONS": ["RECEPTIONS", "REC", "CATCHES"],
        "INTERCEPTIONS": ["INTERCEPTIONS", "INT", "INTS"],
    },
    "NBA": {
        "POINTS": ["POINTS", "PTS"],
        "REBOUNDS": ["REBOUNDS", "REB", "TREB"],
        "ASSISTS": ["ASSISTS", "AST"],
        "3-POINTERS MADE": ["3-POINTERS MADE", "3PM", "3PT", "FG3M"],
        "PTS + REB + AST": ["PTS + REB + AST", "PRA"],
        "STEALS": ["STEALS", "STL"],
        "BLOCKS": ["BLOCKS", "BLK"],
    }
}

def stat_options(sport: str) -> List[str]:
    return list(STAT_MAP.get(sport, {}).keys())

def default_season(sport: str) -> int:
    return 2026

def american_to_implied(odds: float) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)

def implied_to_american(prob: float) -> int:
    prob = max(0.001, min(0.999, prob))
    if prob >= 0.5:
        return int(round(- (prob / (1.0 - prob)) * 100.0))
    else:
        return int(round(((1.0 - prob) / prob) * 100.0))

def extract_stat_series(df: pd.DataFrame, sport: str, stat_category: str) -> pd.Series:
    """Extract numerical values matching the stat category across mapped column names."""
    if df.empty:
        return pd.Series(dtype=float)
    
    clean_cat = str(stat_category).upper().strip()
    target_keys = STAT_MAP.get(sport, {}).get(clean_cat, [clean_cat])
    
    # Check exact match or mapped aliases first
    for col in df.columns:
        col_upper = str(col).upper().strip()
        if col_upper in target_keys or col_upper == clean_cat:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if not s.empty and s.sum() > 0:
                return s
                
    # Fallback partial string match
    for col in df.columns:
        col_upper = str(col).upper().strip()
        for tk in target_keys:
            if tk in col_upper:
                s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                if not s.empty:
                    return s

    return pd.Series(dtype=float)

def evaluate_prop_multisport(
    league: str,
    player_name: str,
    stat_category: str,
    line: float,
    over_odds: float,
    under_odds: float,
    season: int = 2026,
    num_sims: int = 10000,
) -> Dict[str, Any]:
    """Execute Monte Carlo evaluation with fallback stat key mapping."""
    if league == "NFL":
        df_logs, err = fetch_nfl_player_logs(player_name, season)
    else:
        df_logs, err = fetch_nba_player_logs(player_name, season)

    if err or df_logs.empty:
        return {"error": err or f"No game log data available for {player_name}."}

    series = extract_stat_series(df_logs, league, stat_category)
    if series.empty or len(series) == 0:
        return {"error": f"No data found for stat category '{stat_category}'."}

    vals = series.values
    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))
    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
    if std_val == 0:
        std_val = 0.5

    # Monte Carlo simulation drawing from fitted log-normal / normal distribution
    sim_data = np.random.normal(mean_val, std_val, num_sims)
    sim_data = np.maximum(0, sim_data)

    over_hits = float(np.sum(sim_data > line))
    under_hits = float(np.sum(sim_data < line))
    
    over_prob = over_hits / num_sims
    under_prob = under_hits / num_sims

    implied_over = american_to_implied(over_odds)
    implied_under = american_to_implied(under_odds)

    over_edge = (over_prob - implied_over) * 100.0
    under_edge = (under_prob - implied_under) * 100.0

    if over_edge >= under_edge:
        best_side = "OVER"
        best_prob = over_prob
        fair_odds = implied_to_american(best_prob)
        edge = over_edge
        mkt_odds = over_odds
    else:
        best_side = "UNDER"
        best_prob = under_prob
        fair_odds = implied_to_american(best_prob)
        edge = under_edge
        mkt_odds = under_odds

    if mkt_odds > 0:
        b_profit = mkt_odds / 100.0
    else:
        b_profit = 100.0 / abs(mkt_odds)

    ev_pct = (best_prob * b_profit - (1.0 - best_prob)) * 100.0

    gamelog_details = []
    for idx, row in df_logs.iterrows():
        val = series.iloc[idx] if idx < len(series) else 0.0
        gamelog_details.append({
            "x_label": str(row.get("DATE", f"G{idx+1}")),
            "stat_value": float(val),
            "opponent": str(row.get("OPPONENT", "OPP")),
        })

    return {
        "player_name": player_name,
        "stat_category": stat_category,
        "line": line,
        "projection_mean": mean_val,
        "projection_median": median_val,
        "projection_std": std_val,
        "best_side": best_side,
        "fair_odds": fair_odds,
        "edge_vs_market": edge,
        "ev_pct": ev_pct,
        "gamelog_details": gamelog_details,
    }

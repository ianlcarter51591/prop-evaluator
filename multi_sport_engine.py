"""NFL Monte Carlo Prop Evaluator Engine."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from sports_data import fetch_nfl_player_logs

SOURCES: Tuple[str, ...] = ("Model (EWMA mean)", "My mean", "My median")
DEVIG_METHODS: Tuple[str, ...] = ("Multiplicative", "Additive")

DEFAULTS: Tuple[str, float] = ("Aaron Rodgers", 244.5)

STATS: List[str] = [
    "Passing Yards",
    "Pass Completions",
    "Pass Attempts",
    "Passing TDs",
    "Rushing Yards",
    "Rushing Attempts",
    "Receiving Yards",
    "Receptions",
    "Interceptions",
]

# Mapping friendly UI stat labels to actual backend dataframe column keys
STAT_COLUMN_MAP: Dict[str, List[str]] = {
    "PASSING YARDS": ["YDS", "PASSING YARDS", "PASS_YDS"],
    "PASS COMPLETIONS": ["CMP", "PASS COMPLETIONS", "PASS_CMP"],
    "PASS ATTEMPTS": ["ATT", "PASS ATTEMPTS", "PASS_ATT"],
    "PASSING TDS": ["TD", "PASSING TDS", "PASS_TD"],
    "RUSHING YARDS": ["RUSH_YDS", "RUSHING YARDS", "YDS_RUSH"],
    "RUSHING ATTEMPTS": ["RUSH_ATT", "RUSHING ATTEMPTS", "ATT_RUSH"],
    "RECEIVING YARDS": ["REC_YDS", "RECEIVING YARDS", "YDS_REC"],
    "RECEPTIONS": ["REC", "RECEPTIONS"],
    "INTERCEPTIONS": ["INT", "INTERCEPTIONS"],
}

def stat_options() -> List[str]:
    return STATS

def default_season() -> int:
    return 2026

def american_to_implied_prob(american_odds: float) -> float:
    try:
        odds = float(american_odds)
        if odds < 0:
            return round((-odds / (-odds + 100)) * 100, 1)
        return round((100 / (odds + 100)) * 100, 1)
    except Exception:
        return 52.4

def calculate_ewma(stats: np.ndarray, decay: float = 0.8) -> float:
    if len(stats) == 0:
        return 0.0
    weights = decay ** np.arange(len(stats))[::-1]
    weights /= weights.sum()
    return float(np.sum(stats * weights))

def evaluate_prop(
    player_name: str,
    stat_category: str,
    line: float,
    over_odds: float,
    under_odds: float,
    projection_source: str = "Model (EWMA mean)",
    season: int = 2026,
) -> Dict[str, Any]:
    
    try:
        df, err = fetch_nfl_player_logs(player_name, season)

        if err or df is None or df.empty:
            return {"error": err or f"No game logs found for '{player_name}'."}

        # Clean and uppercase column headers
        df.columns = [c.upper().strip() for c in df.columns]

        stat_upper = stat_category.upper().strip()
        
        target_col = None
        if stat_upper in df.columns:
            target_col = stat_upper
        else:
            possible_cols = STAT_COLUMN_MAP.get(stat_upper, [stat_upper])
            for col in possible_cols:
                if col in df.columns:
                    target_col = col
                    break

        if not target_col or target_col not in df.columns:
            return {"error": f"Stat category '{stat_category}' not found in logs for '{player_name}' (Available columns: {list(df.columns)})"}

        stats_list = []
        gamelog_details = []

        for _, row in df.iterrows():
            try:
                val = float(row[target_col])
            except (ValueError, TypeError):
                continue
            
            opp = str(row.get("OPPONENT", row.get("OPP", "OPP")))
            gdate = str(row.get("DATE", row.get("GAME_DATE", "")))
            stats_list.append(val)
            gamelog_details.append({
                "stat_value": val,
                "opponent": opp,
                "date": gdate,
                "x_label": f"{gdate}<br>{opp}" if gdate else opp,
            })

        if not stats_list:
            return {"error": f"No valid numeric data found for stat '{stat_category}' in player logs."}

        # Convert to numpy array and take up to the last 10 games for recent form
        stats_arr = np.array(stats_list[:10])
        gamelog_details = gamelog_details[:10]

        real_mean = float(np.mean(stats_arr))
        real_median = float(np.median(stats_arr))

        if projection_source == "My mean":
            proj_mean = real_mean
        elif projection_source == "My median":
            proj_mean = real_median
        else:
            proj_mean = calculate_ewma(stats_arr)

        std_dev = float(np.std(stats_arr)) if len(stats_arr) > 1 else max(1.0, proj_mean * 0.15)
        simulations = np.random.normal(proj_mean, std_dev if std_dev > 0 else 1.0, 10000)

        over_hits = np.sum(simulations > line)
        over_model_pct = (over_hits / 10000) * 100
        under_model_pct = 100 - over_model_pct

        over_market_pct = american_to_implied_prob(over_odds)
        under_market_pct = american_to_implied_prob(under_odds)

        over_win_pct = (over_model_pct * 0.5) + (over_market_pct * 0.5)
        under_win_pct = (under_model_pct * 0.5) + (under_market_pct * 0.5)

        over_fair_odds = int(-100 * (over_win_pct / (100 - over_win_pct))) if over_win_pct > 50 else int(100 * ((100 - over_win_pct) / over_win_pct))
        under_fair_odds = int(-100 * (under_win_pct / (100 - under_win_pct))) if under_win_pct > 50 else int(100 * ((100 - under_win_pct) / under_win_pct))

        is_over = over_win_pct > under_win_pct
        best_side = f"Over {line}" if is_over else f"Under {line}"
        edge_vs_market = (over_win_pct - over_market_pct) if is_over else (under_win_pct - under_market_pct)
        
        chosen_odds = over_odds if is_over else under_odds
        chosen_win_pct = over_win_pct if is_over else under_win_pct
        
        if chosen_odds < 0:
            ev_pct = (chosen_win_pct * (100 / abs(chosen_odds))) - (100 - chosen_win_pct)
        else:
            ev_pct = (chosen_win_pct * (chosen_odds / 100)) - (100 - chosen_win_pct)

        return {
            "player_name": player_name,
            "projection_mean": real_mean,
            "projection_median": real_median,
            "best_side": best_side,
            "fair_odds": over_fair_odds if is_over else under_fair_odds,
            "edge_vs_market": edge_vs_market,
            "ev_pct": ev_pct,
            "gamelog_details": gamelog_details,
        }

    except Exception as e:
        return {"error": f"Evaluation error: {e}"}

"""NFL & NBA Monte Carlo Prop Evaluator Engine."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from sports_data import fetch_nba_player_logs, fetch_nfl_player_logs

SPORTS: Tuple[str, ...] = ("NFL", "NBA")
SOURCES: Tuple[str, ...] = ("Model (EWMA mean)", "My mean", "My median")
DEVIG_METHODS: Tuple[str, ...] = ("Multiplicative", "Additive")

DEFAULTS: Dict[str, Tuple[str, float]] = {
    "NFL": ("Aaron Rodgers", 244.5),
    "NBA": ("Jayson Tatum", 26.5),
}

STATS: Dict[str, List[str]] = {
    "NFL": [
        "Passing Yards",
        "Pass Completions",
        "Pass Attempts",
        "Passing TDs",
        "Rushing Yards",
        "Rushing Attempts",
        "Receiving Yards",
        "Receptions",
        "Interceptions",
    ],
    "NBA": [
        "Points",
        "Rebounds",
        "Assists",
        "3-Pointers Made",
        "Pts + Reb + Ast",
        "Steals",
        "Blocks",
    ],
}


def stat_options(league: str) -> List[str]:
    return STATS.get(league.upper(), [])


def default_season(league: str) -> int:
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


def evaluate_prop_multisport(
    league: str,
    player_name: str,
    stat_category: str,
    line: float,
    over_odds: float,
    under_odds: float,
    projection_source: str = "Model (EWMA mean)",
    season: int = 2026,
) -> Dict[str, Any]:
    league = league.upper()
    try:
        if league == "NFL":
            df, err = fetch_nfl_player_logs(player_name, season)
        else:
            df, err = fetch_nba_player_logs(player_name, season)

        if err or df is None or df.empty:
            return {"error": err or f"No game logs found for '{player_name}'."}

        target_col = stat_category.upper().strip()
        df.columns = [c.upper().strip() for c in df.columns]

        if target_col not in df.columns:
            return {"error": f"Stat category '{stat_category}' not found in logs for '{player_name}'."}

        stats_list = []
        gamelog_details = []

        for _, row in df.iterrows():
            val = float(row[target_col])
            opp = str(row.get("OPPONENT", "OPP"))
            gdate = str(row.get("DATE", ""))
            stats_list.append(val)
            gamelog_details.append({
                "stat_value": val,
                "opponent": opp,
                "date": gdate,
                "x_label": f"{gdate}<br>{opp}",
            })

        stats_arr = np.array(stats_list[:10])
        gamelog_details = gamelog_details[:10]

        real_mean = float(np.mean(stats_arr))
        real_median = float(np.median(stats_arr))

        proj_mean = calculate_ewma(stats_arr) if projection_source == "Model (EWMA mean)" else real_median

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

        best_side = f"Over {line}" if over_win_pct > under_win_pct else f"Under {line}"
        edge_vs_market = over_win_pct - over_market_pct if over_win_pct > under_win_pct else under_win_pct - under_market_pct
        ev_pct = (over_win_pct * (100 / abs(over_odds))) - (100 - over_win_pct) if over_win_pct > under_win_pct else (under_win_pct * (100 / abs(under_odds))) - (100 - under_win_pct)

        return {
            "player_name": player_name,
            "projection_mean": real_mean,
            "projection_median": real_median,
            "best_side": best_side,
            "fair_odds": over_fair_odds if over_win_pct > under_win_pct else under_fair_odds,
            "edge_vs_market": edge_vs_market,
            "ev_pct": ev_pct,
            "gamelog_details": gamelog_details,
        }

    except Exception as e:
        return {"error": f"Evaluation error: {e}"}

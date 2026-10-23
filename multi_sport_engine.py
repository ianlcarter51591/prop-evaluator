"""NFL Stat Evaluator and Monte Carlo Simulation Engine."""
import numpy as np
import pandas as pd
from sports_data import fetch_nfl_player_logs


def american_to_implied_prob(american_odds):
    """Convert American odds to implied probability percentage."""
    try:
        odds = float(american_odds)
        if odds < 0:
            return round((-odds / (-odds + 100)) * 100, 1)
        else:
            return round((100 / (odds + 100)) * 100, 1)
    except Exception:
        return 52.4


def calculate_ewma(stats, decay=0.8):
    """Calculate Exponentially Weighted Moving Average (EWMA)."""
    if len(stats) == 0:
        return 0.0
    weights = decay ** np.arange(len(stats))[::-1]
    weights /= weights.sum()
    return float(np.sum(stats * weights))


def evaluate_prop(
    player_name,
    stat_category,
    line,
    over_odds,
    under_odds,
    projection_source="Model (EWMA mean)",
    custom_val=22.5,
    ewma_decay=0.8,
    max_recent_games=10,
    matchup_multiplier=1.0,
):
    """Evaluate NFL prop with exact alignment between mean, median, and game logs."""
    try:
        df, err = fetch_nfl_player_logs(player_name)
        if err or df is None or df.empty:
            return {"error": err or f"No game logs found for '{player_name}'."}

        target_col = stat_category.upper().strip()
        df.columns = [c.upper().strip() for c in df.columns]

        if target_col not in df.columns:
            return {"error": f"Stat category '{stat_category}' not found in logs for '{player_name}'."}

        stats = []
        gamelog_details = []

        for _, row in df.iterrows():
            val = float(row[target_col])
            opp = str(row.get("OPPONENT", "vs OPP"))
            gdate = str(row.get("DATE", ""))
            stats.append(val)
            gamelog_details.append({
                "stat_value": val,
                "opponent": opp,
                "date": gdate,
                "x_label": f"{gdate}<br>{opp}"
            })

        stats_arr = np.array(stats[:max_recent_games])
        gamelog_details = gamelog_details[:max_recent_games]

        real_mean = float(np.mean(stats_arr))
        real_median = float(np.median(stats_arr))

        if projection_source == "Model (EWMA mean)":
            base_proj = calculate_ewma(stats_arr, decay=ewma_decay)
        elif projection_source == "My mean":
            base_proj = float(custom_val)
        else:
            base_proj = real_median

        proj_mean = base_proj * matchup_multiplier
        std_dev = float(np.std(stats_arr)) if len(stats_arr) > 1 else max(1.0, proj_mean * 0.15)
        if std_dev == 0:
            std_dev = 1.0

        simulations = np.random.normal(proj_mean, std_dev, 10000)
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

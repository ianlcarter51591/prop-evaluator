"""Multi-sport evaluation engine for NFL, NBA, and MLB props."""
import numpy as np
import pandas as pd
from sports_data import fetch_nba_player_logs, fetch_mlb_player_logs


def american_to_implied_prob(american_odds):
    """Convert American odds (-110, +150) to implied probability percentage."""
    try:
        odds = float(american_odds)
        if odds < 0:
            return round((-odds / (-odds + 100)) * 100, 1)
        else:
            return round((100 / (odds + 100)) * 100, 1)
    except Exception:
        return 52.4


def get_nba_logs(player_name, season="2025-26"):
    """Fetch NBA game logs via ESPN fetcher in sports_data."""
    df, err = fetch_nba_player_logs(player_name)
    if err:
        return None, err
    return df, None


def get_mlb_logs(player_name, season=2026):
    """Fetch MLB game logs via MLB StatsAPI fetcher in sports_data."""
    df, err = fetch_mlb_player_logs(player_name, target_season=season)
    if err:
        return None, err
    return df, None


def calculate_ewma(stats, decay=0.8):
    """Calculate Exponentially Weighted Moving Average (EWMA) on game stats."""
    if len(stats) == 0:
        return 0.0
    weights = decay ** np.arange(len(stats))[::-1]
    weights /= weights.sum()
    return float(np.sum(stats * weights))


def evaluate_prop_multisport(
    league,
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
    """Evaluate a sports prop using Monte Carlo simulation based on recent stats."""
    try:
        # 1. Fetch Game Logs
        if league == "NBA":
            df, err = get_nba_logs(player_name)
            if err:
                return {"error": err}
            stat_map = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PM": "3PM"}
            if stat_category not in stat_map:
                return {"error": f"Stat '{stat_category}' not supported for NBA."}
            col_name = stat_map[stat_category]
            if col_name not in df.columns:
                return {"error": f"Column '{col_name}' missing from NBA game logs."}
            stats = df[col_name].astype(float).values

        elif league == "MLB":
            df, err = get_mlb_logs(player_name)
            if err:
                return {"error": err}
            stat_map = {"Hits": "hits", "Total Bases": "baseOnBalls", "Strikeouts": "strikeOuts"}
            col_name = stat_map.get(stat_category, "hits")
            if col_name not in df.columns:
                return {"error": f"Column '{col_name}' missing from MLB game logs."}
            stats = df[col_name].astype(float).values

        else:  # NFL / Default Mock Fallback for testing
            # Sample distribution array if no active NFL feed connected
            stats = np.array([22.0, 24.0, 19.0, 23.0, 21.0, 25.0, 22.0, 20.0, 26.0, 21.0])

        if len(stats) == 0:
            return {"error": f"No game log records found for {player_name}."}

        # Slice to recent games limit
        stats = stats[:max_recent_games]

        # 2. Determine Mean Projection
        if projection_source == "Model (EWMA mean)":
            base_mean = calculate_ewma(stats, decay=ewma_decay)
        elif projection_source == "My mean":
            base_mean = float(custom_val)
        else:  # My median
            base_mean = float(np.median(stats))

        proj_mean = base_mean * matchup_multiplier
        proj_median = float(np.median(stats))
        std_dev = float(np.std(stats)) if len(stats) > 1 else 3.5
        if std_dev == 0:
            std_dev = 1.0

        # 3. Run Monte Carlo Simulation (10,000 trials)
        simulations = np.random.normal(proj_mean, std_dev, 10000)
        over_hits = np.sum(simulations > line)
        over_model_pct = (over_hits / 10000) * 100
        under_model_pct = 100 - over_model_pct

        # 4. Market Odds & Implied Probabilities
        over_market_pct = american_to_implied_prob(over_odds)
        under_market_pct = american_to_implied_prob(under_odds)

        # Blended Projections
        over_win_pct = (over_model_pct * 0.5) + (over_market_pct * 0.5)
        under_win_pct = (under_model_pct * 0.5) + (under_market_pct * 0.5)

        over_fair_odds = int(-100 * (over_win_pct / (100 - over_win_pct))) if over_win_pct > 50 else int(100 * ((100 - over_win_pct) / over_win_pct))
        under_fair_odds = int(-100 * (under_win_pct / (100 - under_win_pct))) if under_win_pct > 50 else int(100 * ((100 - under_win_pct) / under_win_pct))

        best_side = "Over " + str(line) if over_win_pct > under_win_pct else "Under " + str(line)
        edge_vs_market = over_win_pct - over_market_pct if over_win_pct > under_win_pct else under_win_pct - under_market_pct
        ev_pct = (over_win_pct * (100 / abs(over_odds))) - (100 - over_win_pct) if over_win_pct > under_win_pct else (under_win_pct * (100 / abs(under_odds))) - (100 - under_win_pct)

        return {
            "player_name": player_name,
            "projection_mean": proj_mean,
            "projection_median": proj_median,
            "best_side": best_side,
            "fair_odds": over_fair_odds if over_win_pct > under_win_pct else under_fair_odds,
            "edge_vs_market": edge_vs_market,
            "ev_pct": ev_pct,
            "over_win_pct": over_win_pct,
            "under_win_pct": under_win_pct,
            "over_model_pct": over_model_pct,
            "under_model_pct": under_model_pct,
            "over_market_pct": over_market_pct,
            "under_market_pct": under_market_pct,
            "over_fair_odds": over_fair_odds,
            "under_fair_odds": under_fair_odds,
            "over_edge": over_win_pct - over_market_pct,
            "under_edge": under_win_pct - under_market_pct,
            "over_ev": (over_win_pct * (100 / abs(over_odds))) - (100 - over_win_pct),
            "under_ev": (under_win_pct * (100 / abs(under_odds))) - (100 - under_win_pct),
            "simulations": simulations,
            "recent_games": stats.tolist(),
            "kelly_stake_dollars": 0.53,
            "kelly_pct": 0.05,
        }

    except Exception as e:
        return {"error": f"Evaluation error: {e}"}

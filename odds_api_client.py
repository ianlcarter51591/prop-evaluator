"""Odds API client for fetching live DraftKings player prop markets."""
from __future__ import annotations

import os
import requests
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

API_BASE = "https://api.the-odds-api.com/v4/sports"

MARKETS: Dict[str, Dict[str, str]] = {
    "NFL": {
        "Passing Yards": "player_pass_yds",
        "Pass Completions": "player_pass_completions",
        "Pass Attempts": "player_pass_attempts",
        "Passing TDs": "player_pass_tds",
        "Interceptions": "player_pass_interceptions",
        "Rushing Yards": "player_rush_yds",
        "Rushing Attempts": "player_rush_attempts",
        "Receiving Yards": "player_reception_yds",
        "Receptions": "player_receptions",
        "Kicking Points": "player_kicking_points",
        "Field Goals Made": "player_field_goals",
        "Extra Points Made": "player_extra_points",
        "Anytime Touchdown Scorer": "player_anytime_td",
    }
}


class OddsApiError(Exception):
    """Custom exception for Odds API request issues."""
    pass


def list_events(sport_key: str, api_key: str) -> Tuple[pd.DataFrame, int]:
    """Fetch upcoming games/events for a given sport key."""
    url = f"{API_BASE}/{sport_key.lower()}/events"
    params = {"apiKey": api_key}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise OddsApiError(f"API Error ({resp.status_code}): {resp.text}")
    
    data = resp.json()
    remaining_requests = int(resp.headers.get("x-requests-remaining", 0))
    
    rows = []
    for ev in data:
        rows.append({
            "id": ev["id"],
            "commence_time": ev["commence_time"],
            "home_team": ev["home_team"],
            "away_team": ev["away_team"],
            "label": f"{ev['away_team']} @ {ev['home_team']} ({ev['commence_time']})"
        })
    return pd.DataFrame(rows), remaining_requests


def fetch_prop_board(
    sport: str,
    event_id: str,
    market_label: str,
    bookmakers: Tuple[str, ...],
    api_key: str
) -> Tuple[pd.DataFrame, int]:
    """Fetch specific player prop board for a given event and market."""
    sport_map = MARKETS.get(sport.upper(), {})
    market_key = sport_map.get(market_label)
    if not market_key:
        raise OddsApiError(f"Market label '{market_label}' not recognized for {sport}.")

    url = f"{API_BASE}/{sport.lower()}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": market_key,
        "bookmakers": ",".join(bookmakers),
        "oddsFormat": "american",
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise OddsApiError(f"API Error ({resp.status_code}): {resp.text}")

    remaining_requests = int(resp.headers.get("x-requests-remaining", 0))
    data = resp.json()
    
    rows = []
    for bmark in data.get("bookmakers", []):
        book_title = bmark.get("title", bmark.get("key"))
        for mkt in bmark.get("markets", []):
            for outcome in mkt.get("outcomes", []):
                rows.append({
                    "Bookmaker": book_title,
                    "Player": outcome.get("description", outcome.get("name")),
                    "Side": outcome.get("name"),
                    "Line": outcome.get("point", np.nan),
                    "Odds": outcome.get("price"),
                })
    return pd.DataFrame(rows), remaining_requests


def summarize_board(board: pd.DataFrame) -> pd.DataFrame:
    """Pivot raw outcomes table into a single row per player with Line, Over, and Under columns."""
    if board.empty:
        return pd.DataFrame(columns=["Player", "Line", "Over", "Under"])
    
    pivoted = []
    for player, grp in board.groupby("Player"):
        line = grp["Line"].dropna().iloc[0] if not grp["Line"].dropna().empty else np.nan
        over_odds = grp[grp["Side"] == "Over"]["Odds"].values
        under_odds = grp[grp["Side"] == "Under"]["Odds"].values
        
        over_val = over_odds[0] if len(over_odds) > 0 else np.nan
        under_val = under_odds[0] if len(under_odds) > 0 else np.nan
        
        pivoted.append({
            "Player": player,
            "Line": float(line) if pd.notna(line) else np.nan,
            "Over": float(over_val) if pd.notna(over_val) else np.nan,
            "Under": float(under_val) if pd.notna(under_val) else np.nan,
        })
    return pd.DataFrame(pivoted)


def player_offers(board: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Filter board summary down to a single selected player."""
    summary = summarize_board(board)
    return summary[summary["Player"] == player_name]

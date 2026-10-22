from __future__ import annotations
import os
import requests
import streamlit as st
from typing import Dict, Any, List, Tuple, Optional

BASE_URL = "https://api.the-odds-api.com/v4/sports"

SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb"
}

MARKET_KEYS = {
    "NFL": {
        "Passing Yards": "player_pass_yds",
        "Pass Completions": "player_pass_completions",
        "Passing TDs": "player_pass_tds",
        "Receptions": "player_receptions",
        "Receiving Yards": "player_receive_yds",
        "Rushing Yards": "player_rush_yds",
    },
    "NBA": {
        "PTS": "player_points",
        "Points": "player_points",
        "REB": "player_rebounds",
        "Rebounds": "player_rebounds",
        "AST": "player_assists",
        "Assists": "player_assists",
        "3PM": "player_threes",
        "3-Pointers Made": "player_threes",
    },
    "MLB": {
        "Strikeouts": "pitcher_strikeouts",
        "Hits": "batter_hits",
        "Total Bases": "batter_total_bases",
    }
}

@st.cache_data(ttl=300)
def fetch_player_prop_line(
    sport: str,
    player_name: str,
    stat_category: str,
    bookmaker_preference: str = "draftkings"
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[str]]:
    """Look up live prop line and odds for a specific player from The Odds API."""
    key = st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", ""))
    if not key:
        return None, None, None, "No ODDS_API_KEY found in Streamlit Secrets."

    sport_key = SPORT_KEYS.get(sport)
    market_key = MARKET_KEYS.get(sport, {}).get(stat_category)

    if not sport_key or not market_key:
        return None, None, None, f"Market '{stat_category}' not mapped for {sport}."

    # 1. Fetch active events for the sport
    events_url = f"{BASE_URL}/{sport_key}/events"
    try:
        e_res = requests.get(events_url, params={"apiKey": key}, timeout=10)
        if e_res.status_code != 200:
            return None, None, None, f"Events API Error: {e_res.status_code}"
        events = e_res.json()
    except Exception as e:
        return None, None, None, f"Events fetch failed: {e}"

    if not events:
        return None, None, None, f"No active upcoming {sport} games found."

    # 2. Search upcoming event odds for the player prop market
    player_clean = player_name.strip().lower()

    for event in events[:5]:  # Check first 5 upcoming games
        event_id = event.get("id")
        odds_url = f"{BASE_URL}/{sport_key}/events/{event_id}/odds"
        params = {
            "apiKey": key,
            "regions": "us",
            "markets": market_key,
            "bookmakers": f"{bookmaker_preference},fanduel",
            "oddsFormat": "american"
        }
        try:
            o_res = requests.get(odds_url, params=params, timeout=10)
            if o_res.status_code != 200:
                continue
            data = o_res.json()
            
            for bk in data.get("bookmakers", []):
                for mkt in bk.get("markets", []):
                    if mkt.get("key") == market_key:
                        over_odds, under_odds, line = None, None, None
                        for outcome in mkt.get("outcomes", []):
                            desc = str(outcome.get("description", "")).lower()
                            if player_clean in desc or desc in player_clean:
                                line = outcome.get("point")
                                if outcome.get("name") == "Over":
                                    over_odds = outcome.get("price")
                                elif outcome.get("name") == "Under":
                                    under_odds = outcome.get("price")

                        if line is not None and over_odds is not None and under_odds is not None:
                            return float(line), int(over_odds), int(under_odds), None
        except Exception:
            continue

    return None, None, None, f"No live market line found for '{player_name}' on DraftKings."

from __future__ import annotations
import os
import requests
import streamlit as st
from typing import Dict, Any, List, Tuple

BASE_URL = "https://api.the-odds-api.com/v4/sports"

SPORT_KEYS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb"
}

MARKET_KEYS = {
    "NFL": {
        "Passing Yards": "player_pass_yds",
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
        "Pitcher Strikeouts": "pitcher_strikeouts",
        "Hits": "batter_hits",
        "Total Bases": "batter_total_bases",
    }
}


@st.cache_data(ttl=900)  # Cache for 15 minutes to save API quota
def fetch_live_slate(
    api_key: str | None = None,
    sport: str = "NBA",
    stat_label: str = "PTS"
) -> Tuple[List[Dict[str, Any]], str | None]:
    """Fetch live player prop odds from The Odds API across major sportsbooks."""
    # Retrieve key from parameter or Streamlit Secrets / Environment
    key = api_key or st.secrets.get("ODDS_API_KEY", os.getenv("ODDS_API_KEY", ""))
    if not key:
        return [], "No Odds API key found in secrets or parameter."

    sport_key = SPORT_KEYS.get(sport)
    market_key = MARKET_KEYS.get(sport, {}).get(stat_label)

    if not sport_key or not market_key:
        return [], f"Unsupported sport '{sport}' or market '{stat_label}'."

    url = f"{BASE_URL}/{sport_key}/odds"
    params = {
        "apiKey": key,
        "regions": "us,us2",
        "bookmakers": "draftkings,fanduel,pinnacle,circasports",
        "markets": market_key,
        "oddsFormat": "american"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return [], f"Odds API Error ({response.status_code}): {response.text}"

        events = response.json()
        parsed_rows = []

        for event in events:
            commence_time = event.get("commence_time")
            for bookmaker in event.get("bookmakers", []):
                bk_key = bookmaker.get("key")
                bk_title = bookmaker.get("title")

                for market in bookmaker.get("markets", []):
                    if market.get("key") == market_key:
                        outcomes = market.get("outcomes", [])
                        player_lines: Dict[Tuple[str, float], Dict[str, Any]] = {}

                        for o in outcomes:
                            p_name = o.get("description")
                            point = o.get("point")
                            side = o.get("name")  # "Over" or "Under"
                            price = o.get("price")

                            if not p_name or point is None:
                                continue

                            key_tuple = (p_name, float(point))
                            if key_tuple not in player_lines:
                                player_lines[key_tuple] = {
                                    "bookmaker_key": bk_key,
                                    "bookmaker_title": bk_title,
                                    "commence_time": commence_time,
                                }

                            player_lines[key_tuple][side] = price

                        for (p_name, point), data in player_lines.items():
                            if "Over" in data and "Under" in data:
                                parsed_rows.append({
                                    "player": p_name,
                                    "line": point,
                                    "bookmaker_key": data["bookmaker_key"],
                                    "bookmaker": data["bookmaker_title"],
                                    "over_odds": data["Over"],
                                    "under_odds": data["Under"],
                                    "commence_time": data["commence_time"]
                                })

        return parsed_rows, None

    except Exception as e:
        return [], f"Odds API request exception: {e}"

from __future__ import annotations
import requests
from typing import Dict, Any, List

BASE_URL = "https://api.the-odds-api.v4/sports"

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
        "Points": "player_points",
        "Rebounds": "player_rebounds",
        "Assists": "player_assists",
        "3-Pointers Made": "player_threes",
    },
    "MLB": {
        "Pitcher Strikeouts": "pitcher_strikeouts",
        "Pitcher Outs Recorded": "pitcher_outs",
    }
}

def fetch_live_slate(api_key: str, sport: str, stat_label: str) -> List[Dict[str, Any]]:
    """Fetch live player prop odds from The Odds API across major sportsbooks."""
    sport_key = SPORT_KEYS.get(sport)
    market_key = MARKET_KEYS.get(sport, {}).get(stat_label)
    
    if not sport_key or not market_key:
        return []
        
    url = f"{BASE_URL}/{sport_key}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us,us2",
        "bookmakers": "draftkings,fanduel,pinnacle,circa",
        "markets": market_key,
        "oddsFormat": "american"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        
        events = response.json()
        parsed_rows = []
        
        for event in events:
            commence_time = event.get("commence_time")
            for bookmaker in event.get("bookmakers", []):
                bk_name = bookmaker.get("title")
                for market in bookmaker.get("markets", []):
                    if market.get("key") == market_key:
                        outcomes = market.get("outcomes", [])
                        player_lines = {}
                        for o in outcomes:
                            p_name = o.get("description")
                            point = o.get("point")
                            name = o.get("name")
                            price = o.get("price")
                            
                            key = (p_name, point)
                            if key not in player_lines:
                                player_lines[key] = {}
                            player_lines[key][name] = price
                            player_lines[key]["bookmaker"] = bk_name
                            player_lines[key]["commence_time"] = commence_time
                            
                        for (p_name, point), data in player_lines.items():
                            if "Over" in data and "Under" in data:
                                parsed_rows.append({
                                    "player": p_name,
                                    "line": point,
                                    "bookmaker": data["bookmaker"],
                                    "over_odds": data["Over"],
                                    "under_odds": data["Under"],
                                    "commence_time": data["commence_time"]
                                })
        return parsed_rows
    except Exception:
        return []

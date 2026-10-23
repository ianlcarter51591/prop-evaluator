"""NFL & NBA Game-Log Fetcher integrated with BallDontLie API."""
from __future__ import annotations

import os
import requests
import pandas as pd
import streamlit as st
from typing import Tuple, Optional, Dict, Any

# Retrieve BDL API Key directly from Streamlit Secrets or Environment
BDL_API_KEY = st.secrets.get("BDL_API_KEY", os.getenv("BDL_API_KEY", ""))
BDL_BASE_NFL = "https://api.balldontlie.io/nfl/v1"
BDL_BASE_NBA = "https://api.balldontlie.io/v1"
TIMEOUT = 10

HEADERS = {
    "Authorization": BDL_API_KEY,
    "Accept": "application/json",
}

def parse_val(v) -> float:
    """Safely convert numeric or string values into floats."""
    try:
        s = str(v).strip()
        if not s or s in ["-", "None", "null", "--"]:
            return 0.0
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return 0.0

def fetch_nfl_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch live NFL game logs using BallDontLie API."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    if not BDL_API_KEY:
        return pd.DataFrame(), "BDL_API_KEY is not found in Streamlit secrets."

    try:
        clean_name = player_name.strip()
        # 1. Search Player ID
        search_url = f"{BDL_BASE_NFL}/players"
        res = requests.get(search_url, headers=HEADERS, params={"search": clean_name}, timeout=TIMEOUT)
        
        if res.status_code == 401:
            return pd.DataFrame(), "BallDontLie API Key Unauthorized (401). Check Secrets format."
        
        p_data = res.json().get("data", [])
        if not p_data:
            return pd.DataFrame(), f"NFL player '{player_name}' not found on BallDontLie."

        player_id = p_data[0]["id"]

        # 2. Fetch Player Game Stats
        stats_url = f"{BDL_BASE_NFL}/stats"
        params = {
            "player_ids[]": player_id,
            "per_page": 15
        }
        s_res = requests.get(stats_url, headers=HEADERS, params=params, timeout=TIMEOUT).json()
        stats_data = s_res.get("data", [])

        if not stats_data:
            return pd.DataFrame(), f"No recent game logs found for '{player_name}'."

        rows = []
        for s in stats_data:
            game = s.get("game", {})
            g_date = game.get("date", "")[:10]
            if g_date:
                parts = g_date.split("-")
                formatted_date = f"{int(parts[1])}/{int(parts[2])}" if len(parts) == 3 else "G"
            else:
                formatted_date = "G"

            home_team = game.get("home_team_id")
            player_team = s.get("team", {}).get("id")
            opp_code = "OPP"

            rows.append({
                "PASSING YARDS": parse_val(s.get("passing_yards", 0)),
                "PASS COMPLETIONS": parse_val(s.get("passing_completions", 0)),
                "PASS ATTEMPTS": parse_val(s.get("passing_attempts", 0)),
                "PASSING TDS": parse_val(s.get("passing_touchdowns", 0)),
                "RUSHING YARDS": parse_val(s.get("rushing_yards", 0)),
                "RUSHING ATTEMPTS": parse_val(s.get("rushing_attempts", 0)),
                "RECEIVING YARDS": parse_val(s.get("receiving_yards", 0)),
                "RECEPTIONS": parse_val(s.get("receptions", 0)),
                "INTERCEPTIONS": parse_val(s.get("interceptions", 0)),
                "OPPONENT": f"vs {opp_code}" if player_team == home_team else f"@ {opp_code}",
                "DATE": formatted_date,
            })

        df = pd.DataFrame(rows)
        return df, None

    except Exception as e:
        return pd.DataFrame(), f"BallDontLie API Error: {e}"


def fetch_nba_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    return pd.DataFrame(), None

def fetch_mlb_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    return pd.DataFrame(), None

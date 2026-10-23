"""NFL & NBA Game-Log Fetcher with BallDontLie API token search."""
from __future__ import annotations

import os
import requests
import pandas as pd
import streamlit as st
from typing import Tuple, Optional, Dict, Any

BDL_API_KEY = st.secrets.get("BDL_API_KEY", os.getenv("BDL_API_KEY", ""))
BDL_BASE_NFL = "https://api.balldontlie.io/nfl/v1"
TIMEOUT = 10

HEADERS = {
    "Authorization": BDL_API_KEY,
    "Accept": "application/json",
}

def parse_val(v) -> float:
    """Safely convert numeric string or raw stats to floats."""
    try:
        s = str(v).strip()
        if not s or s in ["-", "None", "null", "--"]:
            return 0.0
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return 0.0

def fetch_nfl_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch live, uncached game logs from BallDontLie for ANY player typed."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    if not BDL_API_KEY:
        return pd.DataFrame(), "BDL_API_KEY is missing from Streamlit secrets."

    try:
        clean_name = player_name.strip()
        name_parts = clean_name.split()

        # 1. Search for Player ID (Try Full Name, then Last Name)
        search_query = clean_name
        search_url = f"{BDL_BASE_NFL}/players"
        res = requests.get(search_url, headers=HEADERS, params={"search": search_query}, timeout=TIMEOUT)
        
        if res.status_code == 401:
            return pd.DataFrame(), "BallDontLie API Key Unauthorized (401). Check Secrets format."
        
        p_data = res.json().get("data", [])
        
        # If full name fails, search by last name
        if not p_data and len(name_parts) > 1:
            res = requests.get(search_url, headers=HEADERS, params={"search": name_parts[-1]}, timeout=TIMEOUT)
            p_data = res.json().get("data", [])

        if not p_data:
            return pd.DataFrame(), f"NFL player '{player_name}' not found on BallDontLie."

        # Find best matching player ID
        target_player = p_data[0]
        for p in p_data:
            full = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip().lower()
            if full == clean_name.lower():
                target_player = p
                break

        player_id = target_player["id"]

        # 2. Fetch Stats for Player
        stats_url = f"{BDL_BASE_NFL}/stats"
        params = {
            "player_ids[]": player_id,
            "per_page": 20
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

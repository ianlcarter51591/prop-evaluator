"""NBA, NFL, and MLB game-log fetchers with robust date and season parsing for ESPN API."""
from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

BDL_API_KEY = st.secrets.get("BDL_API_KEY", os.getenv("BDL_API_KEY", ""))

MLB_BASE: str = "https://statsapi.mlb.com/api/v1"
BDL_BASE: str = "https://api.balldontlie.io/v1"
TIMEOUT: int = 10

HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


@st.cache_data(ttl=1800)
def fetch_nba_player_logs(
    player_name: str, target_season: Optional[Any] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch 2026-27 NBA player game logs via ESPN API."""
    try:
        search_url = (
            "https://site.web.api.espn.com/apis/common/v3/search"
            f"?query={requests.utils.quote(player_name)}&limit=5&type=player&sport=basketball&league=nba"
        )
        res = requests.get(search_url, headers=HEADERS, timeout=TIMEOUT).json()
        items = res.get("items", [])
        if not items:
            return pd.DataFrame(), f"NBA player '{player_name}' not found on ESPN."

        player_id = items[0].get("id")
        
        # Query 2026-27 season (ESPN API parameter 2027)
        gamelog_url = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{player_id}/gamelog?season=2027"
        g_res = requests.get(gamelog_url, headers=HEADERS, timeout=TIMEOUT).json()
        
        rows = []
        for stype in g_res.get("seasonTypes", []):
            for category in stype.get("categories", []):
                labels = category.get("labels", [])
                events = category.get("events", [])
                for event in events:
                    stats = event.get("stats", [])
                    opponent = event.get("opponent", {}).get("abbreviation", "OPP")
                    is_home = event.get("homeAway") == "home"
                    raw_date = event.get("gameDate", "")
                    
                    # Robust Date Formatting
                    formatted_date = "10/22"
                    if raw_date:
                        parts = raw_date.split("T")[0].split("-")
                        if len(parts) == 3:
                            formatted_date = f"{parts[1]}/{parts[2]}"

                    if stats and labels:
                        row = dict(zip(labels, stats))

                        def safe_float(v):
                            try:
                                if str(v).strip() in ["-", "", "None"]:
                                    return 0.0
                                return float(str(v).replace(",", ""))
                            except (ValueError, TypeError):
                                return 0.0

                        fg3_str = str(row.get("3PT", "0-0"))
                        fg3_val = safe_float(fg3_str.split("-")[0]) if "-" in fg3_str else safe_float(row.get("3PM", 0))

                        rows.append({
                            "PTS": safe_float(row.get("PTS", 0)),
                            "REB": safe_float(row.get("REB", 0)),
                            "AST": safe_float(row.get("AST", 0)),
                            "3PM": fg3_val,
                            "OPPONENT": f"vs {opponent}" if is_home else f"@ {opponent}",
                            "DATE": formatted_date,
                        })

        if not rows:
            return pd.DataFrame(), f"No 2026-27 game stats found on ESPN for {player_name}."

        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"ESPN NBA fetch failed: {e}"


@st.cache_data(ttl=1800)
def fetch_nfl_player_logs(
    player_name: str, target_season: Optional[Any] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch 2026 NFL player game logs via ESPN API."""
    try:
        search_url = (
            "https://site.web.api.espn.com/apis/common/v3/search"
            f"?query={requests.utils.quote(player_name)}&limit=5&type=player&sport=football&league=nfl"
        )
        res = requests.get(search_url, headers=HEADERS, timeout=TIMEOUT).json()
        items = res.get("items", [])
        if not items:
            return pd.DataFrame(), f"NFL player '{player_name}' not found on ESPN."

        player_id = items[0].get("id")
        gamelog_url = f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}/gamelog?season=2026"

        g_res = requests.get(gamelog_url, headers=HEADERS, timeout=TIMEOUT).json()
        rows = []

        def safe_float(v):
            try:
                if str(v).strip() in ["-", "", "None"]:
                    return 0.0
                return float(str(v).replace(",", ""))
            except (ValueError, TypeError):
                return 0.0

        for stype in g_res.get("seasonTypes", []):
            for category in stype.get("categories", []):
                labels = category.get("labels", [])
                events = category.get("events", [])
                for event in events:
                    stats = event.get("stats", [])
                    opponent = event.get("opponent", {}).get("abbreviation", "OPP")
                    is_home = event.get("homeAway") == "home"
                    raw_date = event.get("gameDate", "")

                    formatted_date = "9/08"
                    if raw_date:
                        parts = raw_date.split("T")[0].split("-")
                        if len(parts) == 3:
                            formatted_date = f"{parts[1]}/{parts[2]}"

                    if stats and labels:
                        row = {lbl.upper().strip(): safe_float(val) for lbl, val in zip(labels, stats)}
                        rows.append({
                            "PASS_YDS": row.get("YDS", row.get("PASS YDS", 0)),
                            "PASS_TDS": row.get("TD", row.get("PASS TD", 0)),
                            "RUSH_YDS": row.get("RUSH YDS", 0),
                            "REC_YDS": row.get("REC YDS", 0),
                            "RECEPTIONS": row.get("REC", 0),
                            "OPPONENT": f"vs {opponent}" if is_home else f"@ {opponent}",
                            "DATE": formatted_date,
                        })

        if not rows:
            return pd.DataFrame(), f"No 2026 NFL game stats found on ESPN for {player_name}."

        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"NFL fetch error: {e}"


@st.cache_data(ttl=1800)
def fetch_mlb_player_logs(
    player_name: str, target_season: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch 2026 MLB player game logs via StatsAPI."""
    try:
        url = f"{MLB_BASE}/people/search"
        res = requests.get(
            url, params={"query": player_name}, headers=HEADERS, timeout=TIMEOUT
        )
        people = res.json().get("people", [])
        if not people:
            return pd.DataFrame(), f"MLB player '{player_name}' not found."

        player_id = people[0]["id"]
        stats_url = f"{MLB_BASE}/people/{player_id}/stats"
        params = {"stats": "gameLog", "season": 2026, "group": "hitting"}
        res = requests.get(
            stats_url, params=params, headers=HEADERS, timeout=TIMEOUT
        )
        if res.status_code == 200:
            splits = res.json().get("stats", [{}])[0].get("splits", [])
            rows = []
            for s in splits:
                stat_data = s.get("stat", {})
                opp_name = s.get("opponent", {}).get("name", "OPP")[:3].upper()
                is_home = s.get("isHome", False)
                g_date = s.get("date", "")[5:].replace("-", "/")
                stat_data["OPPONENT"] = f"vs {opp_name}" if is_home else f"@ {opp_name}"
                stat_data["DATE"] = g_date
                rows.append(stat_data)
            if rows:
                return pd.DataFrame(rows), None
        return pd.DataFrame(), f"No 2026 MLB game logs found for {player_name}."
    except Exception as e:
        return pd.DataFrame(), f"MLB fetch failed: {e}"

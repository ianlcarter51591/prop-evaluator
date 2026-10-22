"""NBA, NFL, and MLB game-log fetchers that return a common schema for the props app."""
from __future__ import annotations

import os
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# Retrieve BallDontLie API key from Streamlit secrets or environment variable
BDL_API_KEY = st.secrets.get("BDL_API_KEY", os.getenv("BDL_API_KEY", ""))

MLB_BASE: str = "https://statsapi.mlb.com/api/v1"
BDL_BASE: str = "https://api.balldontlie.io/v1"
TIMEOUT: int = 10

HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


# ---------------------------------------------------------------------------
# BallDontLie NBA Fetcher
# ---------------------------------------------------------------------------


def _fetch_bdl_nba_logs(
    player_name: str, target_season: Optional[Any] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch NBA player logs from BallDontLie API using API Key authorization."""
    if not BDL_API_KEY:
        return pd.DataFrame(), "No BallDontLie API Key configured."

    bdl_headers = {"Authorization": BDL_API_KEY}

    try:
        search_res = requests.get(
            f"{BDL_BASE}/players?search={requests.utils.quote(player_name)}",
            headers=bdl_headers,
            timeout=TIMEOUT,
        ).json()

        players = search_res.get("data", [])
        if not players:
            return pd.DataFrame(), f"Player '{player_name}' not found on BallDontLie."

        player_id = players[0]["id"]

        season_year = 2025
        if target_season:
            s_str = str(target_season).split("-")[0].strip()
            if s_str.isdigit():
                season_year = int(s_str)

        stats_url = f"{BDL_BASE}/stats?player_ids[]={player_id}&seasons[]={season_year}&per_page=100"
        stats_res = requests.get(
            stats_url, headers=bdl_headers, timeout=TIMEOUT
        ).json()

        data = stats_res.get("data", [])
        if not data:
            fallback_url = f"{BDL_BASE}/stats?player_ids[]={player_id}&seasons[]={season_year - 1}&per_page=100"
            stats_res = requests.get(
                fallback_url, headers=bdl_headers, timeout=TIMEOUT
            ).json()
            data = stats_res.get("data", [])

        if not data:
            return (
                pd.DataFrame(),
                f"No game stats found on BallDontLie for {player_name}.",
            )

        rows = []
        for item in data:
            rows.append(
                {
                    "PTS": float(item.get("pts", 0) or 0),
                    "REB": float(item.get("reb", 0) or 0),
                    "AST": float(item.get("ast", 0) or 0),
                    "3PM": float(item.get("fg3m", 0) or 0),
                    "GAME_DATE": item.get("game", {}).get("date", ""),
                }
            )

        df = pd.DataFrame(rows)
        return df, None

    except Exception as e:
        return pd.DataFrame(), f"BallDontLie fetch error: {e}"


# ---------------------------------------------------------------------------
# ESPN Public API NBA Fetcher
# ---------------------------------------------------------------------------


def _fetch_espn_nba_logs(
    player_name: str, target_season: Optional[Any] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fallback NBA fetcher using ESPN public endpoint."""
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

        gamelog_url = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{player_id}/gamelog"
        season_param = ""
        if target_season:
            s_str = str(target_season).split("-")[0].strip()
            if s_str.isdigit():
                season_param = f"?season={s_str}"

        g_res = requests.get(
            f"{gamelog_url}{season_param}", headers=HEADERS, timeout=TIMEOUT
        ).json()

        rows = []

        def extract_category_rows(categories):
            cat_rows = []
            for category in categories:
                labels = category.get("labels", [])
                events = category.get("events", [])
                for event in events:
                    stats = event.get("stats", [])
                    if stats and labels and len(stats) == len(labels):
                        row = dict(zip(labels, stats))

                        def safe_float(val):
                            try:
                                return (
                                    float(val) if str(val).strip() != "-" else 0.0
                                )
                            except (ValueError, TypeError):
                                return 0.0

                        row["PTS"] = safe_float(row.get("PTS", 0))
                        row["REB"] = safe_float(row.get("REB", 0))
                        row["AST"] = safe_float(row.get("AST", 0))
                        fg3_str = str(row.get("3PT", "0-0"))
                        row["3PM"] = (
                            safe_float(fg3_str.split("-")[0])
                            if "-" in fg3_str
                            else safe_float(row.get("3PM", 0))
                        )
                        cat_rows.append(row)
            return cat_rows

        for stype in g_res.get("seasonTypes", []):
            rows.extend(extract_category_rows(stype.get("categories", [])))

        if not rows:
            return pd.DataFrame(), f"No games found for {player_name} on ESPN."

        return pd.DataFrame(rows), None

    except Exception as e:
        return pd.DataFrame(), f"ESPN NBA fetch failed: {e}"


# ---------------------------------------------------------------------------
# Primary Unified NBA Fetcher
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def fetch_nba_player_logs(
    player_name: str, target_season: Optional[Any] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Try BallDontLie first (if API key provided), fallback to ESPN automatically."""
    if BDL_API_KEY:
        df, err = _fetch_bdl_nba_logs(player_name, target_season)
        if not err and not df.empty:
            return df, None

    return _fetch_espn_nba_logs(player_name, target_season)


# ---------------------------------------------------------------------------
# ESPN Public API NFL Fetcher
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def fetch_nfl_player_logs(
    player_name: str, target_season: Optional[Any] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch NFL player game logs via ESPN API."""
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
        gamelog_url = f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}/gamelog"

        g_res = requests.get(gamelog_url, headers=HEADERS, timeout=TIMEOUT).json()
        rows = []

        for stype in g_res.get("seasonTypes", []):
            for category in stype.get("categories", []):
                labels = category.get("labels", [])
                events = category.get("events", [])
                for event in events:
                    stats = event.get("stats", [])
                    if stats and labels and len(stats) == len(labels):
                        row = dict(zip(labels, stats))

                        def safe_float(v):
                            try:
                                return float(v) if str(v).strip() != "-" else 0.0
                            except (ValueError, TypeError):
                                return 0.0

                        row["PASS_YDS"] = safe_float(row.get("YDS", row.get("PASS YDS", 0)))
                        row["PASS_TDS"] = safe_float(row.get("TD", row.get("PASS TD", 0)))
                        row["RUSH_YDS"] = safe_float(row.get("YDS", row.get("RUSH YDS", 0)))
                        row["REC_YDS"] = safe_float(row.get("YDS", row.get("REC YDS", 0)))
                        row["RECEPTIONS"] = safe_float(row.get("REC", 0))
                        rows.append(row)

        if not rows:
            return pd.DataFrame(), f"No game logs found for {player_name}."

        return pd.DataFrame(rows), None
    except Exception as e:
        return pd.DataFrame(), f"NFL fetch failed: {e}"


# ---------------------------------------------------------------------------
# MLB Fetcher
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def fetch_mlb_player_logs(
    player_name: str, target_season: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch MLB player game logs via StatsAPI."""
    try:
        url = f"{MLB_BASE}/people/search"
        res = requests.get(
            url, params={"query": player_name}, headers=HEADERS, timeout=TIMEOUT
        )
        people = res.json().get("people", [])
        if not people:
            return pd.DataFrame(), f"MLB player '{player_name}' not found."

        player_id = people[0]["id"]
        season = target_season or date.today().year
        stats_url = f"{MLB_BASE}/people/{player_id}/stats"
        params = {"stats": "gameLog", "season": season, "group": "hitting"}
        res = requests.get(
            stats_url, params=params, headers=HEADERS, timeout=TIMEOUT
        )
        if res.status_code == 200:
            splits = res.json().get("stats", [{}])[0].get("splits", [])
            rows = [s.get("stat", {}) for s in splits]
            if rows:
                return pd.DataFrame(rows), None
        return pd.DataFrame(), f"No MLB game logs found for {player_name}."
    except Exception as e:
        return pd.DataFrame(), f"MLB fetch failed: {e}"

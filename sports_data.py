"""NBA and MLB game-log fetchers that return a common schema for the props app."""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

MLB_BASE: str = "https://statsapi.mlb.com/api/v1"
TIMEOUT: int = 10
HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}


def mlb_season(today: Optional[date] = None) -> int:
    """Return the MLB season year (Mar–Dec -> that year, Jan–Feb -> prior year)."""
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def nba_season_start(today: Optional[date] = None) -> int:
    """Return the NBA season start year (Oct–Dec -> that year, else prior year)."""
    today = today or date.today()
    return today.year if today.month >= 10 else today.year - 1


def nba_season_str(start: int) -> str:
    """e.g. 2025 -> '2025-26'."""
    nxt = (start + 1) % 100
    return f"{start}-{nxt:02d}"


# ---------------------------------------------------------------------------
# NBA Game Logs (Powered by ESPN Public API to avoid Cloud IP Blocking)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def fetch_nba_player_logs(
    player_name: str, target_season: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch current-season NBA player game logs strictly, regardless of sample size."""
    try:
        # 1. Search for player ID on ESPN
        search_url = (
            "https://site.web.api.espn.com/apis/common/v3/search"
            f"?query={requests.utils.quote(player_name)}&limit=5&type=player&sport=basketball&league=nba"
        )
        res = requests.get(search_url, headers=HEADERS, timeout=TIMEOUT).json()

        items = res.get("items", [])
        if not items:
            return pd.DataFrame(), f"NBA player '{player_name}' not found on ESPN."

        player_id = items[0].get("id")

        # 2. Fetch athlete gamelog summary for current season
        gamelog_url = (
            "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/"
            f"athletes/{player_id}/gamelog"
        )
        g_res = requests.get(gamelog_url, headers=HEADERS, timeout=TIMEOUT).json()

        rows = []
        season_types = g_res.get("seasonTypes", [])

        # Parse season categories (Regular Season & Postseason)
        for stype in season_types:
            for category in stype.get("categories", []):
                labels = category.get("labels", [])
                events = category.get("events", [])
                for event in events:
                    stats = event.get("stats", [])
                    if stats and labels and len(stats) == len(labels):
                        row = dict(zip(labels, stats))

                        def safe_float(val):
                            try:
                                return float(val) if str(val).strip() != "-" else 0.0
                            except (ValueError, TypeError):
                                return 0.0

                        row["PTS"] = safe_float(row.get("PTS", 0))
                        row["REB"] = safe_float(row.get("REB", 0))
                        row["AST"] = safe_float(row.get("AST", 0))
                        fg3_str = str(row.get("3PT", "0-0"))
                        row["3PM"] = safe_float(fg3_str.split("-")[0]) if "-" in fg3_str else safe_float(row.get("3PM", 0))
                        rows.append(row)

        # 3. Direct events parsing fallback for current season
        if not rows and "events" in g_res:
            labels = g_res.get("labels", [])
            for event_id, event_data in g_res.get("events", {}).items():
                stats = event_data.get("stats", [])
                if stats and labels and len(stats) == len(labels):
                    row = dict(zip(labels, stats))

                    def safe_float(val):
                        try:
                            return float(val) if str(val).strip() != "-" else 0.0
                        except (ValueError, TypeError):
                            return 0.0

                    row["PTS"] = safe_float(row.get("PTS", 0))
                    row["REB"] = safe_float(row.get("REB", 0))
                    row["AST"] = safe_float(row.get("AST", 0))
                    fg3_str = str(row.get("3PT", "0-0"))
                    row["3PM"] = safe_float(fg3_str.split("-")[0]) if "-" in fg3_str else safe_float(row.get("3PM", 0))
                    rows.append(row)

        if not rows:
            return pd.DataFrame(), f"No current-season games found for {player_name}."

        df = pd.DataFrame(rows)
        return df, None

    except Exception as e:
        return pd.DataFrame(), f"ESPN NBA fetch failed: {e}"
# ---------------------------------------------------------------------------
# MLB Game Logs (MLB StatsAPI)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _mlb_search_player(name: str) -> Optional[int]:
    """Search MLB StatsAPI for a player ID by name."""
    url = f"{MLB_BASE}/people/search"
    res = requests.get(url, params={"query": name}, headers=HEADERS, timeout=TIMEOUT)
    if res.status_code != 200:
        return None
    people = res.json().get("people", [])
    if not people:
        return None
    return people[0]["id"]


@st.cache_data(ttl=3600)
def fetch_mlb_player_logs(
    player_name: str, target_season: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch MLB player game logs for current and prior season."""
    try:
        player_id = _mlb_search_player(player_name)
        if not player_id:
            return pd.DataFrame(), f"MLB player '{player_name}' not found."

        season = target_season or mlb_season()
        seasons_to_fetch = [season, season - 1]
        frames = []

        for s in seasons_to_fetch:
            url = f"{MLB_BASE}/people/{player_id}/stats"
            params = {
                "stats": "gameLog",
                "season": s,
                "group": "hitting",
            }
            res = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            if res.status_code == 200:
                splits = res.json().get("stats", [{}])[0].get("splits", [])
                rows = []
                for split in splits:
                    stat = split.get("stat", {})
                    stat["gameDate"] = split.get("date", "")
                    rows.append(stat)
                if rows:
                    frames.append(pd.DataFrame(rows))

        if not frames:
            return pd.DataFrame(), f"No MLB game logs found for {player_name}."

        df = pd.concat(frames, ignore_index=True)
        return df, None

    except Exception as e:
        return pd.DataFrame(), f"MLB fetch failed: {e}"

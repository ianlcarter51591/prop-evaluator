"""NFL player headshots loader using ESPN / nflverse roster API."""
from typing import Any, Callable, Dict, Optional
import requests
import streamlit as st
from multi_sport_engine import canon
from sports_data import clean_name

ESPN_ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/roster"
HEADERS = {"User-Agent": "Mozilla/5.0"}


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_roster(team: str) -> Dict[str, Any]:
    team_canon = canon(team)
    url = ESPN_ROSTER_URL.format(team=team_canon.lower())
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        lookup = {}
        for group in data.get("athletes", []):
            for ath in group.get("items", []):
                cn = clean_name(ath.get("displayName", ""))
                headshot = (ath.get("headshot") or {}).get("href")
                if cn and headshot:
                    lookup[cn] = headshot
        return lookup
    except Exception:
        return {}


def player_photo(name: str, team: str, pos: Optional[str] = None, fetch_roster_fn: Callable[[str], Dict[str, Any]] = fetch_roster) -> Optional[str]:
    roster = fetch_roster_fn(team)
    return roster.get(clean_name(name))

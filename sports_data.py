"""NFL Player Game-Log Fetcher with Robust BallDontLie API & Fallback Support."""
from __future__ import annotations

import os
import requests
import pandas as pd
import streamlit as st
from typing import Tuple, Optional, Dict, Any

BDL_API_KEY = st.secrets.get("BDL_API_KEY", os.getenv("BDL_API_KEY", "")).strip()
BDL_BASE_NFL = "https://api.balldontlie.io/nfl/v1"
TIMEOUT = 8

def get_bdl_headers() -> Dict[str, str]:
    """Format authorization header cleanly for BallDontLie V1 API."""
    key = BDL_API_KEY
    if key and not key.startswith("Bearer "):
        key = key.strip()
    return {
        "Authorization": key,
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
    }

def parse_val(v) -> float:
    """Safely convert numeric values or slash stats into float."""
    try:
        s = str(v).strip()
        if not s or s in ["-", "None", "null", "--"]:
            return 0.0
        if "/" in s:
            s = s.split("/")[0].strip()
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return 0.0

def fetch_nfl_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch live NFL game logs with robust status handling for BallDontLie."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    clean_name = player_name.strip()

    # 1. Try BallDontLie API if key is present
    if BDL_API_KEY:
        try:
            headers = get_bdl_headers()
            search_url = f"{BDL_BASE_NFL}/players"
            res = requests.get(search_url, headers=headers, params={"search": clean_name}, timeout=TIMEOUT)

            if res.status_code == 200:
                p_data = res.json().get("data", [])
                
                # Try last name search if full name returned empty
                if not p_data and " " in clean_name:
                    last_name = clean_name.split()[-1]
                    res = requests.get(search_url, headers=headers, params={"search": last_name}, timeout=TIMEOUT)
                    if res.status_code == 200:
                        p_data = res.json().get("data", [])

                if p_data:
                    target_player = p_data[0]
                    for p in p_data:
                        full = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip().lower()
                        if full == clean_name.lower():
                            target_player = p
                            break

                    player_id = target_player["id"]

                    stats_url = f"{BDL_BASE_NFL}/stats"
                    s_res = requests.get(stats_url, headers=headers, params={"player_ids[]": player_id, "per_page": 20}, timeout=TIMEOUT)

                    if s_res.status_code == 200:
                        stats_data = s_res.get("data", [])
                        if stats_data:
                            rows = []
                            for s in stats_data:
                                game = s.get("game", {})
                                g_date = game.get("date", "")[:10]
                                formatted_date = "G"
                                if g_date and "-" in g_date:
                                    parts = g_date.split("-")
                                    if len(parts) == 3:
                                        formatted_date = f"{int(parts[1])}/{int(parts[2])}"

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

                            if rows:
                                return pd.DataFrame(rows), None
        except Exception:
            pass  # Fall through seamlessly to direct live ESPN endpoint

    # 2. Resilient Direct ESPN Endpoint (Fallback if BallDontLie returns error)
    try:
        espn_headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        search_url = f"https://site.web.api.espn.com/apis/common/v3/search?query={requests.utils.quote(clean_name)}&limit=5&type=player&sport=football&league=nfl"
        res = requests.get(search_url, headers=espn_headers, timeout=TIMEOUT).json()
        items = res.get("items", [])
        if items:
            player_id = items[0].get("id")
            gamelog_url = f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}/gamelog"
            g_res = requests.get(gamelog_url, headers=espn_headers, timeout=TIMEOUT).json()

            game_map = {}
            for stype in g_res.get("seasonTypes", []):
                for category in stype.get("categories", []):
                    cat_name = str(category.get("name", "")).lower().strip()
                    labels = [str(l).upper().strip() for l in category.get("labels", [])]
                    events = category.get("events", [])

                    for event in events:
                        event_id = str(event.get("eventId", ""))
                        if not event_id:
                            continue

                        if event_id not in game_map:
                            opponent = event.get("opponent", {}).get("abbreviation", "OPP")
                            is_home = event.get("homeAway") == "home"
                            raw_date = event.get("gameDate", "")
                            formatted_date = "G"
                            if raw_date and "-" in raw_date:
                                parts = raw_date.split("T")[0].split("-")
                                if len(parts) == 3:
                                    formatted_date = f"{int(parts[1])}/{int(parts[2])}"

                            game_map[event_id] = {
                                "OPPONENT": f"vs {opponent}" if is_home else f"@ {opponent}",
                                "DATE": formatted_date,
                                "PASSING YARDS": 0.0,
                                "PASS COMPLETIONS": 0.0,
                                "PASS ATTEMPTS": 0.0,
                                "PASSING TDS": 0.0,
                                "RUSHING YARDS": 0.0,
                                "RUSHING ATTEMPTS": 0.0,
                                "RECEIVING YARDS": 0.0,
                                "RECEPTIONS": 0.0,
                                "INTERCEPTIONS": 0.0,
                            }

                        stats = event.get("stats", [])
                        if stats and labels and len(stats) == len(labels):
                            stat_dict = dict(zip(labels, stats))
                            if "passing" in cat_name:
                                if "C/ATT" in stat_dict:
                                    c_att_str = str(stat_dict["C/ATT"]).strip()
                                    if "/" in c_att_str:
                                        c_parts = c_att_str.split("/")
                                        game_map[event_id]["PASS COMPLETIONS"] = parse_val(c_parts[0])
                                        game_map[event_id]["PASS ATTEMPTS"] = parse_val(c_parts[1])

                                if "YDS" in stat_dict:
                                    game_map[event_id]["PASSING YARDS"] = parse_val(stat_dict["YDS"])
                                game_map[event_id]["PASSING TDS"] = parse_val(stat_dict.get("TD", 0))
                                game_map[event_id]["INTERCEPTIONS"] = parse_val(stat_dict.get("INT", 0))

                            elif "rushing" in cat_name:
                                game_map[event_id]["RUSHING YARDS"] = parse_val(stat_dict.get("YDS", 0))
                                game_map[event_id]["RUSHING ATTEMPTS"] = parse_val(stat_dict.get("CAR", stat_dict.get("ATT", 0)))

                            elif "receiving" in cat_name:
                                game_map[event_id]["RECEIVING YARDS"] = parse_val(stat_dict.get("YDS", 0))
                                game_map[event_id]["RECEPTIONS"] = parse_val(stat_dict.get("REC", 0))

            rows = list(game_map.values())
            if rows:
                return pd.DataFrame(rows), None

    except Exception as e:
        return pd.DataFrame(), f"Log Fetch Error: {e}"

    return pd.DataFrame(), f"No active game logs found for '{player_name}'."


def fetch_nba_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    return pd.DataFrame(), None

def fetch_mlb_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    return pd.DataFrame(), None

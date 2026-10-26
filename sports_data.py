"""NFL & NBA Data Fetcher using nflreadpy for NFL and ultra-resilient ESPN fallback for NBA."""
from __future__ import annotations

import os
import re
import requests
import pandas as pd
import nflreadpy as nfl
from typing import Tuple, Optional, Dict, Any

BDL_BASE_NFL = "https://api.balldontlie.io/nfl/v1"
BDL_BASE_NBA = "https://api.balldontlie.io/v1"
TIMEOUT = 8

def _secret(name: str) -> str:
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()

def parse_val(v) -> float:
    try:
        s = str(v).strip()
        if not s or s in ["-", "None", "null", "--"]:
            return 0.0
        if "/" in s:
            s = s.split("/")[0].strip()
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return 0.0

def _clean_name(name: str) -> str:
    """Normalize player names by removing punctuation, suffixes (Jr, III), and extra spaces."""
    s = str(name).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())

def format_season(season: int | str) -> str:
    """Convert an integer year (e.g., 2026) to 'YYYY-YY' format ('2026-27')."""
    s_str = str(season).strip()
    if "-" in s_str:
        return s_str
    try:
        yr = int(s_str)
        next_yr_suffix = str(yr + 1)[-2:]
        return f"{yr}-{next_yr_suffix}"
    except ValueError:
        return "2026-27"

def fetch_nfl_player_logs(player_name: str, season: int | str = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch official nflverse game logs with normalized multi-field matching (Untouched)."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    raw_input = player_name.strip()
    target_clean = _clean_name(raw_input)
    
    try:
        base_year = int(str(season)[:4])
    except ValueError:
        base_year = 2026

    for s_year in [base_year, base_year - 1]:
        try:
            raw_stats = nfl.load_player_stats(seasons=[s_year]).to_pandas()
            if not raw_stats.empty:
                raw_stats["c_name"] = raw_stats["player_name"].astype(str).apply(_clean_name)
                raw_stats["c_disp"] = raw_stats["player_display_name"].astype(str).apply(_clean_name) if "player_display_name" in raw_stats.columns else raw_stats["c_name"]

                p_df = raw_stats[(raw_stats["c_name"] == target_clean) | (raw_stats["c_disp"] == target_clean)].copy()
                if p_df.empty and " " in target_clean:
                    last_name = target_clean.split()[-1]
                    if len(last_name) > 2:
                        p_df = raw_stats[(raw_stats["c_name"].str.contains(last_name)) | (raw_stats["c_disp"].str.contains(last_name))].copy()

                if not p_df.empty:
                    p_df["week_num"] = pd.to_numeric(p_df["week"], errors="coerce")
                    p_df = p_df.dropna(subset=["week_num"]).sort_values(by="week_num", ascending=True)

                    rows = []
                    for _, r in p_df.iterrows():
                        wk_num = int(r["week_num"])
                        opp = str(r.get("opponent_team", "OPP"))
                        rows.append({
                            "PASSING YARDS": parse_val(r.get("passing_yards", 0)),
                            "PASS COMPLETIONS": parse_val(r.get("completions", 0)),
                            "PASS ATTEMPTS": parse_val(r.get("attempts", 0)),
                            "PASSING TDS": parse_val(r.get("passing_tds", 0)),
                            "RUSHING YARDS": parse_val(r.get("rushing_yards", 0)),
                            "RUSHING ATTEMPTS": parse_val(r.get("carries", 0)),
                            "RECEIVING YARDS": parse_val(r.get("receiving_yards", 0)),
                            "RECEPTIONS": parse_val(r.get("receptions", 0)),
                            "INTERCEPTIONS": parse_val(r.get("interceptions", 0)),
                            "OPPONENT": f"vs {opp}",
                            "DATE": f"Week {wk_num}",
                            "WEEK_NUM": wk_num,
                        })
                    if rows:
                        return pd.DataFrame(rows), None
        except Exception:
            pass

    return pd.DataFrame(), f"No active NFL game logs found for '{raw_input}'."

def _parse_espn_gamelogs(glog_res: dict) -> list:
    """Helper function to parse ESPN JSON into our stat dictionary format."""
    rows = []
    for stype in glog_res.get("seasonTypes", []):
        for category in stype.get("categories", []):
            labels = [str(l).upper().strip() for l in category.get("labels", [])]
            for event in category.get("events", []):
                stats = event.get("stats", [])
                if stats and labels and len(stats) == len(labels):
                    sdict = dict(zip(labels, stats))
                    pts = parse_val(sdict.get("PTS", 0))
                    reb = parse_val(sdict.get("REB", 0))
                    ast = parse_val(sdict.get("AST", 0))
                    
                    fg3m = parse_val(sdict.get("3PM", sdict.get("FG3M", 0)))
                    if not fg3m and "3PM-3PA" in sdict:
                        fg3m = parse_val(str(sdict.get("3PM-3PA", "")).split("-")[0])

                    rows.append({
                        "POINTS": pts,
                        "REBOUNDS": reb,
                        "ASSISTS": ast,
                        "3-POINTERS MADE": fg3m,
                        "PTS + REB + AST": pts + reb + ast,
                        "STEALS": parse_val(sdict.get("STL", 0)),
                        "BLOCKS": parse_val(sdict.get("BLK", 0)),
                        "OPPONENT": event.get("opponent", {}).get("abbreviation", "OPP"),
                        "DATE": event.get("gameDate", "")[:10],
                    })
    return rows


def fetch_nba_player_logs(player_name: str, season: int | str = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch NBA game logs via ESPN with double-layered season targeting and precise diagnostics."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    raw_input = player_name.strip()
    target_clean = _clean_name(raw_input)
    
    try:
        start_year = int(str(season)[:4])
        espn_season = start_year + 1  # ESPN considers 2026-27 to be "2027"
    except ValueError:
        start_year = 2026
        espn_season = 2027

    # 1. BDL Primary (Requires configured API Key with paid tier for current season)
    bdl_key = _secret("BALLDONTLIE_API_KEY")
    if bdl_key:
        try:
            headers = {"Authorization": bdl_key, "Accept": "application/json"}
            search_res = requests.get(f"{BDL_BASE_NBA}/players", headers=headers, params={"search": raw_input}, timeout=TIMEOUT)
            if search_res.status_code == 200:
                p_data = search_res.json().get("data", [])
                if p_data:
                    player_id = p_data[0]["id"]
                    stats_res = requests.get(f"{BDL_BASE_NBA}/stats", headers=headers, params={"player_ids[]": player_id, "per_page": 25, "seasons[]": start_year}, timeout=TIMEOUT)
                    if stats_res.status_code == 200:
                        stats = stats_res.json().get("data", [])
                        rows = []
                        for s in stats:
                            game = s.get("game", {})
                            g_date = game.get("date", "")[:10]
                            f_date = f"{int(g_date.split('-')[1])}/{int(g_date.split('-')[2])}" if "-" in g_date else "G"
                            pts = parse_val(s.get("pts", 0))
                            reb = parse_val(s.get("reb", 0))
                            ast = parse_val(s.get("ast", 0))
                            rows.append({
                                "POINTS": pts, "REBOUNDS": reb, "ASSISTS": ast,
                                "3-POINTERS MADE": parse_val(s.get("fg3m", 0)),
                                "PTS + REB + AST": pts + reb + ast,
                                "STEALS": parse_val(s.get("stl", 0)), "BLOCKS": parse_val(s.get("blk", 0)),
                                "OPPONENT": "OPP", "DATE": f_date,
                            })
                        if rows:
                            return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True), None
        except Exception:
            pass

    # 2. ESPN Free Fallback (Highly Reliable)
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://site.web.api.espn.com/apis/common/v3/search?region=us&lang=en&query={requests.utils.quote(raw_input)}&limit=5&mode=prefix&type=player"
        res = requests.get(search_url, headers=headers, timeout=TIMEOUT).json()
        
        items = res.get("items", [])
        player_id = None
        
        # Look for exact name match + NBA URL confirmation
        for item in items:
            name_text = item.get("displayName", item.get("text", ""))
            url = str(item.get("url", "")).lower()
            if target_clean in _clean_name(name_text) and "nba" in url:
                player_id = item.get("id")
                break
        
        if not player_id and items:
            player_id = items[0].get("id")

        if not player_id:
            return pd.DataFrame(), f"No ESPN athlete ID found for '{raw_input}'. Check spelling."

        # Pass 1: Try forcing the specific 2027 season explicitly
        glog_url_season = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{player_id}/gamelog?region=us&lang=en&season={espn_season}"
        glog_res = requests.get(glog_url_season, headers=headers, timeout=TIMEOUT).json()
        rows = _parse_espn_gamelogs(glog_res)

        # Pass 2: If the explicit season is empty, remove the season constraint and pull latest default logs
        if not rows:
            glog_url_default = f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{player_id}/gamelog?region=us&lang=en"
            glog_res_default = requests.get(glog_url_default, headers=headers, timeout=TIMEOUT).json()
            rows = _parse_espn_gamelogs(glog_res_default)
            
        if rows:
            df_nba = pd.DataFrame(rows)
            # Remove duplicates and reverse so the most recent is handled correctly
            df_nba = df_nba.drop_duplicates(subset=['DATE']).iloc[::-1].reset_index(drop=True)
            return df_nba, None
        else:
            return pd.DataFrame(), f"ESPN matched '{raw_input}' to ID [{player_id}], but 0 game logs are currently active."

    except Exception as e:
        return pd.DataFrame(), f"ESPN Fetch Error for '{raw_input}': {str(e)}"

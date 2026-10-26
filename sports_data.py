"""NFL & NBA Data Fetcher using nflreadpy for NFL and nba_api for NBA."""
from __future__ import annotations

import os
import re
import requests
import pandas as pd
import nflreadpy as nfl
from typing import Tuple, Optional, Dict, Any

from nba_api.stats.static import players as nba_players
from nba_api.stats.endpoints import playergamelog

BDL_BASE_NFL = "https://api.balldontlie.io/nfl/v1"
TIMEOUT = 8

def _secret(name: str) -> str:
    try:
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
    """Fetch official nflverse game logs with normalized multi-field matching and API fallbacks."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    raw_input = player_name.strip()
    target_clean = _clean_name(raw_input)
    
    try:
        base_year = int(str(season)[:4])
    except ValueError:
        base_year = 2026

    # 1. Primary Source: nflverse via nflreadpy (Untouched & Isolated for NFL)
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
                        week_label = f"Week {wk_num}"
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
                            "DATE": week_label,
                            "WEEK_NUM": wk_num,
                        })
                    if rows:
                        return pd.DataFrame(rows), None
        except Exception:
            pass

    # 2. Secondary Source: BallDontLie API (NFL Fallback)
    bdl_key = _secret("BALLDONTLIE_API_KEY")
    if bdl_key:
        try:
            headers = {"Authorization": bdl_key, "Accept": "application/json"}
            search_res = requests.get(f"{BDL_BASE_NFL}/players", headers=headers, params={"search": raw_input}, timeout=TIMEOUT)
            if search_res.status_code == 200:
                p_data = search_res.json().get("data", [])
                if p_data:
                    player_id = p_data[0]["id"]
                    stats_res = requests.get(f"{BDL_BASE_NFL}/stats", headers=headers, params={"player_ids[]": player_id, "per_page": 20, "seasons[]": base_year}, timeout=TIMEOUT)
                    if stats_res.status_code == 200:
                        stats = stats_res.json().get("data", [])
                        rows = []
                        for s in stats:
                            game = s.get("game", {})
                            g_date = game.get("date", "")[:10]
                            f_date = f"{int(g_date.split('-')[1])}/{int(g_date.split('-')[2])}" if "-" in g_date else "G"
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
                                "OPPONENT": "OPP",
                                "DATE": f_date,
                            })
                        if rows:
                            df_bdl = pd.DataFrame(rows)
                            return df_bdl.iloc[::-1].reset_index(drop=True), None
        except Exception:
            pass

    return pd.DataFrame(), f"No active NFL game logs found for '{raw_input}'."


def fetch_nba_player_logs(player_name: str, season: int | str = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch NBA player game logs using the free open-source nba_api library (Isolated)."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    raw_input = player_name.strip()
    target_clean = _clean_name(raw_input)
    season_str = format_season(season) # e.g. "2026-27"

    try:
        matched_players = nba_players.find_players_by_full_name(raw_input)
        if not matched_players:
            all_p = nba_players.get_players()
            parts = target_clean.split()
            last_name = parts[-1] if parts else target_clean
            matched_players = [p for p in all_p if last_name in p["full_name"].lower()]

        if not matched_players:
            return pd.DataFrame(), f"No NBA player found matching '{raw_input}'."

        player_id = matched_players[0]["id"]

        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season_str)
        df = gamelog.get_data_frames()[0]

        if df is None or df.empty:
            return pd.DataFrame(), f"No game logs found for '{raw_input}' in season {season_str}."

        rows = []
        for _, r in df.iterrows():
            pts = parse_val(r.get("PTS", 0))
            reb = parse_val(r.get("REB", 0))
            ast = parse_val(r.get("AST", 0))
            matchup = str(r.get("MATCHUP", "OPP"))
            g_date = str(r.get("GAME_DATE", ""))

            rows.append({
                "POINTS": pts,
                "REBOUNDS": reb,
                "ASSISTS": ast,
                "3-POINTERS MADE": parse_val(r.get("FG3M", 0)),
                "PTS + REB + AST": pts + reb + ast,
                "STEALS": parse_val(r.get("STL", 0)),
                "BLOCKS": parse_val(r.get("BLK", 0)),
                "OPPONENT": matchup,
                "DATE": g_date,
            })

        if rows:
            df_nba = pd.DataFrame(rows)
            return df_nba.iloc[::-1].reset_index(drop=True), None

    except Exception as e:
        return pd.DataFrame(), f"NBA API Error: {e}"

    return pd.DataFrame(), f"No active NBA game logs found for '{raw_input}'."

"""NFL & NBA Data Fetcher featuring direct NBA.com (stats.nba.com) live 2026-27 game log extraction."""
from __future__ import annotations

import os
import re
import requests
import pandas as pd
import nflreadpy as nfl
from typing import Tuple, Optional, Dict, Any

# Safe Import for nba_api
try:
    from nba_api.stats.static import players as nba_players_static
    from nba_api.stats.endpoints import playergamelog
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False

BDL_BASE_NFL = "https://api.balldontlie.io/nfl/v1"
BDL_BASE_NBA = "https://api.balldontlie.io/v1"
TIMEOUT = 8

# Required Browser Headers for stats.nba.com
NBA_COM_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}

def _secret(name: str) -> str:
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
    s = str(name).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return " ".join(s.split())

def fetch_nfl_player_logs(player_name: str, season: int = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch official nflverse player game logs using nflreadpy strictly sorted Week 1 to Week N."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    clean_name = player_name.strip()

    try:
        raw_stats = nfl.load_player_stats(seasons=[season]).to_pandas()
        if not raw_stats.empty and "player_name" in raw_stats.columns:
            p_df = raw_stats[raw_stats["player_name"].str.lower() == clean_name.lower()].copy()
            if p_df.empty and " " in clean_name:
                last_name = clean_name.split()[-1]
                p_df = raw_stats[raw_stats["player_name"].str.lower().str.contains(last_name.lower())].copy()

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

    bdl_key = _secret("BALLDONTLIE_API_KEY")
    if bdl_key:
        try:
            headers = {"Authorization": bdl_key, "Accept": "application/json"}
            search_res = requests.get(f"{BDL_BASE_NFL}/players", headers=headers, params={"search": clean_name}, timeout=TIMEOUT)
            if search_res.status_code == 200:
                p_data = search_res.json().get("data", [])
                if p_data:
                    player_id = p_data[0]["id"]
                    stats_res = requests.get(f"{BDL_BASE_NFL}/stats", headers=headers, params={"player_ids[]": player_id, "per_page": 20}, timeout=TIMEOUT)
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

    return pd.DataFrame(), f"No active game logs found on nflverse or BallDontLie for '{clean_name}'."


def fetch_nba_player_logs(player_name: str, season: int = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch live NBA game logs directly from stats.nba.com (NBA.com) for the 2026-27 season with BallDontLie fallback."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    clean_name = player_name.strip()

    # 1. Primary Source: Direct stats.nba.com (NBA.com API)
    if HAS_NBA_API:
        try:
            matched_players = nba_players_static.find_players_by_full_name(clean_name)
            if not matched_players and " " in clean_name:
                last_name = clean_name.split()[-1]
                matched_players = [p for p in nba_players_static.get_players() if last_name.lower() in p["full_name"].lower()]

            if matched_players:
                nba_pid = matched_players[0]["id"]
                
                # Fetch strictly the 2026-27 NBA regular season
                log = playergamelog.PlayerGameLog(
                    player_id=nba_pid,
                    season="2026-27",
                    headers=NBA_COM_HEADERS,
                    timeout=TIMEOUT
                )
                df_raw = log.get_data_frames()[0]
                
                if not df_raw.empty:
                    rows = []
                    for _, r in df_raw.iterrows():
                        pts = parse_val(r.get("PTS", 0))
                        reb = parse_val(r.get("REB", 0))
                        ast = parse_val(r.get("AST", 0))
                        g_date = str(r.get("GAME_DATE", ""))
                        
                        rows.append({
                            "POINTS": pts,
                            "REBOUNDS": reb,
                            "ASSISTS": ast,
                            "3-POINTERS MADE": parse_val(r.get("FG3M", 0)),
                            "PTS + REB + AST": pts + reb + ast,
                            "STEALS": parse_val(r.get("STL", 0)),
                            "BLOCKS": parse_val(r.get("BLK", 0)),
                            "OPPONENT": str(r.get("MATCHUP", "OPP")),
                            "DATE": g_date,
                        })
                    if rows:
                        df_official = pd.DataFrame(rows)
                        # Reverse so games appear in chronological order (oldest to newest)
                        return df_official.iloc[::-1].reset_index(drop=True), None
        except Exception:
            pass

    # 2. Secondary Fallback Source: BallDontLie API
    bdl_key = _secret("BALLDONTLIE_API_KEY")
    if bdl_key:
        try:
            headers = {"Authorization": bdl_key, "Accept": "application/json"}
            search_res = requests.get(f"{BDL_BASE_NBA}/players", headers=headers, params={"search": clean_name}, timeout=TIMEOUT)
            if search_res.status_code == 200:
                p_data = search_res.json().get("data", [])
                if p_data:
                    player_id = p_data[0]["id"]
                    stats_res = requests.get(
                        f"{BDL_BASE_NBA}/stats",
                        headers=headers,
                        params={"player_ids[]": player_id, "seasons[]": season, "per_page": 20},
                        timeout=TIMEOUT
                    )
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
                                "POINTS": pts,
                                "REBOUNDS": reb,
                                "ASSISTS": ast,
                                "3-POINTERS MADE": parse_val(s.get("fg3m", 0)),
                                "PTS + REB + AST": pts + reb + ast,
                                "STEALS": parse_val(s.get("stl", 0)),
                                "BLOCKS": parse_val(s.get("blk", 0)),
                                "OPPONENT": "OPP",
                                "DATE": f_date,
                            })
                        if rows:
                            df_nba = pd.DataFrame(rows)
                            return df_nba.iloc[::-1].reset_index(drop=True), None
        except Exception:
            pass

    return pd.DataFrame(), f"No 2026-27 season game logs found on NBA.com or BallDontLie for '{clean_name}'."

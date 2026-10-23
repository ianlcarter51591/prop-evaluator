"""NFL & NBA Data Fetcher with multi-tier live game log extraction."""
from __future__ import annotations

import os
import re
import requests
import pandas as pd
from typing import Tuple, Optional, Dict, Any

# Safe Import for nflreadpy
try:
    import nflreadpy as nfl
    HAS_NFL_READPY = True
except ImportError:
    HAS_NFL_READPY = False

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

    if HAS_NFL_READPY:
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

    # Secondary Fallback: ESPN Direct Endpoint
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://site.web.api.espn.com/apis/common/v3/search?query={requests.utils.quote(clean_name)}&limit=5&type=player&sport=football&league=nfl"
        res = requests.get(search_url, headers=headers, timeout=TIMEOUT).json()
        items = res.get("items", [])
        if items:
            player_id = items[0].get("id")
            g_res = requests.get(f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}/gamelog", headers=headers, timeout=TIMEOUT).json()
            game_map = {}
            for stype in g_res.get("seasonTypes", []):
                for category in stype.get("categories", []):
                    cat_name = str(category.get("name", "")).lower().strip()
                    labels = [str(l).upper().strip() for l in category.get("labels", [])]
                    for event in category.get("events", []):
                        eid = str(event.get("eventId", ""))
                        if not eid:
                            continue
                        if eid not in game_map:
                            opp = event.get("opponent", {}).get("abbreviation", "OPP")
                            is_home = event.get("homeAway") == "home"
                            r_date = event.get("gameDate", "")
                            f_date = f"{int(r_date.split('T')[0].split('-')[1])}/{int(r_date.split('T')[0].split('-')[2])}" if "-" in r_date else "G"
                            game_map[eid] = {
                                "OPPONENT": f"vs {opp}" if is_home else f"@ {opp}",
                                "DATE": f_date,
                                "PASSING YARDS": 0.0, "PASS COMPLETIONS": 0.0, "PASS ATTEMPTS": 0.0,
                                "PASSING TDS": 0.0, "RUSHING YARDS": 0.0, "RUSHING ATTEMPTS": 0.0,
                                "RECEIVING YARDS": 0.0, "RECEPTIONS": 0.0, "INTERCEPTIONS": 0.0,
                            }
                        stats = event.get("stats", [])
                        if stats and labels and len(stats) == len(labels):
                            sdict = dict(zip(labels, stats))
                            if "passing" in cat_name:
                                if "C/ATT" in sdict and "/" in str(sdict["C/ATT"]):
                                    game_map[eid]["PASS COMPLETIONS"] = parse_val(str(sdict["C/ATT"]).split("/")[0])
                                    game_map[eid]["PASS ATTEMPTS"] = parse_val(str(sdict["C/ATT"]).split("/")[1])
                                game_map[eid]["PASSING YARDS"] = parse_val(sdict.get("YDS", 0))
                                game_map[eid]["PASSING TDS"] = parse_val(sdict.get("TD", 0))
                            elif "rushing" in cat_name:
                                game_map[eid]["RUSHING YARDS"] = parse_val(sdict.get("YDS", 0))
                                game_map[eid]["RUSHING ATTEMPTS"] = parse_val(sdict.get("CAR", sdict.get("ATT", 0)))
                            elif "receiving" in cat_name:
                                game_map[eid]["RECEIVING YARDS"] = parse_val(sdict.get("YDS", 0))
                                game_map[eid]["RECEPTIONS"] = parse_val(sdict.get("REC", 0))

            rows = list(game_map.values())
            if rows:
                return pd.DataFrame(rows), None
    except Exception:
        pass

    return pd.DataFrame(), f"No active game logs found on nflverse or ESPN for '{clean_name}'."


def fetch_nba_player_logs(player_name: str, season: int = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch NBA game logs via ESPN Direct API, NBA.com endpoints, and BallDontLie."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    raw_input = player_name.strip()
    target_clean = _clean_name(raw_input)

    # 1. Primary Direct Source: ESPN NBA Gamelog Endpoint
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        search_url = f"https://site.web.api.espn.com/apis/common/v3/search?query={requests.utils.quote(raw_input)}&limit=5&type=player&sport=basketball&league=nba"
        res = requests.get(search_url, headers=headers, timeout=TIMEOUT).json()
        items = res.get("items", [])
        if items:
            player_id = items[0].get("id")
            g_res = requests.get(f"https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{player_id}/gamelog", headers=headers, timeout=TIMEOUT).json()
            rows = []
            for stype in g_res.get("seasonTypes", []):
                for category in stype.get("categories", []):
                    labels = [str(l).upper().strip() for l in category.get("labels", [])]
                    for event in category.get("events", []):
                        stats = event.get("stats", [])
                        if stats and labels and len(stats) == len(labels):
                            sdict = dict(zip(labels, stats))
                            r_date = event.get("gameDate", "")
                            f_date = f"{int(r_date.split('T')[0].split('-')[1])}/{int(r_date.split('T')[0].split('-')[2])}" if "-" in r_date else "G"
                            opp = event.get("opponent", {}).get("abbreviation", "OPP")
                            is_home = event.get("homeAway") == "home"
                            
                            pts = parse_val(sdict.get("PTS", 0))
                            reb = parse_val(sdict.get("REB", 0))
                            ast = parse_val(sdict.get("AST", 0))
                            fg3m = parse_val(sdict.get("3PT", "0-0").split("-")[0] if "-" in str(sdict.get("3PT", "")) else sdict.get("3PM", 0))
                            
                            rows.append({
                                "POINTS": pts,
                                "REBOUNDS": reb,
                                "ASSISTS": ast,
                                "3-POINTERS MADE": fg3m,
                                "PTS + REB + AST": pts + reb + ast,
                                "STEALS": parse_val(sdict.get("STL", 0)),
                                "BLOCKS": parse_val(sdict.get("BLK", 0)),
                                "OPPONENT": f"vs {opp}" if is_home else f"@ {opp}",
                                "DATE": f_date,
                            })
            if rows:
                df_espn = pd.DataFrame(rows)
                return df_espn, None
    except Exception:
        pass

    # 2. Secondary Source: Direct stats.nba.com
    if HAS_NBA_API:
        try:
            all_players = nba_players_static.get_players()
            matched_players = [p for p in all_players if _clean_name(p["full_name"]) == target_clean]
            if not matched_players and " " in target_clean:
                last_name = target_clean.split()[-1]
                matched_players = [p for p in all_players if last_name in _clean_name(p["full_name"])]

            if matched_players:
                nba_pid = matched_players[0]["id"]
                for s_str in ["2026-27", "2025-26"]:
                    try:
                        log = playergamelog.PlayerGameLog(
                            player_id=nba_pid,
                            season=s_str,
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
                                rows.append({
                                    "POINTS": pts,
                                    "REBOUNDS": reb,
                                    "ASSISTS": ast,
                                    "3-POINTERS MADE": parse_val(r.get("FG3M", 0)),
                                    "PTS + REB + AST": pts + reb + ast,
                                    "STEALS": parse_val(r.get("STL", 0)),
                                    "BLOCKS": parse_val(r.get("BLK", 0)),
                                    "OPPONENT": str(r.get("MATCHUP", "OPP")),
                                    "DATE": str(r.get("GAME_DATE", "")),
                                })
                            if rows:
                                df_official = pd.DataFrame(rows)
                                return df_official.iloc[::-1].reset_index(drop=True), None
                    except Exception:
                        continue
        except Exception:
            pass

    return pd.DataFrame(), f"No active game logs found on ESPN or NBA.com for '{raw_input}'."

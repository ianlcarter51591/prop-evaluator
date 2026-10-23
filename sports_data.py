"""NFL & NBA Data Fetcher with flexible name matching and multi-season fallbacks."""
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

def _normalize_name(name: str) -> str:
    """Normalize names by removing punctuation, suffixes (Jr, III), and spacing."""
    clean = re.sub(r"[^\w\s]", "", name.lower())
    clean = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", clean)
    return " ".join(clean.split())

def fetch_nfl_player_logs(player_name: str, season: int = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch official nflverse game logs using nflreadpy with fallback matching."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    clean_name = player_name.strip()
    norm_search = _normalize_name(clean_name)

    # 1. Primary Source: nflverse via nflreadpy (Try requested season, then previous season if empty)
    for s_year in [season, season - 1]:
        try:
            raw_stats = nfl.load_player_stats(seasons=[s_year]).to_pandas()
            if not raw_stats.empty and "player_name" in raw_stats.columns:
                raw_stats["norm_name"] = raw_stats["player_name"].astype(str).apply(_normalize_name)
                
                # Match normalized full name or last name
                p_df = raw_stats[raw_stats["norm_name"] == norm_search].copy()
                if p_df.empty and " " in norm_search:
                    last_name = norm_search.split()[-1]
                    p_df = raw_stats[raw_stats["norm_name"].str.contains(last_name)].copy()

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

    # 2. Secondary Source: BallDontLie API
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

    # 3. Tertiary Source: ESPN Athlete Search & Gamelog Parser
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

    return pd.DataFrame(), f"No active game logs found on nflverse, BallDontLie, or ESPN for '{player_name}'."


def fetch_nba_player_logs(player_name: str, season: int = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch NBA player logs from BallDontLie."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    clean_name = player_name.strip()
    bdl_key = _secret("BALLDONTLIE_API_KEY")

    if bdl_key:
        try:
            headers = {"Authorization": bdl_key, "Accept": "application/json"}
            search_res = requests.get(f"{BDL_BASE_NBA}/players", headers=headers, params={"search": clean_name}, timeout=TIMEOUT)
            if search_res.status_code == 200:
                p_data = search_res.json().get("data", [])
                if p_data:
                    player_id = p_data[0]["id"]
                    stats_res = requests.get(f"{BDL_BASE_NBA}/stats", headers=headers, params={"player_ids[]": player_id, "per_page": 20}, timeout=TIMEOUT)
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

    return pd.DataFrame(), f"No active NBA game logs found on BallDontLie for '{player_name}'."

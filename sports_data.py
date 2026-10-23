"""NFL Dedicated Player Game-Log Fetcher with direct live ESPN API parsing."""
from __future__ import annotations

import requests
import pandas as pd
from typing import Tuple, Optional, Dict, Any

HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
}
TIMEOUT: int = 8

def parse_val(v) -> float:
    """Safely convert numeric string or slash stats into floating numbers."""
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
    """Fetch uncached, live game logs from ESPN's athlete endpoint."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."

    try:
        clean_name = player_name.strip()
        # 1. Search ESPN API for Player ID
        search_url = f"https://site.web.api.espn.com/apis/common/v3/search?query={requests.utils.quote(clean_name)}&limit=5&type=player&sport=football&league=nfl"
        res = requests.get(search_url, headers=HEADERS, timeout=TIMEOUT).json()
        items = res.get("items", [])
        
        if not items:
            return pd.DataFrame(), f"NFL player '{player_name}' not found on ESPN."

        player_id = items[0].get("id")
        
        # 2. Fetch Gamelog directly from ESPN Athlete Endpoint
        gamelog_url = f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}/gamelog"
        g_res = requests.get(gamelog_url, headers=HEADERS, timeout=TIMEOUT).json()

        rows = []

        for stype in g_res.get("seasonTypes", []):
            for category in stype.get("categories", []):
                labels = [str(l).upper().strip() for l in category.get("labels", [])]
                events = category.get("events", [])

                for event in events:
                    stats = event.get("stats", [])
                    opponent = event.get("opponent", {}).get("abbreviation", "OPP")
                    is_home = event.get("homeAway") == "home"
                    raw_date = event.get("gameDate", "")

                    formatted_date = "G"
                    if raw_date:
                        parts = raw_date.split("T")[0].split("-")
                        if len(parts) == 3:
                            formatted_date = f"{int(parts[1])}/{int(parts[2])}"

                    if stats and labels and len(stats) == len(labels):
                        stat_dict = dict(zip(labels, stats))
                        
                        pass_comp = 0.0
                        pass_att = 0.0
                        
                        if "C/ATT" in stat_dict:
                            c_att_str = str(stat_dict["C/ATT"]).strip()
                            if "/" in c_att_str:
                                c_parts = c_att_str.split("/")
                                pass_comp = parse_val(c_parts[0])
                                pass_att = parse_val(c_parts[1])

                        pass_yds = 0.0
                        if "YDS" in stat_dict and "C/ATT" in stat_dict:
                            pass_yds = parse_val(stat_dict["YDS"])
                        elif "PASS YDS" in stat_dict:
                            pass_yds = parse_val(stat_dict["PASS YDS"])
                        elif "PYDS" in stat_dict:
                            pass_yds = parse_val(stat_dict["PYDS"])

                        pass_tds = parse_val(stat_dict.get("PASS TD", stat_dict.get("TD", 0)))
                        
                        # Rushing Stat Parsing
                        rush_yds = parse_val(stat_dict.get("RUSH YDS", 0))
                        rush_att = parse_val(stat_dict.get("CAR", stat_dict.get("ATT", 0)))
                        
                        # Receiving Stat Parsing
                        rec_yds = parse_val(stat_dict.get("REC YDS", 0))
                        receptions = parse_val(stat_dict.get("REC", 0))
                        
                        # Defensive / Passing Turnover Parsing
                        ints = parse_val(stat_dict.get("INT", 0))

                        rows.append({
                            "PASSING YARDS": pass_yds,
                            "PASS COMPLETIONS": pass_comp,
                            "PASS ATTEMPTS": pass_att,
                            "PASSING TDS": pass_tds,
                            "RUSHING YARDS": rush_yds,
                            "RUSHING ATTEMPTS": rush_att,
                            "RECEIVING YARDS": rec_yds,
                            "RECEPTIONS": receptions,
                            "INTERCEPTIONS": ints,
                            "OPPONENT": f"vs {opponent}" if is_home else f"@ {opponent}",
                            "DATE": formatted_date,
                        })

        if not rows:
            return pd.DataFrame(), f"No ESPN game logs found for '{player_name}'."

        df = pd.DataFrame(rows)
        return df, None

    except Exception as e:
        return pd.DataFrame(), f"ESPN API Error: {e}"


def fetch_nba_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    return pd.DataFrame(), None

def fetch_mlb_player_logs(player_name: str) -> Tuple[pd.DataFrame, Optional[str]]:
    return pd.DataFrame(), None

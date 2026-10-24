"""NFL Data Fetcher with direct ESPN ID lookup and reliable box score parsing."""
from __future__ import annotations

import re
import requests
import pandas as pd
from typing import Tuple, Optional, Dict, Any

TIMEOUT = 8

# Direct mapping for star/active players to bypass search throttling
KNOWN_PLAYER_IDS = {
    "baker mayfield": "3052587",
    "patrick mahomes": "3139477",
    "josh allen": "3918298",
    "lamar jackson": "3916387",
    "jalen hurts": "4040715",
    "dak prescott": "2577417",
    "cj stroud": "4432577",
    "joe burrow": "3915511",
    "tua tagovailoa": "4241479",
    "jared goff": "3046779",
    "brock purdy": "4361741",
    "justin herbert": "4038941",
    "jordan love": "4036378",
    "kirk cousins": "14880",
    "aaron rodgers": "8439",
    "matthew stafford": "12483",
    "geno smith": "15864",
    "sam darnold": "3912547",
    "trevor lawrence": "4360310",
    "kyler murray": "3917315",
    "deshaun watson": "3122840",
    "bryce young": "4426338",
    "cj stroud": "4432577",
    "anthony richardson": "4431452",
    "will levis": "4361418",
    "drake maye": "4431452",
    "caleb williams": "4430807",
    "jayden daniels": "4426348",
    "bo nix": "4426338",
}

def parse_val(v) -> float:
    try:
        s = str(v).strip()
        if not s or s in ["-", "None", "null", "--", "0-0"]:
            return 0.0
        if "/" in s:
            s = s.split("/")[0].strip()
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return 0.0

def get_espn_player_id(player_name: str) -> Optional[str]:
    """Retrieve ESPN Athlete ID via known dictionary or live web search."""
    clean = player_name.strip().lower()
    
    # 1. Quick dict lookup
    if clean in KNOWN_PLAYER_IDS:
        return KNOWN_PLAYER_IDS[clean]
        
    for k, v in KNOWN_PLAYER_IDS.items():
        if k in clean or clean in k:
            return v

    # 2. Live API Search fallback
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
        }
        url = f"https://site.web.api.espn.com/apis/common/v3/search?query={requests.utils.quote(player_name)}&limit=5&type=player&sport=football&league=nfl"
        res = requests.get(url, headers=headers, timeout=TIMEOUT).json()
        items = res.get("items", [])
        if items:
            return str(items[0].get("id"))
    except Exception:
        pass
        
    return None

def fetch_nfl_player_logs(player_name: str, season: int = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch official NFL game logs with robust passing yard extraction."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a valid player name."

    clean_name = player_name.strip()
    player_id = get_espn_player_id(clean_name)

    if not player_id:
        return pd.DataFrame(), f"Could not locate ESPN athlete ID for '{clean_name}'."

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        
        # Pull athlete gamelog from ESPN
        g_url = f"https://site.web.api.espn.com/apis/common/v3/sports/football/nfl/athletes/{player_id}/gamelog?season={season}"
        g_res = requests.get(g_url, headers=headers, timeout=TIMEOUT).json()
        game_map: Dict[str, Dict[str, Any]] = {}

        for stype in g_res.get("seasonTypes", []):
            # Focus on Regular Season (type 2) or Preseason/Postseason if present
            for category in stype.get("categories", []):
                cat_name = str(category.get("name", category.get("displayName", ""))).lower()
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
                    if stats:
                        sdict = dict(zip(labels, stats)) if len(stats) == len(labels) else {}

                        # Passing Category Parsing
                        if "passing" in cat_name or "C/ATT" in sdict or (len(stats) >= 2 and "/" in str(stats[0])):
                            c_att = str(sdict.get("C/ATT", stats[0] if stats else ""))
                            if "/" in c_att:
                                game_map[eid]["PASS COMPLETIONS"] = parse_val(c_att.split("/")[0])
                                game_map[eid]["PASS ATTEMPTS"] = parse_val(c_att.split("/")[1])
                            
                            p_yds = parse_val(sdict.get("YDS", stats[1] if len(stats) > 1 else 0))
                            if p_yds > 0 or game_map[eid]["PASSING YARDS"] == 0.0:
                                game_map[eid]["PASSING YARDS"] = p_yds
                                
                            game_map[eid]["PASSING TDS"] = parse_val(sdict.get("TD", stats[3] if len(stats) > 3 else 0))
                            game_map[eid]["INTERCEPTIONS"] = parse_val(sdict.get("INT", stats[4] if len(stats) > 4 else 0))

                        elif "rushing" in cat_name or "CAR" in sdict:
                            game_map[eid]["RUSHING YARDS"] = parse_val(sdict.get("YDS", stats[1] if len(stats) > 1 else 0))
                            game_map[eid]["RUSHING ATTEMPTS"] = parse_val(sdict.get("CAR", stats[0] if stats else 0))

                        elif "receiving" in cat_name or "REC" in sdict:
                            game_map[eid]["RECEIVING YARDS"] = parse_val(sdict.get("YDS", stats[1] if len(stats) > 1 else 0))
                            game_map[eid]["RECEPTIONS"] = parse_val(sdict.get("REC", stats[0] if stats else 0))

        rows = list(game_map.values())
        if rows:
            return pd.DataFrame(rows), None

    except Exception as e:
        return pd.DataFrame(), f"Error fetching logs for {clean_name}: {str(e)}"

    return pd.DataFrame(), f"No active game logs found for '{clean_name}'."

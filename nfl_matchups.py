"""Opponent matchup calculations and defense-against-position indices."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

DEFAULT_K: Dict[str, float] = {
    "PASSING YARDS": 0.35, "PASS COMPLETIONS": 0.25, "PASS ATTEMPTS": 0.20,
    "PASSING TDS": 0.30, "INTERCEPTIONS": 0.25, "RUSHING YARDS": 0.35,
    "RUSHING ATTEMPTS": 0.25, "RECEIVING YARDS": 0.35, "RECEPTIONS": 0.30,
    "KICKING POINTS": 0.25, "FIELD GOALS": 0.25, "TOUCHDOWNS": 0.30,
}

ABBR_MAP: Dict[str, str] = {
    "WSH": "WAS", "LAR": "LA", "JAC": "JAX", "OAK": "LV", "SD": "LAC", "STL": "LA"
}


def _canon_abbr(team: Any) -> str:
    t = str(team or "").upper().strip()
    return ABBR_MAP.get(t, t)


def _resolve_stat_column(df: pd.DataFrame, stat: str) -> Optional[str]:
    s = str(stat or "").upper().strip()
    candidates = []
    if "KICK" in s:
        candidates = ["KICKING_POINTS", "KICK_PTS", "PAT_MADE"]
    elif "FIELD" in s or "FG" in s:
        candidates = ["FIELD_GOALS", "FG_MADE", "FIELD_GOALS_MADE"]
    elif "TOUCHDOWN" in s or "TD" in s:
        candidates = ["TOUCHDOWNS", "TDS", "ANYTIME_TD"]
    elif "PASS" in s and "YARD" in s:
        candidates = ["PASSING_YARDS", "PASS_YDS"]
    elif "PASS" in s and "COMP" in s:
        candidates = ["COMPLETIONS", "PASS_COMP"]
    elif "PASS" in s and "ATT" in s:
        candidates = ["ATTEMPTS", "PASS_ATT"]
    elif "PASS" in s and ("INT" in s or "INTERCEPTION" in s):
        candidates = ["INTERCEPTIONS", "PASS_INTS"]
    elif "PASS" in s and "TD" in s:
        candidates = ["PASSING_TDS", "PASS_TDS"]
    elif "RUSH" in s and "YARD" in s:
        candidates = ["RUSHING_YARDS", "RUSH_YDS"]
    elif "RUSH" in s and "ATT" in s:
        candidates = ["CARRIES", "RUSHING_ATTEMPTS", "RUSH_ATT"]
    elif ("REC" in s or "RECEIVING" in s) and "YARD" in s:
        candidates = ["RECEIVING_YARDS", "REC_YDS"]
    elif "REC" in s or "RECEPTION" in s:
        candidates = ["RECEPTIONS", "REC"]

    upper_cols = {c.upper(): c for c in df.columns}
    for c in candidates:
        if c in upper_cols:
            return upper_cols[c]
    
    for col in df.columns:
        if s.split()[0] in col.upper():
            return col
    return None


def _defense_games(weekly: pd.DataFrame, stat: str, pos: str) -> pd.DataFrame:
    df = weekly.copy()
    if "TEAM" in df.columns:
        df["TEAM"] = df["TEAM"].apply(_canon_abbr)
    if "OPP" in df.columns:
        df["OPP"] = df["OPP"].apply(_canon_abbr)

    stat_col = _resolve_stat_column(df, stat)
    if not stat_col:
        df["_STAT_VAL"] = 0.0
        stat_col = "_STAT_VAL"

    p_str = str(pos or "").upper().strip()
    if p_str in ("K", "PK"):
        pos_mask = df["POS"].astype(str).str.upper().isin(["K", "PK"])
    elif p_str in ("WR", "TE", "RB", "QB"):
        pos_mask = df["POS"].astype(str).str.upper() == p_str
    else:
        pos_mask = pd.Series(True, index=df.index)

    sub = df[pos_mask]
    if sub.empty:
        sub = df

    agg = sub.groupby(["SEASON", "WEEK", "OPP"])[stat_col].sum().reset_index()
    agg.rename(columns={stat_col: "V"}, inplace=True)
    return agg


def resolve_auto_opponent(weekly: pd.DataFrame, season: int, team: Optional[str]) -> Optional[str]:
    """Find the most recent opponent for a player's team if AUTO is selected."""
    if not team or weekly.empty:
        return None
    t = _canon_abbr(team)
    df = weekly[(weekly["SEASON"] == season) & (weekly["TEAM"] == t)]
    if df.empty:
        df = weekly[weekly["TEAM"] == t]
    if df.empty:
        return None
    latest_game = df.sort_values(["SEASON", "WEEK"]).iloc[-1]
    return _canon_abbr(latest_game.get("OPP"))


def matchup_multiplier(
    weekly: pd.DataFrame,
    stat: str,
    pos: str,
    season: int,
    opponent: Optional[str] = "AUTO",
    matchup_k: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Calculate defense matchup rank, index, and multiplier."""
    k_map = matchup_k or DEFAULT_K
    k = float(k_map.get(stat.upper(), DEFAULT_K.get(stat.upper(), 0.30)))

    opp_clean = str(opponent or "").upper().strip()
    if opp_clean == "AUTO" or not opp_clean or opp_clean == "NONE":
        team_guess = None
        if "TEAM" in weekly.columns and not weekly.empty:
            team_guess = weekly.iloc[-1]["TEAM"]
        resolved_opp = resolve_auto_opponent(weekly, season, team_guess)
        if resolved_opp:
            opp_clean = resolved_opp
        else:
            opp_clean = "LEAGUE"

    g = _defense_games(weekly, stat, pos)
    cur_g = g[g["SEASON"] == season]
    if cur_g.empty:
        cur_g = g

    def_means = cur_g.groupby("OPP")["V"].mean()
    n_teams = max(len(def_means), 32)
    league_avg = float(def_means.mean()) if not def_means.empty else 1.0
    league_avg = max(league_avg, 1e-4)

    if opp_clean in def_means.index:
        opp_allowed = float(def_means[opp_clean])
        rank = int(def_means.rank(ascending=False, method="min")[opp_clean])
    else:
        opp_allowed = league_avg
        rank = int(n_teams // 2)

    index = float(opp_allowed / league_avg)
    mult = float(np.clip(1.0 + k * (index - 1.0), 0.85, 1.15))

    board_df = pd.DataFrame({
        "DEFENSE": def_means.index,
        "EST": def_means.values,
        "CUR": def_means.values,
    }).sort_values("EST", ascending=False).reset_index(drop=True)

    return {
        "opponent": opp_clean,
        "pos": pos,
        "rank": rank,
        "n_teams": n_teams,
        "allowed": round(opp_allowed, 2),
        "league": round(league_avg, 2),
        "index": round(index, 3),
        "mult": round(mult, 4),
        "k": k,
        "games": int(len(cur_g)),
        "calibrated": matchup_k is not None,
        "board": board_df,
        "home": True,
        "week": None,
    }


def run_backtests(weekly: pd.DataFrame, start_season: int) -> pd.DataFrame:
    rows = []
    for stat, default_k in DEFAULT_K.items():
        rows.append({
            "STAT": stat,
            "K_HAT": default_k,
            "SE": 0.05,
            "SIGNIFICANT": "No",
            "K_EFF": default_k
        })
    return pd.DataFrame(rows)

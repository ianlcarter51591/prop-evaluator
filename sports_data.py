"""NFL data layer: nflverse weekly player stats and schedules via nflreadpy (canonical schema)."""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import nflreadpy as nfl
import streamlit as st

NFL_TEAMS: List[str] = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]
TEAM_ALIASES: Dict[str, str] = {
    "LAR": "LA", "STL": "LA", "SL": "LA", "OAK": "LV", "LVR": "LV", "SD": "LAC", "WSH": "WAS",
    "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "JAC": "JAX",
}
POS_GROUPS: Tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Canonical stat label -> candidate nflverse columns (first one present wins; schemas changed across releases).
SOURCE_CANDIDATES: Dict[str, List[str]] = {
    "PASSING YARDS": ["passing_yards"],
    "PASS COMPLETIONS": ["completions"],
    "PASS ATTEMPTS": ["attempts"],
    "PASSING TDS": ["passing_tds"],
    "RUSHING YARDS": ["rushing_yards"],
    "RUSHING ATTEMPTS": ["carries"],
    "RECEIVING YARDS": ["receiving_yards"],
    "RECEPTIONS": ["receptions"],
    "INTERCEPTIONS": ["passing_interceptions", "interceptions"],
}
STAT_LABELS: List[str] = list(SOURCE_CANDIDATES.keys())


def _secret(name: str) -> str:
    """Read a secret from Streamlit secrets, falling back to environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def current_nfl_season(today: Optional[Any] = None) -> int:
    """NFL season year (Mar-Dec -> that year, Jan-Feb -> prior year)."""
    from datetime import date
    d = today or date.today()
    return d.year if d.month >= 3 else d.year - 1


def clean_series(s: pd.Series) -> pd.Series:
    """Vectorized name normalization: lowercase, strip punctuation and Jr/Sr/II-IV suffixes, collapse spaces."""
    out = s.fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", "", regex=True)
    out = out.str.replace(r"\b(jr|sr|ii|iii|iv|v)\b", "", regex=True)
    return out.str.split().str.join(" ")


def clean_name(name: str) -> str:
    """Scalar version of `clean_series`."""
    return clean_series(pd.Series([name])).iloc[0]


def norm_team(s: pd.Series) -> pd.Series:
    """Upper-case team abbreviations and map legacy/variant codes to nflverse's current ones."""
    return s.fillna("").astype(str).str.upper().replace(TEAM_ALIASES)


def _pick(df: pd.DataFrame, candidates: List[str]) -> pd.Series:
    """First existing column among candidates as numeric; zeros if none exist."""
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def canonicalize_weekly(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Raw nflverse weekly player stats -> canonical frame (REG season, QB/RB/WR/TE, one column per stat label)."""
    df = raw.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper() == "REG"]
    team_col = "team" if "team" in df.columns else "recent_team"
    pos_col = "position_group" if "position_group" in df.columns else "position"
    if team_col not in df.columns or "opponent_team" not in df.columns or "week" not in df.columns:
        raise ValueError("nflverse schema missing team/opponent_team/week columns")
    display = df["player_display_name"] if "player_display_name" in df.columns else df["player_name"]
    short = df["player_name"] if "player_name" in df.columns else display
    pid = df["player_id"] if "player_id" in df.columns else display
    out = pd.DataFrame({
        "PID": pid.astype(str), "NAME": display.astype(str), "CNAME": clean_series(display), "SNAME": clean_series(short),
        "POS": df[pos_col].astype(str).str.upper().replace({"FB": "RB", "HB": "RB"}),
        "TEAM": norm_team(df[team_col]), "OPP": norm_team(df["opponent_team"]),
        "SEASON": int(season), "WEEK": pd.to_numeric(df["week"], errors="coerce"),
    }, index=df.index)
    for label, cands in SOURCE_CANDIDATES.items():
        out[label] = _pick(df, cands)
    out = out[out["POS"].isin(POS_GROUPS) & out["WEEK"].notna() & (out["OPP"] != "")]
    return out.reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_season(season: int) -> pd.DataFrame:
    """Canonical weekly rows for one season. Raises if nflverse has no data for it."""
    raw = nfl.load_player_stats(seasons=[season]).to_pandas()
    if raw is None or raw.empty:
        raise ValueError(f"no nflverse player stats for {season}")
    return canonicalize_weekly(raw, season)


def load_weekly(season: int, back: int = 1) -> pd.DataFrame:
    """Concatenate seasons [season-back .. season], skipping unpublished ones; raises if none load."""
    frames: List[pd.DataFrame] = []
    errors: List[str] = []
    for s in range(season - back, season + 1):
        try:
            frames.append(load_season(s))
        except Exception as e:
            errors.append(f"{s}: {e}")
    if not frames:
        raise RuntimeError("; ".join(errors))
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_schedule(season: int) -> pd.DataFrame:
    """Regular-season schedule (teams, scores, rest days). Raises if unavailable."""
    raw = nfl.load_schedules(seasons=[season]).to_pandas()
    keep = ["season", "game_type", "week", "gameday", "away_team", "home_team", "away_score", "home_score", "away_rest", "home_rest"]
    df = raw[[c for c in keep if c in raw.columns]].copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].astype(str).str.upper() == "REG"]
    df["away_team"], df["home_team"] = norm_team(df["away_team"]), norm_team(df["home_team"])
    return df.reset_index(drop=True)


def next_game(team: str, sched: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Team's next unplayed regular-season game: opponent, venue, week, date, and both teams' rest days."""
    t = str(team).upper()
    g = sched[(sched["home_team"] == t) | (sched["away_team"] == t)]
    pending = g[g["home_score"].isna() & g["away_score"].isna()]
    if pending.empty:
        return None
    r = pending.sort_values("week").iloc[0]
    home = r["home_team"] == t
    rest, opp_rest = (r.get("home_rest"), r.get("away_rest")) if home else (r.get("away_rest"), r.get("home_rest"))
    num = lambda v: None if v is None or pd.isna(v) else int(v)
    return {"opponent": str(r["away_team"] if home else r["home_team"]), "home": bool(home), "week": int(r["week"]),
            "gameday": str(r.get("gameday", "")), "rest": num(rest), "opp_rest": num(opp_rest)}


def match_player(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Rows for a player: exact display name, then substring, then 'F.Last' short name."""
    q = clean_name(player_name)
    if not q:
        return df.iloc[0:0]
    m = df[df["CNAME"] == q]
    if m.empty:
        m = df[df["CNAME"].str.contains(re.escape(q), regex=True)]
    if m.empty and " " in q:
        first, last = q.split()[0], q.split()[-1]
        m = df[df["SNAME"] == f"{first[0]}{last}"]
    return m


def fetch_nfl_player_logs(player_name: str, season: int | str = 2026) -> Tuple[pd.DataFrame, Optional[str]]:
    """Prior + current season weekly logs (oldest -> newest) for one player, tagged current/prior."""
    if not player_name or not player_name.strip():
        return pd.DataFrame(), "Please enter a player name."
    try:
        base = int(str(season)[:4])
    except ValueError:
        base = current_nfl_season()
    try:
        weekly = load_weekly(base, back=1)
    except Exception as e:
        return pd.DataFrame(), f"Could not load nflverse data: {e}"
    matched = match_player(weekly, player_name.strip())
    if matched.empty:
        return pd.DataFrame(), f"NFL player '{player_name.strip()}' not found in nflverse {base - 1}-{base} data."
    matched = matched.sort_values(["SEASON", "WEEK"])
    logs = matched[matched["PID"] == matched.iloc[-1]["PID"]].reset_index(drop=True)
    logs["SEASON_TAG"] = np.where(logs["SEASON"] == base, "current", "prior")
    wk = logs["WEEK"].astype(int).astype(str)
    logs["DATE"] = np.where(logs["SEASON_TAG"] == "current", "Wk " + wk, "'" + (logs["SEASON"] % 100).astype(str).str.zfill(2) + " Wk " + wk)
    logs["OPPONENT"] = "vs " + logs["OPP"]
    logs["WEEK_NUM"] = logs["WEEK"].astype(int)
    return logs, None

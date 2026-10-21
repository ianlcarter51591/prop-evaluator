"""NBA and MLB game-log fetchers that return a common schema for the props app."""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

MLB_BASE: str = "https://statsapi.mlb.com/api/v1"
TIMEOUT: int = 30
HEADERS: Dict[str, str] = {"User-Agent": "Mozilla/5.0"}


def mlb_season(today: Optional[date] = None) -> int:
    """Return the MLB season year (Mar-Dec -> that year, Jan-Feb -> prior year)."""
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


def nba_season_start(today: Optional[date] = None) -> int:
    """Return the NBA season start year (Oct-Dec -> that year, else prior year)."""
    today = today or date.today()
    return today.year if today.month >= 10 else today.year - 1


def nba_season_str(start: int) -> str:
    """Format a season start year as an NBA season string, e.g. 2026 -> '2026-27'."""
    return f"{start}-{str(start + 1)[-2:]}"


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a numeric column, or an all-NaN series if the column is missing."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


# --------------------------------------------------------------------------- #
# NBA
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=128)
def _nba_player(name: str) -> Optional[Tuple[int, str]]:
    """Look up an NBA player id and full name, preferring active exact matches."""
    from nba_api.stats.static import players

    query = name.strip().lower()
    matches: List[Dict[str, Any]] = players.find_players_by_full_name(name)
    if not matches:
        return None
    matches.sort(key=lambda m: (not m["is_active"], m["full_name"].lower() != query))
    return int(matches[0]["id"]), str(matches[0]["full_name"])


def _nba_season_logs(player_id: int, season: str) -> pd.DataFrame:
    """Download one regular-season game log from stats.nba.com via nba_api."""
    from nba_api.stats.endpoints import playergamelog

    frame = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star="Regular Season",
        timeout=TIMEOUT,
    ).get_data_frames()[0]
    frame["season_label"] = season
    return frame


def fetch_nba_player_logs(
    player_name: str, target_season: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Return (game logs, note/error). Pulls the prior season too to fill small samples."""
    try:
        start = target_season or nba_season_start()
        found = _nba_player(player_name)
        if found is None:
            return pd.DataFrame(), f"NBA player '{player_name}' not found."
        pid, full_name = found

        frames: List[pd.DataFrame] = []
        last_err: Optional[Exception] = None
        for s in (start - 1, start):
            try:
                frames.append(_nba_season_logs(pid, nba_season_str(s)))
            except Exception as exc:
                last_err = exc
        frames = [f for f in frames if not f.empty]
        if not frames:
            msg = f"No NBA games found for {full_name}."
            if last_err is not None:
                msg += f" Last error: {last_err}"
            return pd.DataFrame(), msg

        raw = pd.concat(frames, ignore_index=True)
        dates = pd.to_datetime(raw["GAME_DATE"], errors="coerce")
        pts, reb, ast = _num(raw, "PTS"), _num(raw, "REB"), _num(raw, "AST")
        out = pd.DataFrame(
            {
                "player": full_name,
                "season": raw["season_label"],
                "game_date": dates,
                "label": dates.dt.strftime("%m/%d"),
                "opponent": raw["MATCHUP"].astype(str).str.split().str[-1],
                "points": pts,
                "rebounds": reb,
                "assists": ast,
                "threes": _num(raw, "FG3M"),
                "pts_reb_ast": pts + reb + ast,
                "steals": _num(raw, "STL"),
                "blocks": _num(raw, "BLK"),
                "minutes": _num(raw, "MIN"),
            }
        )
        out = out.sort_values("game_date").reset_index(drop=True)

        cur, prev = nba_season_str(start), nba_season_str(start - 1)
        n_cur = int((out["season"] == cur).sum())
        note: Optional[str] = None
        if n_cur == 0:
            note = f"No {cur} games yet; using {prev} logs."
        elif n_cur < 10:
            note = f"Only {n_cur} games in {cur}; {prev} games included to fill the sample."
        return out, note
    except Exception as exc:
        return pd.DataFrame(), f"NBA fetcher error: {exc}"


# --------------------------------------------------------------------------- #
# MLB
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=128)
def _mlb_player(name: str) -> Optional[Tuple[int, str]]:
    """Look up an MLB player id and full name, preferring active exact matches."""
    resp = requests.get(
        f"{MLB_BASE}/people/search",
        params={"names": name, "sportIds": 1},
        headers=HEADERS,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    people: List[Dict[str, Any]] = resp.json().get("people", [])
    if not people:
        return None
    query = name.strip().lower()
    people.sort(
        key=lambda p: (not p.get("active", False), str(p.get("fullName", "")).lower() != query)
    )
    best = people[0]
    return int(best["id"]), str(best.get("fullName", name))


def _mlb_game_log(pid: int, season: int, group: str) -> pd.DataFrame:
    """Download one season of MLB game-log splits (group: 'pitching' or 'hitting')."""
    last_err: Optional[Exception] = None
    for extra in ({"gameType": "R"}, {}):
        try:
            resp = requests.get(
                f"{MLB_BASE}/people/{pid}/stats",
                params={"stats": "gameLog", "group": group, "season": season, **extra},
                headers=HEADERS,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            blocks = resp.json().get("stats") or [{}]
            splits = blocks[0].get("splits", [])
            rows = [
                {
                    "game_date": s.get("date"),
                    "opponent": (s.get("opponent") or {}).get("name"),
                    "season": season,
                    **(s.get("stat") or {}),
                }
                for s in splits
            ]
            return pd.DataFrame(rows)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"MLB API error: {last_err}")


def _innings_to_outs(ip: pd.Series) -> pd.Series:
    """Convert MLB innings strings ('6.1' = 6 and 1/3) to outs recorded (vectorized)."""
    parts = ip.astype(str).str.split(".", n=1, expand=True)
    whole = pd.to_numeric(parts[0], errors="coerce")
    frac = pd.to_numeric(parts[1], errors="coerce").fillna(0) if 1 in parts.columns else 0.0
    return whole * 3 + frac


def fetch_mlb_player_logs(
    player_name: str, group: str, target_season: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Return (game logs, note/error) for a pitcher or hitter; pulls two seasons."""
    try:
        season = target_season or mlb_season()
        found = _mlb_player(player_name)
        if found is None:
            return pd.DataFrame(), f"MLB player '{player_name}' not found."
        pid, full_name = found

        frames: List[pd.DataFrame] = []
        for s in (season - 1, season):
            df = _mlb_game_log(pid, s, group)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame(), (
                f"No MLB {group} logs for {full_name} in {season - 1}-{season}. "
                f"Check that the stat type matches the player (pitcher vs. batter)."
            )

        raw = pd.concat(frames, ignore_index=True)
        dates = pd.to_datetime(raw["game_date"], errors="coerce")
        out = pd.DataFrame(
            {
                "player": full_name,
                "season": raw["season"],
                "game_date": dates,
                "label": dates.dt.strftime("%m/%d"),
                "opponent": raw["opponent"],
            }
        )
        note: Optional[str] = None

        if group == "pitching":
            ip = raw["inningsPitched"] if "inningsPitched" in raw.columns else pd.Series(np.nan, index=raw.index)
            out["p_strikeouts"] = _num(raw, "strikeOuts")
            out["p_outs"] = _innings_to_outs(ip)
            out["p_earned_runs"] = _num(raw, "earnedRuns")
            out["p_hits_allowed"] = _num(raw, "hits")
            out["p_walks_allowed"] = _num(raw, "baseOnBalls")
            out["games_started"] = _num(raw, "gamesStarted")
            starts = out[out["games_started"] > 0]
            if len(starts) >= 3:
                out = starts
            else:
                note = "Fewer than 3 starts found; using all appearances (relief outings included)."
        else:
            hits, runs, rbi = _num(raw, "hits"), _num(raw, "runs"), _num(raw, "rbi")
            out["b_hits"] = hits
            out["b_total_bases"] = _num(raw, "totalBases")
            out["b_hrr"] = hits + runs + rbi
            out["b_home_runs"] = _num(raw, "homeRuns")
            out["b_rbis"] = rbi
            out["b_runs"] = runs
            out["b_stolen_bases"] = _num(raw, "stolenBases")
            out["b_strikeouts"] = _num(raw, "strikeOuts")

        out = out.sort_values("game_date").reset_index(drop=True)
        return out, note
    except Exception as exc:
        return pd.DataFrame(), f"MLB fetcher error: {exc}"

"""NBA and MLB game-log fetchers that return a common schema for the props app."""
from __future__ import annotations

import os
import time
import unicodedata
from datetime import date
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

MLB_BASE: str = "https://statsapi.mlb.com/api/v1"
TIMEOUT: int = 30
HEADERS: Dict[str, str] = {"User-Agent": "Mozilla/5.0"}
SMALL_SAMPLE_GAMES: int = 5  # below this, attach an informational small-sample note


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
# NBA: BALLDONTLIE when a key is set (works from cloud servers), nba_api as the fallback
# --------------------------------------------------------------------------- #
BDL_BASE: str = "https://api.balldontlie.io/v1"


class BdlError(Exception):
    """BALLDONTLIE failure; status is the HTTP code (0 for network problems)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _bdl_key() -> str:
    """Return the BALLDONTLIE key from the environment (Streamlit Secrets are exposed there)."""
    return os.environ.get("BALLDONTLIE_API_KEY", "").strip()


def _bdl_get(path: str, params: Dict[str, Any], key: str) -> Dict[str, Any]:
    """GET a BALLDONTLIE endpoint. The key travels in a header and never appears in messages."""
    for attempt in (1, 2):
        try:
            resp = requests.get(
                f"{BDL_BASE}{path}", params=params, headers={"Authorization": key}, timeout=TIMEOUT
            )
        except requests.RequestException as exc:
            raise BdlError(0, f"network error: {str(exc).replace(key, '***')}") from None
        if resp.status_code == 429 and attempt == 1:
            try:
                wait = min(float(resp.headers.get("Retry-After", 5)), 10.0)
            except (TypeError, ValueError):
                wait = 5.0
            time.sleep(wait)
            continue
        if resp.status_code == 401:
            raise BdlError(
                401,
                "key rejected, or your plan does not include game stats "
                "(they need the paid ALL-STAR tier or higher)",
            )
        if resp.status_code == 429:
            raise BdlError(429, "rate limit reached (the free tier allows 5 requests per minute)")
        if resp.status_code >= 400:
            raise BdlError(resp.status_code, f"HTTP {resp.status_code}")
        return resp.json()
    raise BdlError(429, "rate limit reached")


def _norm(text: str) -> str:
    """Lowercase and strip accents so 'Doncic' matches 'Dončić'."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return plain.casefold().strip()


@lru_cache(maxsize=256)
def _bdl_player(name: str, key: str) -> Optional[Tuple[int, str]]:
    """Look up a BALLDONTLIE player id and full name (exact name preferred, accents ignored)."""
    tokens = _norm(name).split()
    if not tokens:
        return None
    term = max(tokens, key=len)
    candidates: List[Dict[str, Any]] = []
    cursor: Optional[int] = None
    for _ in range(3):
        params: Dict[str, Any] = {"search": term, "per_page": 100}
        if cursor is not None:
            params["cursor"] = cursor
        data = _bdl_get("/players", params, key)
        candidates.extend(data.get("data", []))
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            break

    def full(p: Dict[str, Any]) -> str:
        return f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()

    wanted = " ".join(tokens)
    picks = [p for p in candidates if _norm(full(p)) == wanted] or [
        p for p in candidates if all(t in _norm(full(p)) for t in tokens)
    ]
    if not picks:
        return None
    picks.sort(key=lambda p: p.get("team") is None)  # prefer players currently on a team
    return int(picks[0]["id"]), full(picks[0])


@lru_cache(maxsize=4)
def _bdl_teams(key: str) -> Dict[int, str]:
    """Map BALLDONTLIE team ids to abbreviations."""
    data = _bdl_get("/teams", {}, key)
    return {int(t["id"]): str(t.get("abbreviation", "")) for t in data.get("data", [])}


def _bdl_stats(player_id: int, season: int, key: str) -> List[Dict[str, Any]]:
    """Fetch one regular season of per-game stat lines (season = start year, paginated)."""
    rows: List[Dict[str, Any]] = []
    cursor: Optional[int] = None
    for _ in range(5):
        params: Dict[str, Any] = {
            "player_ids[]": [player_id],
            "seasons[]": [season],
            "season_type": "regular",
            "per_page": 100,
        }
        if cursor is not None:
            params["cursor"] = cursor
        data = _bdl_get("/stats", params, key)
        rows.extend(data.get("data", []))
        cursor = (data.get("meta") or {}).get("next_cursor")
        if not cursor:
            break
    return rows


def _nba_frame(
    full_name: str,
    season: pd.Series,
    dates: pd.Series,
    opponent: pd.Series,
    box: pd.DataFrame,
    source: str,
) -> pd.DataFrame:
    """Build the common NBA schema from per-game columns pts, reb, ast, fg3m, stl, blk, min."""
    pts, reb, ast = box["pts"], box["reb"], box["ast"]
    out = pd.DataFrame(
        {
            "player": full_name,
            "season": season.to_numpy(),
            "game_date": dates.to_numpy(),
            "label": dates.dt.strftime("%m/%d").to_numpy(),
            "opponent": opponent.to_numpy(),
            "points": pts.to_numpy(),
            "rebounds": reb.to_numpy(),
            "assists": ast.to_numpy(),
            "threes": box["fg3m"].to_numpy(),
            "pts_reb_ast": (pts + reb + ast).to_numpy(),
            "steals": box["stl"].to_numpy(),
            "blocks": box["blk"].to_numpy(),
            "minutes": box["min"].to_numpy(),
            "source": source,
        }
    )
    return out.sort_values("game_date").reset_index(drop=True)


def _nba_logs_from_bdl(
    player_name: str, start: int, include_prior: bool, key: str
) -> Tuple[Optional[pd.DataFrame], str, Optional[str]]:
    """Return (logs, full_name, error). None = failure; an empty frame = no completed games."""
    try:
        found = _bdl_player(player_name, key)
        if found is None:
            return None, player_name, f"player '{player_name}' not found"
        pid, full_name = found
        teams = _bdl_teams(key)
        rows: List[Dict[str, Any]] = []
        for s in ((start - 1, start) if include_prior else (start,)):
            rows.extend(_bdl_stats(pid, s, key))
        records = [
            {
                "date": (r.get("game") or {}).get("date"),
                "season": (r.get("game") or {}).get("season"),
                "state": (r.get("game") or {}).get("status_state") or (r.get("game") or {}).get("status"),
                "team_id": (r.get("team") or {}).get("id"),
                "home_id": (r.get("game") or {}).get("home_team_id"),
                "visitor_id": (r.get("game") or {}).get("visitor_team_id"),
                "min": r.get("min"),
                "pts": r.get("pts"),
                "reb": r.get("reb"),
                "ast": r.get("ast"),
                "fg3m": r.get("fg3m"),
                "stl": r.get("stl"),
                "blk": r.get("blk"),
            }
            for r in rows
        ]
        df = pd.DataFrame(records)
        if df.empty:
            return pd.DataFrame(), full_name, None

        minutes = pd.to_numeric(df["min"].astype(str).str.split(":").str[0], errors="coerce")
        final = df["state"].astype(str).str.lower().eq("final")  # skip live (partial) games
        df = df[final & (minutes > 0)].reset_index(drop=True)  # skip DNPs
        if df.empty:
            return pd.DataFrame(), full_name, None

        minutes = pd.to_numeric(df["min"].astype(str).str.split(":").str[0], errors="coerce")
        box = pd.DataFrame(
            {c: pd.to_numeric(df[c], errors="coerce") for c in ("pts", "reb", "ast", "fg3m", "stl", "blk")}
        )
        box["min"] = minutes
        opp_id = pd.Series(np.where(df["team_id"] == df["home_id"], df["visitor_id"], df["home_id"]))
        season_year = pd.to_numeric(df["season"], errors="coerce").fillna(start).astype(int)
        out = _nba_frame(
            full_name,
            season_year.map(nba_season_str),
            pd.to_datetime(df["date"], errors="coerce"),
            opp_id.map(teams).fillna(""),
            box,
            "BALLDONTLIE",
        )
        return out, full_name, None
    except BdlError as exc:
        return None, player_name, str(exc)
    except Exception as exc:
        return None, player_name, f"unexpected error: {exc}"


@lru_cache(maxsize=128)
def _nba_player(name: str) -> Optional[Tuple[int, str]]:
    """Look up an NBA player id and full name via nba_api, preferring active exact matches."""
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


def _nba_logs_from_nba_api(
    player_name: str, start: int, include_prior: bool
) -> Tuple[Optional[pd.DataFrame], str, Optional[str]]:
    """Return (logs, full_name, error). None = failure; an empty frame = no games yet."""
    try:
        found = _nba_player(player_name)
        if found is None:
            return None, player_name, f"player '{player_name}' not found"
        pid, full_name = found

        frames: List[pd.DataFrame] = []
        last_err: Optional[Exception] = None
        any_call_ok = False
        for s in ((start - 1, start) if include_prior else (start,)):
            try:
                frame = _nba_season_logs(pid, nba_season_str(s))
                any_call_ok = True
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                last_err = exc
        if not frames:
            if not any_call_ok and last_err is not None:
                return None, full_name, f"request failed: {last_err}"
            return pd.DataFrame(), full_name, None

        raw = pd.concat(frames, ignore_index=True)
        dates = pd.to_datetime(raw["GAME_DATE"], format="%b %d, %Y", errors="coerce")
        if dates.isna().any():  # fall back to flexible parsing for any other date format
            dates = dates.fillna(pd.to_datetime(raw["GAME_DATE"], errors="coerce"))
        box = pd.DataFrame(
            {
                "pts": _num(raw, "PTS"),
                "reb": _num(raw, "REB"),
                "ast": _num(raw, "AST"),
                "fg3m": _num(raw, "FG3M"),
                "stl": _num(raw, "STL"),
                "blk": _num(raw, "BLK"),
                "min": _num(raw, "MIN"),
            }
        )
        out = _nba_frame(
            full_name,
            raw["season_label"],
            dates,
            raw["MATCHUP"].astype(str).str.split().str[-1],
            box,
            "nba_api",
        )
        return out, full_name, None
    except Exception as exc:
        return None, player_name, str(exc)


def fetch_nba_player_logs(
    player_name: str, target_season: Optional[int] = None, include_prior: bool = False
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Return (game logs, message) using this season's games only (any count, minimum 1).

    Source order: BALLDONTLIE when BALLDONTLIE_API_KEY is set, then nba_api. A source is
    skipped only when it errors (a free BALLDONTLIE plan has no game stats, so it falls back).
    Contract: an EMPTY DataFrame means failure and the message is the error text; a
    NON-EMPTY DataFrame means success and the message is only an informational note.
    Set include_prior=True to also pull last season.
    """
    try:
        start = target_season or nba_season_start()
        cur, prev = nba_season_str(start), nba_season_str(start - 1)
        key = _bdl_key()

        sources: List[Tuple[str, Any]] = []
        if key:
            sources.append(("BALLDONTLIE", lambda: _nba_logs_from_bdl(player_name, start, include_prior, key)))
        sources.append(("nba_api", lambda: _nba_logs_from_nba_api(player_name, start, include_prior)))

        problems: List[str] = []
        out: Optional[pd.DataFrame] = None
        full_name, used = player_name, ""
        for label, run in sources:
            frame, name, err = run()
            if frame is not None:
                out, full_name, used = frame, name, label
                break
            problems.append(f"{label}: {err}")
        if out is None:
            return pd.DataFrame(), "NBA data unavailable. " + " | ".join(problems)
        if out.empty:
            return pd.DataFrame(), (
                f"No {cur} regular-season games found for {full_name} yet. "
                "They may not have played this season."
            )

        notes: List[str] = []
        if problems:
            notes.append(f"{problems[0]}; used {used} instead.")
        n_cur = int((out["season"] == cur).sum())
        if n_cur == 0:
            notes.append(f"No {cur} games yet; showing {prev} logs.")
        elif include_prior and n_cur < 10:
            notes.append(f"Only {n_cur} games in {cur}; {prev} games included to fill the sample.")
        elif n_cur < SMALL_SAMPLE_GAMES:
            notes.append(
                f"Small sample: {n_cur} game(s) so far in {cur}. "
                "The projection leans heavily on those games."
            )
        return out, " ".join(notes) or None
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
                "source": "MLB Stats API",
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

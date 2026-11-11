"""
NFL Vegas environment: the market's implied team points as a projection multiplier, with a no-look-ahead backtest.

Math
----
nflverse schedules carry the closing total T and spread s (positive = home team favored) for every game, so
    implied_home = T/2 + s/2          implied_away = T/2 - s/2
Baseline for a team before a game = its average points in EARLIER games this season, shrunk to last season's average:
    base = (sum_prev + n0 * prior) / (n_prev + n0)         n0 = BASE_PRIOR_GAMES
    ratio = implied / base                                  (1.10 = the market expects 10% more points than usual)
    multiplier = clip(1 + k_stat * (ratio - 1), MULT_LO, MULT_HI)
Backtest (clustered by player), with the opponent-vs-position index in the same regression so the weight is what Vegas adds
ON TOP of the existing matchup factor:
    stat / player's season mean = a + k_m * (matchup_index - 1) + k_v * (ratio - 1) + e
k_v is then shrunk toward the default by its standard error (nfl_matchups.calibrated_k) before use.
Live games use ESPN's current line when it shows a total and a favorite, else the nflverse line for that week.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

import vegas_env as ve
from nfl_matchups import BT_MIN_MEAN, STAT_POS_GROUPS, _defense_games, _expanding_index, _ols_cluster, calibrated_k

BASE_PRIOR_GAMES = 4.0
MULT_LO, MULT_HI = 0.92, 1.08
# How much of a change in implied team points reaches each stat. Assumptions until the backtest calibrates them.
DEFAULT_K: Dict[str, float] = {
    "PASSING YARDS": 0.30, "PASS COMPLETIONS": 0.20, "PASS ATTEMPTS": 0.10, "PASSING TDS": 0.50, "RUSHING YARDS": 0.25,
    "RUSHING ATTEMPTS": 0.15, "RECEIVING YARDS": 0.30, "RECEPTIONS": 0.20, "INTERCEPTIONS": 0.0,
}
_CACHE: Dict[Tuple[Any, ...], Tuple[float, Any]] = {}


def _ttl(key: Tuple[Any, ...], ttl: float, fn: Callable[[], Any]) -> Any:
    """Tiny TTL cache (successful calls only)."""
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (time.time(), val)
    return val


def normalize_lines(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw nflverse schedule -> SEASON, WEEK, HOME, AWAY, HOME_PTS, AWAY_PTS, SPREAD (home favored > 0), TOTAL (regular season)."""
    import sports_data as sd
    norm = getattr(sd, "norm_team", lambda s: s.fillna("").astype(str).str.upper())
    d = raw.copy()
    if "game_type" in d.columns:
        d = d[d["game_type"].astype(str).str.upper() == "REG"]
    num = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)
    return pd.DataFrame({"SEASON": num("season"), "WEEK": num("week"), "HOME": norm(d["home_team"]), "AWAY": norm(d["away_team"]),
                         "HOME_PTS": num("home_score"), "AWAY_PTS": num("away_score"), "SPREAD": num("spread_line"),
                         "TOTAL": num("total_line")}).dropna(subset=["SEASON", "WEEK"]).reset_index(drop=True)


def load_lines(season: int, back: int = 1) -> pd.DataFrame:
    """Schedules with closing lines for `season` and `back` earlier seasons (cached 6 h). Raises if nflverse is unavailable."""
    def fetch() -> pd.DataFrame:
        import nflreadpy as nfl
        return normalize_lines(nfl.load_schedules(seasons=list(range(int(season) - int(back), int(season) + 1))).to_pandas())
    return _ttl(("lines", int(season), int(back)), 21600.0, fetch)


def team_games(lines: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: SEASON, WEEK, TEAM, OPP, HOME, POINTS, IMPLIED, TOTAL (vectorised)."""
    half_t, half_s = lines["TOTAL"] / 2.0, lines["SPREAD"] / 2.0
    home = pd.DataFrame({"SEASON": lines["SEASON"], "WEEK": lines["WEEK"], "TEAM": lines["HOME"], "OPP": lines["AWAY"], "HOME": True,
                         "POINTS": lines["HOME_PTS"], "IMPLIED": half_t + half_s, "TOTAL": lines["TOTAL"]})
    away = pd.DataFrame({"SEASON": lines["SEASON"], "WEEK": lines["WEEK"], "TEAM": lines["AWAY"], "OPP": lines["HOME"], "HOME": False,
                         "POINTS": lines["AWAY_PTS"], "IMPLIED": half_t - half_s, "TOTAL": lines["TOTAL"]})
    return pd.concat([home, away], ignore_index=True).sort_values(["TEAM", "SEASON", "WEEK"]).reset_index(drop=True)


def with_baseline(tg: pd.DataFrame, n0: float = BASE_PRIOR_GAMES) -> pd.DataFrame:
    """Adds N_PREV, BASE (points baseline from EARLIER games + last season) and RATIO = IMPLIED / BASE. No look-ahead."""
    g = tg.sort_values(["TEAM", "SEASON", "WEEK"]).reset_index(drop=True)
    played = g["POINTS"].notna()
    pts = g["POINTS"].fillna(0.0)
    grp = [g["TEAM"], g["SEASON"]]
    g["N_PREV"] = played.astype(float).groupby(grp).cumsum() - played.astype(float)
    cum = pts.groupby(grp).cumsum() - pts
    prior = g[played].groupby(["TEAM", "SEASON"])["POINTS"].mean().rename("PRI").reset_index()
    prior["SEASON"] = prior["SEASON"] + 1
    g = g.merge(prior, on=["TEAM", "SEASON"], how="left")
    with_prior = (cum + n0 * g["PRI"]) / (g["N_PREV"] + n0)
    no_prior = cum / g["N_PREV"].where(g["N_PREV"] > 0)
    g["BASE"] = with_prior.where(g["PRI"].notna(), no_prior)
    g["RATIO"] = g["IMPLIED"] / g["BASE"].where(g["BASE"] > 0)
    return g


def fetch_live_lines(team: str, opponent: str, gameday: Optional[str]) -> Optional[Dict[str, Any]]:
    """ESPN's current total / favorite for the game (cached 5 min); None if ESPN has no such game."""
    def fetch() -> Optional[Dict[str, Any]]:
        import requests
        from nfl_six_point import ESPN_SCOREBOARD, canon
        if gameday:
            day = pd.Timestamp(gameday)
            start, end = day - pd.Timedelta(days=1), day + pd.Timedelta(days=1)
        else:
            start = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
            end = start + pd.Timedelta(days=14)
        resp = requests.get(ESPN_SCOREBOARD, params={"dates": f"{start:%Y%m%d}-{end:%Y%m%d}", "limit": 100}, timeout=8)
        resp.raise_for_status()
        return ve.find_espn_lines(resp.json(), [team, opponent], canon)
    return _ttl(("live", str(team), str(opponent), str(gameday or "")), 300.0, fetch)


def environment(team: str, opponent: Optional[str], stat: str, season: int, week: Optional[int], lines: pd.DataFrame,
                live: Optional[Mapping[str, Any]] = None, k_map: Optional[Mapping[str, float]] = None) -> Optional[Dict[str, Any]]:
    """Multiplier for one player's game. Uses `live` (ESPN) when it has a total and a favorite, else the nflverse line for `week`.
    None when the game, a line, or a points baseline is missing."""
    if not opponent or lines is None or lines.empty:
        return None
    tg = with_baseline(team_games(lines))
    mine = tg[(tg["TEAM"] == str(team).upper()) & (tg["SEASON"] == season) & (tg["OPP"] == str(opponent).upper())]
    if week is not None and pd.notna(week):
        mine = mine[mine["WEEK"] == int(week)] if (mine["WEEK"] == int(week)).any() else mine
    mine = mine[mine["POINTS"].isna()] if mine["POINTS"].isna().any() else mine
    if mine.empty:
        return None
    row = mine.sort_values("WEEK").iloc[0]
    base = row["BASE"]
    implied, total, source = row["IMPLIED"], row["TOTAL"], "nflverse line"
    if live and live.get("total") and live.get("home_spread") is not None:
        team_spread = live["home_spread"] if bool(row["HOME"]) else -live["home_spread"]
        implied, total, source = ve.implied_team_total(live["total"], team_spread), float(live["total"]), "ESPN current line"
    if not np.isfinite(base) or base <= 0 or implied is None or not np.isfinite(implied):
        return None
    k = float((k_map or {}).get(stat, DEFAULT_K.get(stat, 0.0)))
    ratio = float(implied / base)
    mult = float(np.clip(1.0 + k * (ratio - 1.0), MULT_LO, MULT_HI))
    return {"implied": float(implied), "total": float(total), "baseline": float(base), "ratio": ratio, "k": k, "mult": mult,
            "source": source, "calibrated": bool(k_map and stat in k_map),
            "note": f"Vegas ({source}): {team} implied {implied:.1f} pts vs {base:.1f} usual (x{ratio:.3f}) -> projection x{mult:.3f}"}


def vegas_layer(team: str, opponent: Optional[str], stat: str, season: int, week: Optional[int], gameday: Optional[str],
                k_map: Optional[Mapping[str, float]] = None) -> Optional[Dict[str, Any]]:
    """What the engine calls: load lines, try ESPN's live line, return the environment dict (or None)."""
    if not opponent:
        return None
    lines = load_lines(int(season), 1)
    try:
        live = fetch_live_lines(str(team), str(opponent), gameday)
    except Exception:
        live = None
    return environment(team, opponent, stat, int(season), week, lines, live, k_map)


def backtest(weekly: pd.DataFrame, lines: pd.DataFrame, stat: str, first_season: int) -> Optional[Dict[str, Any]]:
    """k_v for one stat from closing lines, controlling for the matchup index. None if the sample is too small."""
    tg = with_baseline(team_games(lines)).dropna(subset=["RATIO"])
    tg = tg[tg["N_PREV"] + BASE_PRIOR_GAMES * tg["PRI"].notna() >= 3][["TEAM", "SEASON", "WEEK", "RATIO"]]
    rows: List[pd.DataFrame] = []
    for pos in STAT_POS_GROUPS[stat]:
        idx = _expanding_index(_defense_games(weekly, stat, pos), 6.0)
        p = weekly[(weekly["POS"] == pos) & (weekly["SEASON"] >= first_season)][["PID", "SEASON", "WEEK", "TEAM", "OPP", stat]].copy()
        grp = p.groupby(["PID", "SEASON"])[stat]
        p["PMEAN"], p["CNT"] = grp.transform("mean"), grp.transform("size")
        p = p[(p["CNT"] >= 8) & (p["PMEAN"] >= BT_MIN_MEAN[stat])]
        p = p.merge(idx, on=["OPP", "SEASON", "WEEK"], how="inner").merge(tg, on=["TEAM", "SEASON", "WEEK"], how="inner")
        p = p.dropna(subset=["INDEX", "RATIO"])
        p["R"] = p[stat] / p["PMEAN"]
        rows.append(p[["PID", "R", "INDEX", "RATIO"]])
    d = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(d) < 200 or d["PID"].nunique() < 15:
        return None
    X = np.column_stack([np.ones(len(d)), d["INDEX"].to_numpy(float) - 1.0, d["RATIO"].to_numpy(float) - 1.0])
    beta, se = _ols_cluster(X, d["R"].to_numpy(float), d["PID"].to_numpy())
    return {"STAT": stat, "K_V": float(beta[2]), "SE_V": float(se[2]), "K_MATCHUP": float(beta[1]), "N": int(len(d)),
            "PLAYERS": int(d["PID"].nunique()), "SD_RATIO": float(d["RATIO"].std())}


def run_backtests(weekly: pd.DataFrame, lines: pd.DataFrame, first_season: int) -> pd.DataFrame:
    """Backtest every stat; adds the default, the shrunk weight K_EFF and a significance flag."""
    out: List[Dict[str, Any]] = []
    for stat in DEFAULT_K:
        if stat not in weekly.columns or stat not in STAT_POS_GROUPS:
            continue
        r = backtest(weekly, lines, stat, first_season)
        if r is None:
            continue
        r["K_DEFAULT"] = DEFAULT_K[stat]
        r["K_EFF"] = calibrated_k(r["K_V"], r["SE_V"], DEFAULT_K[stat])
        r["SIGNIFICANT"] = "yes" if abs(r["K_V"]) >= 2 * r["SE_V"] else "no"
        out.append(r)
    return pd.DataFrame(out)

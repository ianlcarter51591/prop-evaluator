"""NFL last-season weight: per-stat calibration from nflverse weekly logs, and the role-change override."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

N_PRIOR = 4.0                    # fallback last-season weight (pseudo-games) for stats without their own value
# Per-stat weights. All 4.0 until calibrated in the Matchup Backtest tab; "Apply" there saves its own values to the Gist,
# which take priority over these. Paste the printed dict here to make them the built-in defaults.
N_PRIOR_BY_STAT: Dict[str, float] = {}
PRIOR_GRID: List[float] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
EARLY_WEEKS = 10                 # score predictions over each player's first 10 games of a season
WINDOW, DECAY = 8, 0.85          # match the engine's EWMA
MIN_PRIOR_GAMES = 4              # a player-season needs 4+ games last season to be scored
CALIBRATE_STATS: Tuple[str, ...] = ("PASSING YARDS", "PASS COMPLETIONS", "PASS ATTEMPTS", "PASSING TDS", "RUSHING YARDS",
                                    "RUSHING ATTEMPTS", "RECEIVING YARDS", "RECEPTIONS", "INTERCEPTIONS",
                                    "KICKING POINTS", "FIELD GOALS MADE", "ANYTIME TOUCHDOWN SCORER")
# Role change: the opportunity stat behind each prop, and how far it must move (absolute floor, share) to count.
VOLUME_STAT: Dict[str, str] = {
    "PASSING YARDS": "PASS ATTEMPTS", "PASS COMPLETIONS": "PASS ATTEMPTS", "PASS ATTEMPTS": "PASS ATTEMPTS",
    "PASSING TDS": "PASS ATTEMPTS", "INTERCEPTIONS": "PASS ATTEMPTS", "RUSHING YARDS": "RUSHING ATTEMPTS",
    "RUSHING ATTEMPTS": "RUSHING ATTEMPTS", "RECEIVING YARDS": "TARGETS", "RECEPTIONS": "TARGETS",
    "ANYTIME TOUCHDOWN SCORER": "TOUCHES",
}
VOLUME_FLOOR: Dict[str, float] = {"PASS ATTEMPTS": 6.0, "RUSHING ATTEMPTS": 4.0, "TARGETS": 2.0, "TOUCHES": 4.0}
ROLE_SHARE = 0.30                # volume change >= max(floor, 30% of last season) = new role
ROLE_RECENT_N, ROLE_MIN_GAMES = 4, 3
ROLE_PRIOR_CUT = 0.25            # on a role change last season keeps a quarter of its weight


def prior_weight(stat: str, cfg: Optional[Mapping[str, Any]] = None) -> float:
    """Last-season weight for this stat: applied/saved per-stat value > built-in N_PRIOR_BY_STAT > N_PRIOR."""
    cfg = cfg or {}
    if not cfg.get("per_stat_prior", True):
        return float(N_PRIOR)
    applied = cfg.get("n0_by_stat") or {}
    return float(applied.get(stat, N_PRIOR_BY_STAT.get(stat, N_PRIOR)))


def role_shift(logs: pd.DataFrame, stat: str) -> Optional[Dict[str, Any]]:
    """
    New role if either:
        team changed  : his latest team this season != his last team last season
        volume moved  : |mean(last 4 games of the volume stat) - mean(last season)| >= max(floor, 30% of last season)  (3+ games)
    Returns {"why", "recent", "before", "volume"} or None."""
    if logs is None or logs.empty or "SEASON_TAG" not in logs.columns:
        return None
    cur, pri = logs[logs["SEASON_TAG"] == "current"], logs[logs["SEASON_TAG"] == "prior"]
    if cur.empty or pri.empty:
        return None
    if "TEAM" in logs.columns and str(cur["TEAM"].iloc[-1]) and str(cur["TEAM"].iloc[-1]) != str(pri["TEAM"].iloc[-1]):
        return {"why": "team", "new_team": str(cur["TEAM"].iloc[-1]), "old_team": str(pri["TEAM"].iloc[-1])}
    vol = VOLUME_STAT.get(stat)
    if vol is None or vol not in logs.columns or len(cur) < ROLE_MIN_GAMES or len(pri) < ROLE_MIN_GAMES:
        return None
    recent = float(pd.to_numeric(cur[vol], errors="coerce").tail(ROLE_RECENT_N).mean())
    before = float(pd.to_numeric(pri[vol], errors="coerce").mean())
    if not np.isfinite(recent) or not np.isfinite(before):
        return None
    if abs(recent - before) < max(VOLUME_FLOOR.get(vol, 2.0), ROLE_SHARE * before):
        return None
    return {"why": "volume", "volume": vol, "recent": recent, "before": before}


def role_note(player: str, shift: Optional[Mapping[str, Any]], manual: bool, n0: float, used: float) -> str:
    """Plain-English warning for the override ('' when it didn't fire)."""
    if not (shift or manual):
        return ""
    if shift and shift.get("why") == "team":
        why = f"{player} changed teams ({shift['old_team']} → {shift['new_team']})"
    elif shift:
        why = (f"{player} is averaging {shift['recent']:.1f} {shift['volume'].lower()} over his last games vs "
               f"{shift['before']:.1f} last season")
    else:
        why = f"{player} is on your 'Role changed' list"
    return f"🔄 New role: {why}, so last season counts a quarter as much ({n0:g} → {used:g} pseudo-games)."


def apply_role(n0: float, shift: Optional[Mapping[str, Any]], manual: bool) -> float:
    """Last-season weight after the override."""
    return float(n0) * ROLE_PRIOR_CUT if (shift or manual) else float(n0)


def _season_arrays(weekly: pd.DataFrame, stat: str, first_season: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(X, PRI): X = (P, EARLY_WEEKS) first games of each player-season (NaN-padded, played games only), PRI = last-season mean."""
    d = weekly[["PID", "SEASON", "WEEK", stat]].copy()
    d[stat] = pd.to_numeric(d[stat], errors="coerce")
    d = d.dropna(subset=[stat]).sort_values(["PID", "SEASON", "WEEK"])
    g = d.groupby(["PID", "SEASON"])[stat]
    prior = pd.DataFrame({"PRI": g.mean(), "NP": g.size()}).reset_index()
    prior["SEASON"] += 1
    prior = prior[(prior["NP"] >= MIN_PRIOR_GAMES) & (prior["PRI"] > 0)]
    cur = d[d["SEASON"] >= first_season].copy()
    cur["G"] = cur.groupby(["PID", "SEASON"]).cumcount()
    cur = cur[cur["G"] < EARLY_WEEKS].merge(prior, on=["PID", "SEASON"], how="inner")
    if cur.empty:
        return None
    keys = cur[["PID", "SEASON"]].drop_duplicates().reset_index(drop=True)
    keys["P"] = np.arange(len(keys))
    cur = cur.merge(keys, on=["PID", "SEASON"])
    X = np.full((len(keys), EARLY_WEEKS), np.nan)
    X[cur["P"].to_numpy(), cur["G"].to_numpy()] = cur[stat].to_numpy(float)
    pri = cur.drop_duplicates("P").sort_values("P")["PRI"].to_numpy(float)
    return X, pri


def loss_curve(X: np.ndarray, pri: np.ndarray, grid: Sequence[float]) -> Tuple[np.ndarray, int]:
    """
    For each n0 on the grid, mean over scored games of ((actual - pred) / last-season mean)^2 with
        pred_g = (n*EWMA(previous min(g, 8) games) + n0*last-season mean) / (n + n0),  n = min(g, 8);  g = 0 -> last-season mean.
    """
    n0 = np.asarray(grid, float)[:, None]
    sse, cnt = np.zeros(len(n0)), 0
    for g in range(X.shape[1]):
        ok = ~np.isnan(X[:, g])
        if not ok.any():
            continue
        n = min(g, WINDOW)
        if n:
            w = DECAY ** np.arange(n)[::-1]
            ewma = (X[ok, g - n:g] * w).sum(1) / w.sum()
            pred = (n * ewma[None, :] + n0 * pri[ok][None, :]) / (n + n0)
        else:
            pred = np.broadcast_to(pri[ok][None, :], (len(n0), int(ok.sum())))
        sse += (((X[ok, g][None, :] - pred) / pri[ok][None, :]) ** 2).sum(1)
        cnt += int(ok.sum())
    return (sse / max(cnt, 1)), cnt


def estimate(weekly: pd.DataFrame, first_season: int, grid: Optional[Sequence[float]] = None,
             current: Optional[Mapping[str, float]] = None) -> pd.DataFrame:
    """Best last-season weight per stat (same columns as the NBA app's calibration table)."""
    grid = list(PRIOR_GRID if grid is None else grid)
    out: List[Dict[str, Any]] = []
    for stat in CALIBRATE_STATS:
        if stat not in weekly.columns:
            continue
        arrays = _season_arrays(weekly, stat, first_season)
        if arrays is None:
            continue
        c, cnt = loss_curve(*arrays, grid)
        if cnt == 0:
            continue
        best = int(np.argmin(c))
        near = [grid[i] for i in range(len(grid)) if c[i] <= c[best] * 1.01]
        default = float((current or {}).get(stat, N_PRIOR_BY_STAT.get(stat, N_PRIOR)))
        base = float(np.interp(default, grid, c))
        out.append({"STAT": stat, "N0_BEST": float(grid[best]), "N0_RANGE": f"{min(near):g}-{max(near):g}",
                    "LOSS_BEST": float(c[best]), "LOSS_AT_CURRENT": base,
                    "GAIN_VS_CURRENT_%": (base - float(c[best])) / base * 100.0 if base > 0 else 0.0, "PLAYER_GAMES": cnt})
    return pd.DataFrame(out)

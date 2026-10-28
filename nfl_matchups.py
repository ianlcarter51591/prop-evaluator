"""Opponent-vs-position matchup tables, multipliers, and a no-look-ahead calibration backtest."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PRIOR_GAMES = 6.0            # pseudo-games of last-season weight in a defense's "allowed" average
MULT_LO, MULT_HI = 0.85, 1.15
K_TAU = 0.3                  # prior sd when shrinking a backtested k toward the default k
# How strongly the opponent index moves the projection: mult = 1 + k * (index - 1). Assumptions until backtested.
DEFAULT_K: Dict[str, float] = {
    "PASSING YARDS": 0.35, "PASS COMPLETIONS": 0.30, "PASS ATTEMPTS": 0.15, "PASSING TDS": 0.25,
    "RUSHING YARDS": 0.35, "RUSHING ATTEMPTS": 0.15, "RECEIVING YARDS": 0.35, "RECEPTIONS": 0.30, "INTERCEPTIONS": 0.20,
}
# Which offensive position groups a stat is evaluated against.
STAT_POS_GROUPS: Dict[str, Tuple[str, ...]] = {
    "PASSING YARDS": ("QB",), "PASS COMPLETIONS": ("QB",), "PASS ATTEMPTS": ("QB",), "PASSING TDS": ("QB",),
    "INTERCEPTIONS": ("QB",), "RUSHING YARDS": ("RB", "QB"), "RUSHING ATTEMPTS": ("RB", "QB"),
    "RECEIVING YARDS": ("WR", "TE", "RB"), "RECEPTIONS": ("WR", "TE", "RB"),
}
# Backtest only players with meaningful volume (per-game mean at least this much).
BT_MIN_MEAN: Dict[str, float] = {
    "PASSING YARDS": 120.0, "PASS COMPLETIONS": 12.0, "PASS ATTEMPTS": 20.0, "PASSING TDS": 0.6, "RUSHING YARDS": 20.0,
    "RUSHING ATTEMPTS": 6.0, "RECEIVING YARDS": 25.0, "RECEPTIONS": 2.5, "INTERCEPTIONS": 0.3,
}


def _defense_games(weekly: pd.DataFrame, stat: str, pos: str) -> pd.DataFrame:
    """One row per (SEASON, OPP, WEEK) defense-game; V = total `stat` allowed to position group `pos` (0 if none)."""
    games = weekly[["SEASON", "OPP", "WEEK"]].drop_duplicates()
    tot = (weekly[weekly["POS"] == pos].groupby(["SEASON", "OPP", "WEEK"], as_index=False)[stat].sum().rename(columns={stat: "V"}))
    g = games.merge(tot, on=["SEASON", "OPP", "WEEK"], how="left")
    g["V"] = g["V"].fillna(0.0)
    return g


def matchup_table(weekly: pd.DataFrame, stat: str, pos: str, season: int, n0: float = PRIOR_GAMES) -> pd.DataFrame:
    """
    Per-defense table indexed by team: GAMES, CUR (this season's avg allowed), EST (shrunk to last season),
    LEAGUE (mean EST), INDEX (EST / LEAGUE), RANK (1 = allows the most = softest matchup).
    EST = (n*CUR + n0*PRIOR) / (n + n0); falls back to whichever season exists.
    """
    g = _defense_games(weekly, stat, pos)
    cur = g[g["SEASON"] == season].groupby("OPP")["V"].agg(GAMES="size", CUR="mean")
    pri = g[g["SEASON"] == season - 1].groupby("OPP")["V"].mean().rename("PRIOR")
    t = cur.join(pri, how="outer")
    n = t["GAMES"].fillna(0.0)
    blended = (n * t["CUR"].fillna(0.0) + n0 * t["PRIOR"].fillna(0.0)) / (n + n0)
    t["EST"] = blended.where(t["PRIOR"].notna(), t["CUR"])
    t = t.dropna(subset=["EST"]).copy()
    t["GAMES"] = t["GAMES"].fillna(0).astype(int)
    t["LEAGUE"] = t["EST"].mean()
    t["INDEX"] = (t["EST"] / t["LEAGUE"]) if t["LEAGUE"].iloc[0] > 0 else 1.0   # league avg of 0 (e.g. QB receptions) -> neutral
    t["RANK"] = t["EST"].rank(ascending=False, method="min").astype(int)
    return t.drop(columns="PRIOR").sort_values("RANK")


def matchup_multiplier(index: float, k: float) -> float:
    """mult = 1 + k * (index - 1), clipped to [0.85, 1.15]."""
    if not np.isfinite(index):
        return 1.0
    return float(np.clip(1.0 + k * (index - 1.0), MULT_LO, MULT_HI))


def _ols_cluster(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """OLS with cluster-robust (CR1) standard errors; returns (beta, se)."""
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    e = y - X @ beta
    scores = pd.DataFrame(X * e[:, None]).groupby(groups).sum().to_numpy()
    n_g, n, k = scores.shape[0], len(y), X.shape[1]
    v = xtx_inv @ (scores.T @ scores) @ xtx_inv * (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - k, 1))
    return beta, np.sqrt(np.clip(np.diag(v), 0.0, None))


def _expanding_index(g: pd.DataFrame, n0: float) -> pd.DataFrame:
    """Opponent index for every defense-game using ONLY earlier weeks (+ last season's average): no look-ahead."""
    g = g.sort_values(["OPP", "SEASON", "WEEK"]).reset_index(drop=True)
    grp = g.groupby(["OPP", "SEASON"])["V"]
    g["CUM_N"] = grp.cumcount()
    g["CUM_SUM"] = grp.cumsum() - g["V"]
    prior = g.groupby(["OPP", "SEASON"])["V"].mean().rename("PRI").reset_index()
    prior["SEASON"] = prior["SEASON"] + 1
    g = g.merge(prior, on=["OPP", "SEASON"], how="left")
    with_prior = (g["CUM_SUM"] + n0 * g["PRI"]) / (g["CUM_N"] + n0)
    no_prior = g["CUM_SUM"] / g["CUM_N"].where(g["CUM_N"] > 0)
    g["EST"] = with_prior.where(g["PRI"].notna(), no_prior)
    league = g.groupby(["SEASON", "WEEK"])["EST"].transform("mean")
    g["INDEX"] = (g["EST"] / league.where(league > 0)).fillna(1.0).where(g["EST"].notna())
    return g[["OPP", "SEASON", "WEEK", "INDEX"]]


def matchup_backtest(weekly: pd.DataFrame, stat: str, first_season: int, n0: float = PRIOR_GAMES) -> Optional[Dict[str, Any]]:
    """
    Estimate k in  stat_game / player_season_mean = a + k * (opp_index - 1) + e   (clustered by player).
    The index is built from earlier weeks only, so k is the weight a live projection should actually give it.
    Returns None if the sample is too small.
    """
    rows: List[pd.DataFrame] = []
    for pos in STAT_POS_GROUPS[stat]:
        idx = _expanding_index(_defense_games(weekly, stat, pos), n0)
        p = weekly[(weekly["POS"] == pos) & (weekly["SEASON"] >= first_season)][["PID", "SEASON", "WEEK", "OPP", stat]].copy()
        grp = p.groupby(["PID", "SEASON"])[stat]
        p["PMEAN"], p["CNT"] = grp.transform("mean"), grp.transform("size")
        p = p[(p["CNT"] >= 8) & (p["PMEAN"] >= BT_MIN_MEAN[stat])]
        p = p.merge(idx, on=["OPP", "SEASON", "WEEK"], how="inner").dropna(subset=["INDEX"])
        p["R"] = p[stat] / p["PMEAN"]
        rows.append(p[["PID", "R", "INDEX"]])
    d = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(d) < 200 or d["PID"].nunique() < 15:
        return None
    X = np.column_stack([np.ones(len(d)), d["INDEX"].to_numpy(float) - 1.0])
    beta, se = _ols_cluster(X, d["R"].to_numpy(float), d["PID"].to_numpy())
    return {"STAT": stat, "POS": "/".join(STAT_POS_GROUPS[stat]), "K_HAT": float(beta[1]), "SE": float(se[1]),
            "N": int(len(d)), "PLAYERS": int(d["PID"].nunique())}


def calibrated_k(k_hat: float, se: float, k0: float, tau: float = K_TAU) -> float:
    """Shrink a backtested k toward the default: k0 + (k_hat - k0) * tau^2 / (tau^2 + se^2), clipped to [0, 0.8]."""
    return float(np.clip(k0 + (k_hat - k0) * tau ** 2 / (tau ** 2 + se ** 2), 0.0, 0.8))


def run_backtests(weekly: pd.DataFrame, first_season: int) -> pd.DataFrame:
    """Backtest every stat; adds default k, shrunk k (K_EFF) and a significance flag."""
    out: List[Dict[str, Any]] = []
    for stat in DEFAULT_K:
        r = matchup_backtest(weekly, stat, first_season)
        if r is None:
            continue
        r["K_DEFAULT"] = DEFAULT_K[stat]
        r["K_EFF"] = calibrated_k(r["K_HAT"], r["SE"], DEFAULT_K[stat])
        r["SIGNIFICANT"] = "yes" if abs(r["K_HAT"]) >= 2 * r["SE"] else "no"
        out.append(r)
    return pd.DataFrame(out)

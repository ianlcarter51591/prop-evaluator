"""
NFL bet log: bets you actually placed, graded by week, with closing-line value (CLV), units, ROI and calibration.

Math
----
Implied probability of American odds o:  q = |o| / (|o| + 100) if o < 0 else 100 / (o + 100).
Two-way devig at the close:              q_close(side) = q_side / (q_over + q_under).
CLV (percentage points)                = q_close(your side) - devigged market probability when you bet,
                                         measured only when the closing line equals your line (else LINE_MOVE shows the move:
                                         positive = the line moved in your favour).
PROFIT (units)                         = stake * b on a win (b = 100/|o| if o < 0 else o/100), -stake on a loss, 0 on a push.
Brier score                            = mean((p - won)^2), for the model's probability and for the market's.
A bet's game is found like a snapshot's: WEEK if the matchup gave one, else the first game logged after LAST_WEEK_SEEN.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

import nfl_snapshots as sn

LOG_NAME = "nfl_bet_log.csv"
LOG_COLS: List[str] = [
    "ID", "TS", "SEASON", "WEEK", "LAST_WEEK_SEEN", "PLAYER", "STAT", "SIDE", "LINE", "ODDS", "STAKE", "OPPONENT", "TIER",
    "P_MODEL", "P_MARKET", "EDGE_PP", "EV_PCT", "CLOSE_LINE", "CLOSE_OVER", "CLOSE_UNDER", "LINE_MOVE", "CLV_PP",
    "RESULT", "ACTUAL", "PROFIT",
]
NUM_COLS: List[str] = ["SEASON", "WEEK", "LAST_WEEK_SEEN", "LINE", "ODDS", "STAKE", "P_MODEL", "P_MARKET", "EDGE_PP", "EV_PCT",
                       "CLOSE_LINE", "CLOSE_OVER", "CLOSE_UNDER", "LINE_MOVE", "CLV_PP", "ACTUAL", "PROFIT"]
EDITABLE_COLS: List[str] = ["STAKE", "WEEK", "CLOSE_LINE", "CLOSE_OVER", "CLOSE_UNDER", "RESULT", "ACTUAL"]
RESULT_OPTIONS: List[str] = ["Pending", "Win", "Loss", "Push"]
KEY_COLS: List[str] = ["PLAYER", "STAT", "SIDE", "LINE", "SEASON", "LAST_WEEK_SEEN"]


def _implied(odds: np.ndarray) -> np.ndarray:
    """Vectorised implied probability of American odds."""
    o = np.asarray(odds, dtype=float)
    return np.where(o < 0, -o / (-o + 100.0), 100.0 / (o + 100.0))


def recompute_derived(df: pd.DataFrame) -> pd.DataFrame:
    """LINE_MOVE, CLV_PP and PROFIT from the editable columns (vectorised)."""
    df = df.copy()
    is_over = df["SIDE"].eq("Over").to_numpy()
    df["LINE_MOVE"] = (df["CLOSE_LINE"] - df["LINE"]) * np.where(is_over, 1.0, -1.0)
    co, cu = df["CLOSE_OVER"].to_numpy(float), df["CLOSE_UNDER"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (np.abs(co) >= 100) & (np.abs(cu) >= 100) & (df["CLOSE_LINE"] == df["LINE"]).to_numpy()
        io_, iu_ = _implied(co), _implied(cu)
        q_close = np.where(is_over, io_, iu_) / (io_ + iu_)
        df["CLV_PP"] = np.where(valid, q_close * 100.0 - df["P_MARKET"].to_numpy(float), np.nan)
        odds, stake = df["ODDS"].to_numpy(float), df["STAKE"].to_numpy(float)
        b = np.where(odds < 0, 100.0 / np.abs(odds), odds / 100.0)
        res = df["RESULT"]
        df["PROFIT"] = np.select([res.eq("Win"), res.eq("Loss"), res.eq("Push")], [stake * b, -stake, 0.0], default=np.nan)
    return df


def normalize_log(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure columns/dtypes, valid results, unique ids; then recompute the derived columns."""
    df = df.copy()
    for c in LOG_COLS:
        if c not in df.columns:
            df[c] = np.nan if c in NUM_COLS else ""
    df = df[LOG_COLS]
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in [c for c in LOG_COLS if c not in NUM_COLS]:
        df[c] = df[c].fillna("").astype(str)
    df["RESULT"] = np.where(df["RESULT"].isin(RESULT_OPTIONS), df["RESULT"], "Pending")
    df = df[df["PLAYER"].str.strip() != ""].copy()
    missing = df["ID"].str.strip() == ""
    df.loc[missing, "ID"] = [uuid.uuid4().hex[:8] for _ in range(int(missing.sum()))]
    return recompute_derived(df.drop_duplicates("ID", keep="last").reset_index(drop=True))


def merge_logs(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """Union of two logs by ID (rows in `b` win): used for CSV import and for recovering a saved copy."""
    return normalize_log(pd.concat([normalize_log(a), normalize_log(b)], ignore_index=True))


def make_bet_row(res: Mapping[str, Any], stat_category: str, line: float, side: str, stake: float, season: int) -> Dict[str, Any]:
    """One log row from an evaluation result for the side you bet."""
    s = res["sides"][side]
    matchup = res.get("matchup") or {}
    six = res.get("six_point") or {}
    return {
        "ID": uuid.uuid4().hex[:8], "TS": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"), "SEASON": int(season),
        "WEEK": matchup.get("week"), "LAST_WEEK_SEEN": sn._last_week_seen(res), "PLAYER": res.get("player_name", ""),
        "STAT": stat_category.upper().strip(), "SIDE": side, "LINE": float(line), "ODDS": s["odds"], "STAKE": float(stake),
        "OPPONENT": matchup.get("opponent") or "", "TIER": (six.get(side) or {}).get("tier", ""),
        "P_MODEL": s["p_model"], "P_MARKET": s["p_market"], "EDGE_PP": s["edge_pp"], "EV_PCT": s["ev_pct"], "RESULT": "Pending",
    }


def add_bet(log: pd.DataFrame, row: Dict[str, Any]) -> Tuple[pd.DataFrame, bool]:
    """Append unless the same bet (player, stat, side, line, season, game) is already pending or graded."""
    log, probe = normalize_log(log), normalize_log(pd.DataFrame([row]))
    key = lambda d: d[KEY_COLS].astype(str).agg("|".join, axis=1)
    if len(log) and key(log).isin(key(probe)).any():
        return log, False
    return normalize_log(pd.concat([log, probe], ignore_index=True)), True


def apply_board_as_closing(log: pd.DataFrame, board_summary: pd.DataFrame, stat_category: str) -> Tuple[pd.DataFrame, int]:
    """Copy the current board's line and prices into CLOSE_* for pending bets on that stat (use right before kickoff)."""
    if board_summary.empty:
        return log, 0
    summ = board_summary.drop_duplicates("Player").set_index("Player")
    mask = (log["RESULT"] == "Pending") & (log["STAT"] == stat_category.upper().strip()) & log["PLAYER"].isin(summ.index)
    if not mask.any():
        return log, 0
    log = log.copy()
    for col, src in (("CLOSE_LINE", "Line"), ("CLOSE_OVER", "Over"), ("CLOSE_UNDER", "Under")):
        log.loc[mask, col] = log.loc[mask, "PLAYER"].map(summ[src]).astype(float)
    return recompute_derived(log), int(mask.sum())


def auto_grade(log: pd.DataFrame, fetch_logs: Callable[[str, int], Tuple[pd.DataFrame, Optional[str]]]) -> Tuple[pd.DataFrame, int, List[str]]:
    """Grade pending bets whose game is in the weekly logs. Vectorised lookup; one fetch per player."""
    log = normalize_log(log)
    due = log[log["RESULT"] == "Pending"]
    if due.empty:
        return log, 0, []
    notes: List[str] = []
    logs: Dict[str, pd.DataFrame] = {}
    for name, season in due[["PLAYER", "SEASON"]].dropna().drop_duplicates().itertuples(index=False):
        try:
            lg, err = fetch_logs(name, int(season))
        except Exception as e:
            lg, err = pd.DataFrame(), str(e)
        if err:
            notes.append(f"{name}: {err}")
        if name not in logs or (lg is not None and not lg.empty):
            logs[name] = lg
    look = sn._lookup(logs, sorted(due["STAT"].unique()))
    if look.empty:
        return log, 0, notes
    cand = due[["ID", "PLAYER", "STAT", "SEASON", "WEEK", "LAST_WEEK_SEEN"]].merge(look, on=["PLAYER", "STAT", "SEASON"], how="left", suffixes=("", "_L"))
    hit = np.where(cand["WEEK"].notna(), cand["WEEK_L"] == cand["WEEK"], cand["WEEK_L"] > cand["LAST_WEEK_SEEN"].fillna(0))
    got = cand[hit & cand["ACTUAL"].notna()].sort_values(["ID", "WEEK_L"]).drop_duplicates("ID", keep="first").set_index("ID")
    if got.empty:
        return log, 0, notes
    log = log.set_index("ID", drop=False)
    actual, line = got["ACTUAL"].to_numpy(float), log.loc[got.index, "LINE"].to_numpy(float)
    over = log.loc[got.index, "SIDE"].eq("Over").to_numpy()
    log.loc[got.index, "ACTUAL"] = actual
    log.loc[got.index, "WEEK"] = got["WEEK_L"].to_numpy()
    log.loc[got.index, "RESULT"] = np.where(actual == line, "Push", np.where((actual > line) == over, "Win", "Loss"))
    return recompute_derived(log.reset_index(drop=True)), int(len(got)), notes


def summarize_log(log: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """(headline stats, per-stat table, calibration table by model-probability bucket)."""
    wl = log[log["RESULT"].isin(["Win", "Loss"])]
    won = wl["RESULT"].eq("Win").to_numpy(float)
    staked = float(wl["STAKE"].sum())
    clv = log["CLV_PP"].dropna()
    pm, pk = wl["P_MODEL"].to_numpy(float) / 100.0, wl["P_MARKET"].to_numpy(float) / 100.0
    nan = float("nan")
    stats: Dict[str, float] = {
        "bets": float(len(log)), "graded": float(log["RESULT"].isin(RESULT_OPTIONS[1:]).sum()),
        "win_pct": float(won.mean() * 100.0) if len(wl) else nan, "units": float(log["PROFIT"].sum()),
        "roi": float(log["PROFIT"].sum() / staked * 100.0) if staked > 0 else nan,
        "avg_clv": float(clv.mean()) if len(clv) else nan, "pos_clv": float((clv > 0).mean() * 100.0) if len(clv) else nan,
        "n_clv": float(len(clv)), "brier_model": float(np.mean((pm - won) ** 2)) if len(wl) else nan,
        "brier_market": float(np.mean((pk - won) ** 2)) if len(wl) else nan,
    }
    tmp = log.assign(_staked=np.where(log["RESULT"].isin(["Win", "Loss"]), log["STAKE"], 0.0))
    by_stat = tmp.groupby("STAT").agg(BETS=("ID", "count"), UNITS=("PROFIT", "sum"), STAKED=("_staked", "sum"),
                                      AVG_EDGE_PP=("EDGE_PP", "mean"), AVG_CLV_PP=("CLV_PP", "mean"))
    by_stat["ROI_%"] = np.where(by_stat["STAKED"] > 0, by_stat["UNITS"] / by_stat["STAKED"].replace(0, np.nan) * 100.0, np.nan)
    by_stat = by_stat.drop(columns="STAKED").round(2).reset_index()
    if len(wl):
        bins = pd.cut(wl["P_MODEL"], [0, 45, 50, 55, 60, 65, 101], right=False)
        calib = wl.assign(_bin=bins, _w=won * 100.0).groupby("_bin", observed=True).agg(
            BETS=("ID", "count"), AVG_MODEL_PCT=("P_MODEL", "mean"), ACTUAL_WIN_PCT=("_w", "mean")).round(1).reset_index()
        calib["_bin"] = calib["_bin"].astype(str)
        calib = calib.rename(columns={"_bin": "MODEL_PROB_BUCKET"})
    else:
        calib = pd.DataFrame(columns=["MODEL_PROB_BUCKET", "BETS", "AVG_MODEL_PCT", "ACTUAL_WIN_PCT"])
    return stats, by_stat, calib

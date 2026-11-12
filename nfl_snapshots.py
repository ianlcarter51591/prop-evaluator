"""
NFL prediction snapshots: every evaluation is saved with what the model believed, then graded against the real game.

Logic
-----
A snapshot stores the projection (MU), P(Over)/P(Under), six-point scores and failed criteria, the week of the game (when the
matchup resolved it) and the last week already in the player's log (LAST_WEEK_SEEN, so the game can be found later even when
the week is unknown). Grading finds the player's game: WEEK == WEEK if known, else the first log row after LAST_WEEK_SEEN, then
    STAT_ERR = ACTUAL - MU                Z = STAT_ERR / RECENT_SD         (RECENT_SD = sd of his last 10 games at snapshot time)
    VOL_ERR_PCT = (VOL_ACT - VOL_EXP) / VOL_EXP    VOL_EXP = mean of his previous 8 games of the volume stat
        (pass attempts for passing props, rushing attempts for rushing yards, receptions for receiving yards)
Heuristic miss reason (a hint, not proof): usage miss if |VOL_ERR_PCT| >= USAGE_MISS_PCT; else outlier if the result was more
than 1 sd against the pick; else normal variance. A player with later games logged but none for the target week is voided (DNP).
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

SNAP_PATH = os.environ.get("SNAPSHOT_PATH", "nfl_snapshots.csv")
USAGE_MISS_PCT = 0.25
VOL_WINDOW = 8
KEY_COLS: List[str] = ["PLAYER", "STAT", "LINE", "SEASON", "GAME_KEY"]
# Volume stat that drives each prop (None = the prop is itself the volume).
VOLUME_COL: Dict[str, Optional[str]] = {
    "PASSING YARDS": "PASS ATTEMPTS", "PASS COMPLETIONS": "PASS ATTEMPTS", "PASSING TDS": "PASS ATTEMPTS",
    "INTERCEPTIONS": "PASS ATTEMPTS", "RUSHING YARDS": "RUSHING ATTEMPTS", "RECEIVING YARDS": "RECEPTIONS",
    "PASS ATTEMPTS": None, "RUSHING ATTEMPTS": None, "RECEPTIONS": None,
    "ANYTIME TOUCHDOWN SCORER": "TOUCHES", "TOUCHES": None,
    "KICKING POINTS": "FIELD GOAL ATTEMPTS", "FIELD GOALS MADE": "FIELD GOAL ATTEMPTS", "EXTRA POINTS MADE": "EXTRA POINT ATTEMPTS",
}
TEXT_COLS: List[str] = ["SNAP_ID", "TS_ET", "GAMEDAY", "GAME_KEY", "GAME", "PLAYER", "TEAM", "POS", "STAT", "OPPONENT", "SOURCE",
                        "SHOWN_SIDE", "EV_SIDE", "TIER_SHOWN", "FAILED_OVER", "FAILED_UNDER", "STATUS", "SHOWN_RESULT",
                        "EV_RESULT", "WHY"]
NUM_COLS: List[str] = ["SEASON", "WEEK", "LAST_WEEK_SEEN", "LINE", "OVER_ODDS", "UNDER_ODDS", "MU", "PROJ_MEAN", "RAW_AVG",
                       "MATCHUP_MULT", "USAGE_MULT", "ENV_MULT", "P_OVER", "P_UNDER", "PM_OVER", "PM_UNDER", "SIX_OVER", "SIX_UNDER", "RECENT_SD", "ACTUAL",
                       "STAT_ERR", "Z", "VOL_EXP", "VOL_ACT", "VOL_ERR_PCT"]
COLS: List[str] = TEXT_COLS + NUM_COLS


def _failed(six_side: Optional[Mapping[str, Any]]) -> str:
    """Labels of the criteria that did not pass, '; '-joined (unverified ones are marked)."""
    if not six_side:
        return ""
    return "; ".join(f"{c['label']}{'' if c.get('verified', True) else ' (unverified)'}"
                     for c in six_side.get("checks", []) if not c.get("passed"))


def _last_week_seen(res: Mapping[str, Any]) -> Optional[int]:
    """Last current-season week in the result's recent-games list (labels look like 'Wk 9<br>vs DAL'); older seasons start with an apostrophe."""
    import re
    weeks = [int(m.group(1)) for g in (res.get("gamelog_details") or [])
             for m in [re.match(r"Wk (\d+)", str(g.get("x_label", "")))] if m]
    return max(weeks) if weeks else None


def make_row(res: Mapping[str, Any], stat_category: str, line: float, over: float, under: float, source: str,
             now_et: str, season: int, gameday: str = "") -> Dict[str, Any]:
    """One snapshot row from an evaluate_prop result (a dict without 'error')."""
    six = res.get("six_point") or {}
    sides = res.get("sides") or {}
    shown = six.get("best", "")
    matchup = res.get("matchup") or {}
    week = matchup.get("week")
    last = _last_week_seen(res)
    vals = np.array([g["stat_value"] for g in (res.get("gamelog_details") or [])], dtype=float)
    game_key = f"{season}-W{int(week)}" if week not in (None, "") and pd.notna(week) else f"{season}-after-W{last if last is not None else 0}"
    team, opp = str(res.get("team") or ""), str(matchup.get("opponent") or "")
    game = f"{game_key} {'-'.join(sorted([team, opp]))}" if team and opp else ""       # same text for both teams' players
    return {
        "SNAP_ID": uuid.uuid4().hex[:8], "TS_ET": now_et, "GAMEDAY": gameday or matchup.get("gameday") or "", "GAME_KEY": game_key,
        "GAME": game, "ENV_MULT": res.get("env_mult"),
        "SEASON": int(season), "WEEK": week, "LAST_WEEK_SEEN": last, "PLAYER": res.get("player_name", ""), "TEAM": res.get("team", ""),
        "POS": res.get("pos", ""), "STAT": stat_category.upper().strip(), "LINE": float(line), "OVER_ODDS": float(over),
        "UNDER_ODDS": float(under), "OPPONENT": matchup.get("opponent") or "", "SOURCE": source,
        "MU": res.get("mu"), "PROJ_MEAN": res.get("mean"), "RAW_AVG": res.get("raw_avg"), "MATCHUP_MULT": res.get("matchup_mult"),
        "USAGE_MULT": res.get("usage_mult"), "P_OVER": (sides.get("Over") or {}).get("p_model"),
        "P_UNDER": (sides.get("Under") or {}).get("p_model"),
        "PM_OVER": (sides.get("Over") or {}).get("p_market"), "PM_UNDER": (sides.get("Under") or {}).get("p_market"), "RECENT_SD": float(np.std(vals, ddof=1)) if len(vals) > 2 else np.nan,
        "SHOWN_SIDE": shown, "EV_SIDE": six.get("ev_side", ""), "TIER_SHOWN": (six.get(shown) or {}).get("tier", ""),
        "SIX_OVER": (six.get("Over") or {}).get("score"), "SIX_UNDER": (six.get("Under") or {}).get("score"),
        "FAILED_OVER": _failed(six.get("Over")), "FAILED_UNDER": _failed(six.get("Under")), "STATUS": "Pending",
    }


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every column exists with the right dtype; drop rows with no player."""
    d = df.copy()
    for c in COLS:
        if c not in d.columns:
            d[c] = np.nan if c in NUM_COLS else ""
    d = d[COLS]
    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    for c in TEXT_COLS:
        d[c] = d[c].fillna("").astype(str)
    d["STATUS"] = np.where(d["STATUS"].isin(["Graded", "Void"]), d["STATUS"], "Pending")
    return d[d["PLAYER"].str.strip() != ""].reset_index(drop=True)


def upsert(existing: pd.DataFrame, rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Add rows; a newer snapshot of the same (player, stat, line, season, game) replaces an older PENDING one. Graded rows stay."""
    new = normalize(pd.DataFrame(rows)) if rows else normalize(pd.DataFrame(columns=COLS))
    allr = pd.concat([normalize(existing), new], ignore_index=True)
    key = allr[KEY_COLS].astype(str).agg("|".join, axis=1)
    done = key[allr["STATUS"] != "Pending"]
    allr = allr[~(key.isin(done) & (allr["STATUS"] == "Pending"))]
    return allr.drop_duplicates(KEY_COLS, keep="last").reset_index(drop=True)


def _lookup(logs_by_player: Mapping[str, pd.DataFrame], stats: List[str]) -> pd.DataFrame:
    """(PLAYER, STAT, SEASON, WEEK, ACTUAL, VOL_ACT, VOL_EXP) for every logged game; VOL_EXP uses only earlier games. Vectorised."""
    parts: List[pd.DataFrame] = []
    for name, df in logs_by_player.items():
        if df is None or df.empty or not {"SEASON", "WEEK"} <= set(df.columns):
            continue
        d = df.sort_values(["SEASON", "WEEK"]).reset_index(drop=True)
        for s in stats:
            if s not in d.columns:
                continue
            vcol = VOLUME_COL.get(s)
            vol = pd.to_numeric(d[vcol], errors="coerce") if vcol and vcol in d.columns else pd.Series(np.nan, index=d.index)
            parts.append(pd.DataFrame({
                "PLAYER": name, "STAT": s, "SEASON": pd.to_numeric(d["SEASON"], errors="coerce"),
                "WEEK": pd.to_numeric(d["WEEK"], errors="coerce"), "ACTUAL": pd.to_numeric(d[s], errors="coerce"),
                "VOL_ACT": vol, "VOL_EXP": vol.shift(1).rolling(VOL_WINDOW, min_periods=3).mean()}))
    cols = ["PLAYER", "STAT", "SEASON", "WEEK", "ACTUAL", "VOL_ACT", "VOL_EXP"]
    return pd.concat(parts, ignore_index=True).drop_duplicates(["PLAYER", "STAT", "SEASON", "WEEK"]) if parts else pd.DataFrame(columns=cols)


def grade(df: pd.DataFrame, fetch_logs: Callable[[str, int], Tuple[pd.DataFrame, Optional[str]]]) -> Tuple[pd.DataFrame, int, List[str]]:
    """Grade pending snapshots whose game has been played. `fetch_logs(name, season)` -> (weekly log with SEASON, WEEK, stat cols; error)."""
    d = normalize(df)
    due = d[d["STATUS"] == "Pending"]
    if due.empty:
        return d, 0, []
    notes: List[str] = []
    logs: Dict[str, pd.DataFrame] = {}
    for name, season in due[["PLAYER", "SEASON"]].drop_duplicates().itertuples(index=False):
        try:
            lg, err = fetch_logs(name, int(season))
        except Exception as e:  # one failure must not stop the others
            lg, err = pd.DataFrame(), str(e)
        if err:
            notes.append(f"{name}: {err}")
        if name not in logs or (lg is not None and not lg.empty):
            logs[name] = lg
    look = _lookup(logs, sorted(due["STAT"].unique()))
    if look.empty:
        return d, 0, notes
    cand = due[["SNAP_ID", "PLAYER", "STAT", "SEASON", "WEEK", "LAST_WEEK_SEEN"]].merge(look, on=["PLAYER", "STAT", "SEASON"], how="left", suffixes=("", "_L"))
    wk_known = cand["WEEK"].notna()
    hit = np.where(wk_known, cand["WEEK_L"] == cand["WEEK"], cand["WEEK_L"] > cand["LAST_WEEK_SEEN"].fillna(0))
    got = cand[hit & cand["ACTUAL"].notna()].sort_values(["SNAP_ID", "WEEK_L"]).drop_duplicates("SNAP_ID", keep="first").set_index("SNAP_ID")
    d = d.set_index("SNAP_ID", drop=False)
    d.loc[got.index, ["ACTUAL", "VOL_ACT", "VOL_EXP"]] = got[["ACTUAL", "VOL_ACT", "VOL_EXP"]].to_numpy()
    d.loc[got.index, "WEEK"] = got["WEEK_L"].to_numpy()
    d.loc[got.index, "STATUS"] = "Graded"
    # DNP: the target week is known, later weeks are logged, but the target week itself is missing.
    maxwk = look.groupby(["PLAYER", "SEASON"])["WEEK"].max().rename("MAXWK").reset_index()
    pend = due[~due["SNAP_ID"].isin(got.index) & due["WEEK"].notna()].merge(maxwk, on=["PLAYER", "SEASON"], how="left")
    void = pend[pend["MAXWK"] > pend["WEEK"]]["SNAP_ID"]
    d.loc[void, "STATUS"] = "Void"
    d.loc[void, "WHY"] = "No stats recorded for that week (did not play?)"
    d = _derive(d.reset_index(drop=True))
    return d, int(len(got)), notes


def _result(actual: pd.Series, line: pd.Series, side: pd.Series) -> pd.Series:
    """Hit / Miss / Push for a chosen side."""
    hit = np.where(side == "Over", actual > line, np.where(side == "Under", actual < line, False))
    return pd.Series(np.where(actual == line, "Push", np.where(side.isin(["Over", "Under"]), np.where(hit, "Hit", "Miss"), "")), index=actual.index)


def _derive(d: pd.DataFrame) -> pd.DataFrame:
    """Error columns, side results and the heuristic WHY for graded rows."""
    d = d.copy()
    g = d["STATUS"] == "Graded"
    d.loc[g, "STAT_ERR"] = d.loc[g, "ACTUAL"] - d.loc[g, "MU"]
    d.loc[g, "Z"] = d.loc[g, "STAT_ERR"] / d.loc[g, "RECENT_SD"].where(d.loc[g, "RECENT_SD"] > 0)
    d.loc[g, "VOL_ERR_PCT"] = (d.loc[g, "VOL_ACT"] - d.loc[g, "VOL_EXP"]) / d.loc[g, "VOL_EXP"].where(d.loc[g, "VOL_EXP"] > 0)
    d.loc[g, "SHOWN_RESULT"] = _result(d.loc[g, "ACTUAL"], d.loc[g, "LINE"], d.loc[g, "SHOWN_SIDE"])
    d.loc[g, "EV_RESULT"] = _result(d.loc[g, "ACTUAL"], d.loc[g, "LINE"], d.loc[g, "EV_SIDE"])
    miss = g & (d["SHOWN_RESULT"] == "Miss")
    usage = d["VOL_ERR_PCT"].abs() >= USAGE_MISS_PCT
    against = np.where(d["SHOWN_SIDE"] == "Over", d["Z"] <= -1.0, d["Z"] >= 1.0)
    d.loc[g, "WHY"] = ""
    d.loc[g & (d["SHOWN_RESULT"] == "Hit"), "WHY"] = "Hit"
    d.loc[g & (d["SHOWN_RESULT"] == "Push"), "WHY"] = "Push"
    d.loc[miss, "WHY"] = "Within normal variance"
    d.loc[miss & against, "WHY"] = "Outlier game: more than 1 sd against the pick"
    d.loc[miss & usage & (d["VOL_ERR_PCT"] < 0), "WHY"] = "Usage miss: volume well below normal"
    d.loc[miss & usage & (d["VOL_ERR_PCT"] > 0), "WHY"] = "Usage miss: volume well above normal"
    return d


def by_tier(d: pd.DataFrame) -> pd.DataFrame:
    """Graded results per displayed tier: picks, hit rate (pushes excluded), share of misses that were usage misses."""
    g = d[(d["STATUS"] == "Graded") & d["SHOWN_RESULT"].isin(["Hit", "Miss"])]
    if g.empty:
        return pd.DataFrame(columns=["TIER", "PICKS", "HIT_%", "AVG_Z", "USAGE_MISSES_%_OF_MISSES"])
    rows = []
    for tier, s in g.groupby(g["TIER_SHOWN"].replace("", "n/a")):
        miss = s[s["SHOWN_RESULT"] == "Miss"]
        rows.append({"TIER": tier, "PICKS": len(s), "HIT_%": 100.0 * (s["SHOWN_RESULT"] == "Hit").mean(), "AVG_Z": s["Z"].mean(),
                     "USAGE_MISSES_%_OF_MISSES": 100.0 * miss["WHY"].str.startswith("Usage miss").mean() if len(miss) else np.nan})
    return pd.DataFrame(rows).sort_values("TIER").reset_index(drop=True)


def load(path: Optional[str] = None) -> pd.DataFrame:
    """Read the snapshot file (empty frame if missing or unreadable)."""
    path = path or SNAP_PATH
    try:
        if os.path.exists(path):
            return normalize(pd.read_csv(path, dtype=str))
    except Exception:
        pass
    return normalize(pd.DataFrame(columns=COLS))


def save(df: pd.DataFrame, path: Optional[str] = None) -> None:
    """Write atomically; failures are swallowed (the in-session copy and the download button remain)."""
    path = path or SNAP_PATH
    try:
        tmp = f"{path}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        pass

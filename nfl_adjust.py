"""NFL injury / opportunity redistribution and defense-tier splits (called natively by multi_sport_engine.evaluate_prop).

Math (all constants are named assumptions you can tune or backtest):
  Opportunity shares come from the team's last 17 games: s_j = stat_j / team_stat over the games j played
  (targets for receiving and passing quality, carries for rushing).
  1. Receiving / rushing: a missing teammate j frees s_j.  Teammate i absorbs
         lift_ij = alpha * s_j * B_ij / sum_{k != j} a_k * s_k * B_kj
     (a_k = P(k plays); B = 1.5 when k and j share a position, else 1), so lift is proportional to existing share.
     Multiplier = 1 + gamma_stat * sum_j (q_j - f_j) * lift_ij, where q_j = P(j misses) and f_j = the share of i's
     baseline built from games j missed (no double counting; a returning teammate pulls usage back down).
  2. Passing: lost receiving quality = sum_j (q_j - f_j) * s_j;  multiplier = 1 - lambda_stat * lost.
  3. QB1 out lowers receiver output by beta * (q - f).
"""
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

from nfl_matchups import _defense_games
from sports_data import clean_name, clean_series

INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
WINDOW_GAMES, RECENT_GAMES, N_PRIOR, DECAY = 17, 8, 4.0, 0.85     # N_PRIOR and DECAY match the engine's projection
SAME_POS_BOOST, MAX_LIFT, MULT_BOUNDS = 1.5, 1.00, (0.70, 1.55)
ABSORB: Dict[str, float] = {"rec": 0.90, "rush": 0.85}
RECEIVERS, RUSHERS = {"WR", "TE", "RB"}, {"RB"}
# engine stat label -> (group, gamma (rec/rush) or lambda (pass), qb_beta)
STAT_MAP: Dict[str, Tuple[str, float, float]] = {
    "PASS ATTEMPTS": ("pass", 0.00, 0.0), "PASS COMPLETIONS": ("pass", 0.10, 0.0),
    "PASSING YARDS": ("pass", 0.15, 0.0), "PASSING TDS": ("pass", 0.12, 0.0),
    "RUSHING ATTEMPTS": ("rush", 0.90, 0.0), "RUSHING YARDS": ("rush", 0.85, 0.0),
    "RECEPTIONS": ("rec", 0.95, 0.05), "RECEIVING YARDS": ("rec", 0.85, 0.08),
}
INJ_COLS = ["NAME", "STATUS", "P_OUT"]


def _walk(obj: Any) -> Iterator[Dict[str, Any]]:
    """Yield every dict nested anywhere in a JSON structure."""
    stack: List[Any] = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            yield o
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)


def _status_text(v: Any) -> str:
    """Status may be a string or a dict with description/name."""
    return str(v.get("description") or v.get("name") or "") if isinstance(v, dict) else str(v or "")


def p_out(status: str, text: str = "") -> float:
    """Probability a listed player misses the game."""
    s, t = status.lower(), f"{status} {text}".lower()
    if re.search(r"\b(out|ir|pup|nfi|injured reserve|suspend\w*)\b", s):
        return 1.0
    if "doubtful" in t:
        return 0.90
    if "questionable" in t or "day" in s:
        return 0.40
    if "probable" in t:
        return 0.05
    return 0.0


def parse_injuries(data: Any) -> pd.DataFrame:
    """Any dict with an 'athlete' and a 'status' is one injury entry (structure-agnostic)."""
    rows: List[Dict[str, Any]] = []
    for d in _walk(data):
        ath = d.get("athlete")
        if not isinstance(ath, dict) or "status" not in d:
            continue
        name = str(ath.get("displayName") or ath.get("fullName") or ath.get("shortName") or "")
        if not name:
            continue
        typ = d.get("type") if isinstance(d.get("type"), dict) else {}
        status = _status_text(d.get("status")).strip() or _status_text(typ).strip()
        rows.append({"NAME": name, "STATUS": status, "P_OUT": p_out(status, f"{_status_text(typ)} {d.get('shortComment', '')} {d.get('longComment', '')}")})
    return pd.DataFrame(rows, columns=INJ_COLS)


@st.cache_data(ttl=600, show_spinner=False)
def load_injuries() -> Tuple[pd.DataFrame, bool]:
    """ESPN NFL injury report and whether it parsed (a league-wide report is never empty in season)."""
    try:
        resp = requests.get(INJURY_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        df = parse_injuries(resp.json())
        return df, bool(len(df))
    except Exception:
        return pd.DataFrame(columns=INJ_COLS), False


def presence(win: pd.DataFrame, pids: List[str], gkeys: List[int]) -> np.ndarray:
    """(K, n) bool: did player k appear in the team's weekly stats for game g?"""
    seen = set(zip(win["PID"], win["GKEY"]))
    return np.array([[(k, g) in seen for g in gkeys] for k in pids], dtype=bool).reshape(len(pids), len(gkeys))


def injury_layer(player_name: str, pid: str, team: str, stat: str, weekly: pd.DataFrame, season: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stat multiplier from injured / returning teammates, plus the table behind it."""
    out: Dict[str, Any] = {"mult": 1.0, "table": [], "listed": [], "notes": [], "feed_ok": True, "share": None, "group": None,
                           "checked": False, "self_status": "", "qb_flag": ""}
    spec = STAT_MAP.get(stat)
    if spec is None:
        out["notes"].append(f"Injury adjustment skipped: no opportunity model for {stat.title()}.")
        return out
    group, coef, beta = spec
    out["group"] = group
    share_col = "RUSHING ATTEMPTS" if group == "rush" else ("TARGETS" if "TARGETS" in weekly.columns else "RECEPTIONS")
    if group != "rush" and share_col != "TARGETS":
        out["notes"].append("Target counts unavailable (update sports_data.py); using reception share instead.")
    w = weekly.assign(GKEY=weekly["SEASON"].astype(int) * 100 + weekly["WEEK"].astype(int))
    rows = w[w["TEAM"] == team]
    if rows.empty:
        out["notes"].append(f"Injury adjustment skipped: no weekly stats found for {team}.")
        return out
    games = rows[["GKEY"]].drop_duplicates().sort_values("GKEY").tail(WINDOW_GAMES)
    win = rows[rows["GKEY"].isin(games["GKEY"])]
    tt = win.groupby("GKEY")[share_col].sum().rename("TT")
    w2 = win.join(tt, on="GKEY")
    pl = w2.groupby("PID").agg(NAME=("NAME", "last"), CNAME=("CNAME", "last"), POS=("POS", "last"), G=("GKEY", "nunique"),
                               NUM=(share_col, "sum"), DEN=("TT", "sum"), PA=("PASS ATTEMPTS", "sum"))
    pl["SHARE"] = pl["NUM"] / pl["DEN"].clip(lower=1e-9)
    if pid not in pl.index:
        out["notes"].append(f"Injury adjustment skipped: {player_name} not found in {team}'s recent weekly stats.")
        return out

    league, feed_ok = load_injuries()
    out["feed_ok"] = feed_ok
    feed = (league.assign(KEY=clean_series(league["NAME"])).sort_values("P_OUT", ascending=False).drop_duplicates("KEY").set_index("KEY")
            if not league.empty else pd.DataFrame(columns=INJ_COLS))
    # The player's own report status (for the six-point injury check); manual overrides apply to him too.
    self_cn = pl.loc[pid, "CNAME"]
    self_p = float(feed["P_OUT"].get(self_cn, 0.0)) if not feed.empty else 0.0
    self_status = str(feed["STATUS"].get(self_cn, "")) if not feed.empty else ""
    for k_, val, tag in (("manual_q", 0.5, "QUESTIONABLE (manual)"), ("manual_out", 1.0, "OUT (manual)"), ("manual_in", 0.0, "ACTIVE (manual)")):
        if self_cn in {clean_name(n) for n in cfg.get(k_, [])}:
            self_p, self_status = val, tag
    out["self_status"] = (self_status or "listed") if self_p > 0 else ""
    pool = pl[(pl["G"] >= 3) & (pl["SHARE"] > 0) & pl["POS"].isin(RUSHERS if group == "rush" else RECEIVERS)]
    pids = pool.index.tolist()
    K = len(pids)
    if group != "pass" and pid not in pids:
        out["notes"].append("Injury adjustment skipped: player is not a regular target/carry share on this team.")
        return out
    q = (pool["CNAME"].map(feed["P_OUT"]) if not feed.empty else pd.Series(np.nan, index=pool.index)).to_numpy(float)
    listed = ~np.isnan(q)
    q = np.nan_to_num(q, nan=0.0)
    status = list(pool["CNAME"].map(feed["STATUS"]).fillna("")) if not feed.empty else [""] * K
    cn = pool["CNAME"].to_numpy()
    for k_, val, tag in (("manual_q", 0.5, "QUESTIONABLE (manual)"), ("manual_out", 1.0, "OUT (manual)"), ("manual_in", 0.0, "ACTIVE (manual)")):
        names = {clean_name(n) for n in cfg.get(k_, [])}
        hit = np.array([c in names for c in cn], dtype=bool)
        q, listed = np.where(hit, val, q), listed | hit
        status = [tag if h else s for h, s in zip(hit, status)]

    mine = win[(win["PID"] == pid) & (win["SEASON"] == int(season))].sort_values("GKEY").tail(RECENT_GAMES)
    gk = mine["GKEY"].tolist()
    n = len(gk)
    present = presence(win, pids, gk) if n else np.zeros((K, 0), dtype=bool)
    wts = DECAY ** np.arange(n)[::-1] if n else np.array([])
    wts = wts / wts.sum() if n else wts
    f_core = ((~present) * wts[None, :]).sum(1) if n else np.zeros(K)
    streak = np.cumprod((~present)[:, ::-1], axis=1).sum(1) if n else np.zeros(K)
    inferred = np.zeros(K, dtype=bool)
    if not feed_ok:
        inferred = (streak >= 3) & ~listed
        q = np.where(inferred, 0.75, q)
        out["notes"].append("Injury feed unavailable: teammates who missed the last 3+ games are counted 75% likely out. Add names in the injury settings to override.")
    f = f_core * n / (n + N_PRIOR)
    i = pids.index(pid) if pid in pids else -1
    if i >= 0 and q[i] > 0:
        out["notes"].append(f"ESPN lists {player_name} as {status[i] or 'injured'}; the projection assumes he plays.")
    if i >= 0:
        q[i] = 0.0
    net = q - f
    if i >= 0:
        net[i] = 0.0
    S, pos = pool["SHARE"].to_numpy(float), pool["POS"].to_numpy()
    a = 1.0 - q
    if i >= 0:
        a[i] = 1.0
    alpha = min(ABSORB.get(group, 0.9) * float(cfg.get("absorb_scale", 1.0)), 1.0)
    if group in ("rec", "rush"):
        B = np.where(pos[:, None] == pos[None, :], SAME_POS_BOOST, 1.0) if group == "rec" else np.ones((K, K))
        M = (a * S)[None, :] * B * (1.0 - np.eye(K))
        lift = np.minimum(alpha * S * B[:, i] / np.maximum(M.sum(1), 1e-9), MAX_LIFT)
        eff = coef * net * lift
    else:
        eff = -coef * net * S
    mult = 1.0 + float(eff.sum())
    if group == "rec" and beta > 0:               # QB1 out lowers receiver output
        qbs = pl[pl["POS"] == "QB"]
        if not qbs.empty and qbs["PA"].max() > 0:
            qb = qbs["PA"].idxmax()
            qcn = qbs.loc[qb, "CNAME"]
            qq = float(feed["P_OUT"].get(qcn, 0.0)) if not feed.empty else 0.0
            qq = 1.0 if qcn in {clean_name(x) for x in cfg.get("manual_out", [])} else qq
            if qq > 0:
                out["qb_flag"] = "QB1 OUT" if qq >= 0.99 else f"QB1 {qq:.0%} to miss"
            qb_games = set(win.loc[win["PID"] == qb, "GKEY"])
            qf = float(sum(wi for g, wi in zip(gk, wts) if g not in qb_games) * n / (n + N_PRIOR)) if n else 0.0
            pen = beta * (qq - qf)
            mult -= pen
            if abs(pen) > 0.005:
                lab = "QB1 OUT" if qq >= 0.99 else f"QB1 {qq:.0%} to miss" if qq > 0 else "QB1 back (missed recent games)"
                out["table"].append({"Teammate": qbs.loc[qb, "NAME"], "Pos": "QB", "Status": lab, "Share %": None,
                                     "Recent games missed": f"{qf:.0%}", "Weight": round(qq - qf, 2), "Stat effect %": round(-pen * 100, 1)})
    out["mult"] = float(np.clip(mult, *MULT_BOUNDS))
    out["share"] = float(pl.loc[pid, "SHARE"]) * 100.0
    out["listed"] = [{"name": str(pool.iloc[j]["NAME"]), "status": status[j] or "listed", "p_out": float(q[j])}
                     for j in np.flatnonzero(listed) if j != i]
    out["checked"] = True   # reached the end: listed teammates, QB and the player's own status are all populated
    show = np.flatnonzero((np.abs(net) > 0.02) & (np.abs(eff) > 1e-4))
    for j in show[np.argsort(-np.abs(eff[show]))]:
        label = ("OUT" if q[j] >= 0.99 else f"{q[j]:.0%} to miss" + (" (inferred)" if inferred[j] else "")) if q[j] > 0 else "Not listed: treated as returning (missed recent games)"
        out["table"].append({"Teammate": pool.iloc[j]["NAME"], "Pos": pool.iloc[j]["POS"], "Status": label,
                             "Share %": round(float(S[j]) * 100, 1), "Recent games missed": f"{f_core[j]:.0%}",
                             "Weight": round(float(net[j]), 2), "Stat effect %": round(float(eff[j]) * 100, 1)})
    return out


def tier_splits(weekly: pd.DataFrame, pid: str, pos: str, stat: str) -> Optional[Dict[str, Any]]:
    """Player's average vs the toughest / softest defenses (ranked per season by what they allow his position for this stat)."""
    g = _defense_games(weekly, stat, pos)
    allowed = g.groupby(["SEASON", "OPP"])["V"].mean()
    rank = allowed.groupby(level=0).rank(ascending=False, method="min")       # 1 = allows the most = softest
    n_teams = allowed.groupby(level=0).size()
    mine = weekly[weekly["PID"] == pid].copy()
    if mine.empty:
        return None
    mine["RANK"] = rank.reindex(pd.MultiIndex.from_arrays([mine["SEASON"], mine["OPP"]])).to_numpy()
    mine["NT"] = n_teams.reindex(mine["SEASON"]).to_numpy()
    mine = mine.dropna(subset=["RANK"])
    if mine.empty:
        return None

    def tier(mask: pd.Series) -> Dict[str, Any]:
        v = mine.loc[mask, stat]
        return {"avg": float(v.mean()) if len(v) else None, "n": int(len(v))}

    return {"overall": tier(mine["RANK"] > 0), "soft5": tier(mine["RANK"] <= 5), "soft10": tier(mine["RANK"] <= 10),
            "tough10": tier(mine["RANK"] > mine["NT"] - 10), "tough5": tier(mine["RANK"] > mine["NT"] - 5)}

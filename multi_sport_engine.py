"""NFL Monte Carlo prop engine: shrunk EWMA projection x real matchup, Gamma/Normal/NegBin sims, devigged edge, Kelly, 6-Point Assessment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nfl_matchups import DEFAULT_K, matchup_multiplier, matchup_table
from nfl_adjust import injury_layer, tier_splits
from nfl_six_point import canon_stat
from sports_data import current_nfl_season, fetch_nfl_player_logs, load_schedule, load_weekly, next_game

SOURCES: Tuple[str, ...] = ("Model (EWMA mean)", "My mean", "My median")
DEVIG_METHODS: Tuple[str, ...] = ("Multiplicative",)
DEFAULTS: Tuple[str, float] = ("Patrick Mahomes", 265.5)

STATS: List[str] = [
    "Passing Yards", "Pass Completions", "Pass Attempts", "Passing TDs", "Interceptions",
    "Rushing Yards", "Rushing Attempts", "Receiving Yards", "Receptions",
    "Kicking Points", "Field Goals Made", "Anytime Touchdown Scorer",
]
WINDOW = 8
DECAY = 0.85
N_PRIOR = 4.0
N_CV_PRIOR = 4.0
N_SIMS = 50_000
KELLY_FRACTION, KELLY_CAP = 0.25, 0.05

FAMILY: Dict[str, str] = {
    "PASSING YARDS": "gamma", "RUSHING YARDS": "gamma", "RECEIVING YARDS": "gamma", "KICKING POINTS": "gamma",
    "PASS COMPLETIONS": "normal", "PASS ATTEMPTS": "normal", "RUSHING ATTEMPTS": "normal",
    "PASSING TDS": "count", "RECEPTIONS": "count", "INTERCEPTIONS": "count",
    "FIELD GOALS": "count", "TOUCHDOWNS": "count",
}

DEFAULT_CV: Dict[str, float] = {
    "PASSING YARDS": 0.25, "PASS COMPLETIONS": 0.22, "PASS ATTEMPTS": 0.18, "PASSING TDS": 0.65, "RUSHING YARDS": 0.55,
    "RUSHING ATTEMPTS": 0.35, "RECEIVING YARDS": 0.65, "RECEPTIONS": 0.50, "INTERCEPTIONS": 1.00,
    "KICKING POINTS": 0.35, "FIELD GOALS": 0.65, "TOUCHDOWNS": 0.85,
}


def stat_options() -> List[str]:
    return STATS


def default_season() -> int:
    return current_nfl_season()


@dataclass
class Projection:
    mu: float
    sigma: float
    n_current: int
    n_prior: int
    mult: float


def ewma_weights(n: int, decay: float = DECAY) -> np.ndarray:
    w = decay ** np.arange(n)[::-1]
    return w / w.sum()


def _cv(x: np.ndarray) -> Optional[float]:
    if len(x) < 3 or float(np.mean(x)) <= 0:
        return None
    return float(np.std(x, ddof=1) / np.mean(x))


def build_projection(cur_x: np.ndarray, pri_x: np.ndarray, stat: str, mult: float = 1.0) -> Projection:
    n, n0 = len(cur_x), (N_PRIOR if len(pri_x) > 0 else 0.0)
    if n + n0 == 0:
        raise ValueError("No usable games (current or prior season).")
    parts: List[Tuple[float, float]] = []
    if n:
        parts.append((float(n), float(np.sum(ewma_weights(n) * cur_x))))
    if n0:
        parts.append((n0, float(np.mean(pri_x))))
    base = sum(w * v for w, v in parts) / sum(w for w, _ in parts)
    mu = max(base * mult, 0.05)
    default_spread = DEFAULT_CV.get(stat, 0.50)
    cvs = [(w, c) for w, c in ((float(n), _cv(cur_x)), (n0, _cv(pri_x))) if c is not None and w > 0] + [(N_CV_PRIOR, default_spread)]
    cv = max(sum(w * c for w, c in cvs) / sum(w for w, _ in cvs), 0.6 * default_spread)
    return Projection(mu=mu, sigma=cv * mu * float(np.sqrt(1.0 + 1.0 / (n + n0))), n_current=n, n_prior=len(pri_x), mult=mult)


def simulate(mu: float, sigma: float, family: str, n_sims: int = N_SIMS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    var = max(sigma ** 2, 1e-4)
    if family == "gamma":
        shape = max(mu ** 2 / var, 0.5)
        scale = var / mu
        return rng.gamma(shape=shape, scale=scale, size=n_sims)
    if family == "count":
        if var > mu * 1.0001:
            r = max(mu ** 2 / (var - mu), 0.5)
            p = float(r / (r + mu))
            return rng.negative_binomial(n=r, p=p, size=n_sims).astype(float)
        return rng.poisson(lam=mu, size=n_sims).astype(float)
    return np.maximum(rng.normal(loc=mu, scale=sigma, size=n_sims), 0.0)


def implied_prob(odds: float) -> float:
    o = float(odds)
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def payout(odds: float) -> float:
    o = float(odds)
    return 100.0 / abs(o) if o < 0 else o / 100.0


def devig(over_odds: float, under_odds: float) -> Tuple[float, float]:
    po, pu = implied_prob(over_odds), implied_prob(under_odds)
    tot = po + pu
    return (po / tot, pu / tot) if tot > 0 else (0.5, 0.5)


def prob_to_american(p: float) -> int:
    p = float(np.clip(p, 0.001, 0.999))
    return int(round(-100.0 * p / (1.0 - p))) if p >= 0.5 else int(round(100.0 * (1.0 - p) / p))


def kelly_fraction(p_win: float, p_lose: float, odds: float) -> float:
    b = payout(odds)
    return float(np.clip((b * p_win - p_lose) / max(b, 1e-5) * KELLY_FRACTION, 0.0, KELLY_CAP))


def resolve_matchup(team: str, pos: str, stat: str, season: int, opponent: Optional[str], weekly: pd.DataFrame,
                    k_map: Optional[Dict[str, float]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not opponent or str(opponent).upper() == "NONE":
        return None, "No opponent selected: matchup adjustment skipped."
    info: Dict[str, Any] = {}
    note: Optional[str] = None
    try:
        nxt = next_game(team, load_schedule(season))
    except Exception as e:
        nxt, note = None, f"Schedule unavailable ({e})."
    if str(opponent).upper() == "AUTO":
        if nxt is None:
            return None, (note or f"No upcoming game found for {team}; pick an opponent manually.")
        opp, info = nxt["opponent"], nxt
    else:
        opp = str(opponent).upper()
        info = nxt if nxt and nxt["opponent"] == opp else {}
    table = matchup_table(weekly, stat, pos, season)
    if opp not in table.index:
        return None, f"No {pos} defense data for {opp}."
    row = table.loc[opp]
    k = (k_map or {}).get(stat, DEFAULT_K.get(stat, 0.30))
    return {
        "opponent": opp, "pos": pos, "week": info.get("week"), "home": info.get("home"), "gameday": info.get("gameday"),
        "rest": info.get("rest"), "opp_rest": info.get("opp_rest"), "rank": int(row["RANK"]), "n_teams": int(len(table)),
        "allowed": float(row["EST"]), "current_allowed": float(row["CUR"]) if pd.notna(row["CUR"]) else None,
        "league": float(row["LEAGUE"]), "index": float(row["INDEX"]), "games": int(row["GAMES"]),
        "k": float(k), "calibrated": bool(k_map and stat in k_map), "mult": matchup_multiplier(float(row["INDEX"]), float(k)),
        "board": table.reset_index().rename(columns={"OPP": "TEAM"})[["RANK", "TEAM", "EST", "CUR", "INDEX", "GAMES"]],
    }, note


def evaluate_prop(player_name: str, stat_category: str, line: float, over_odds: float, under_odds: float,
                  projection_source: str = "Model (EWMA mean)", season: int = 2026, opponent: Optional[str] = None,
                  matchup_k: Optional[Dict[str, float]] = None, adj_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        line = float(line)
        over_odds = float(over_odds)
        under_odds = float(under_odds)
        
        c_stat = canon_stat(stat_category)
        if c_stat not in FAMILY:
            return {"error": f"Unknown stat category '{stat_category}'."}
        
        logs, err = fetch_nfl_player_logs(player_name, season)
        if err or logs is None or logs.empty:
            return {"error": err or f"No game logs found for '{player_name}'."}
        
        cur = logs[logs["SEASON_TAG"] == "current"].tail(WINDOW)
        pri = logs[logs["SEASON_TAG"] == "prior"]
        team, pos = str(logs["TEAM"].iloc[-1]), str(logs["POS"].mode().iloc[0])

        warnings: List[str] = []
        matchup: Optional[Dict[str, Any]] = None
        weekly: Optional[pd.DataFrame] = None
        try:
            weekly = load_weekly(season, 1)
            matchup, note = resolve_matchup(team, pos, c_stat, season, opponent, weekly, matchup_k)
            if note:
                warnings.append(note)
        except Exception as e:
            warnings.append(f"Matchup data unavailable ({e}); no adjustment applied.")
        mult = float(matchup["mult"]) if matchup else 1.0

        pid = str(logs["PID"].iloc[-1])
        inj: Optional[Dict[str, Any]] = None
        tiers: Optional[Dict[str, Any]] = None
        if weekly is not None:
            try:
                tiers = tier_splits(weekly, pid, pos, c_stat)
            except Exception as e:
                warnings.append(f"Defense-tier splits unavailable ({e}).")
            if adj_cfg and adj_cfg.get("injuries"):
                try:
                    inj = injury_layer(player_name, pid, team, c_stat, weekly, season, adj_cfg)
                    if isinstance(inj, dict) and "notes" in inj:
                        warnings.extend(inj["notes"])
                except Exception as e:
                    warnings.append(f"Injury adjustment unavailable ({e}); none applied.")
        usage_mult = float(inj["mult"]) if inj else 1.0
        total_mult = mult * usage_mult

        stat_col = c_stat if c_stat in cur.columns else stat_category
        cur_vals = cur[stat_col].to_numpy(float) if stat_col in cur.columns else np.array([])
        pri_vals = pri[stat_col].to_numpy(float) if stat_col in pri.columns else np.array([])

        proj = build_projection(cur_vals, pri_vals, c_stat, total_mult)
        recent = logs.tail(10)
        vals = recent[stat_col].to_numpy(float) if stat_col in recent.columns else np.array([])
        mu = proj.mu
        if projection_source == "My mean" and len(vals):
            mu = max(float(np.mean(vals)) * total_mult, 0.05)
        elif projection_source == "My median" and len(vals):
            mu = max(float(np.median(vals)) * total_mult, 0.05)
        if mu != proj.mu:
            proj = Projection(mu, proj.sigma * mu / proj.mu, proj.n_current, proj.n_prior, total_mult)
        sims = simulate(proj.mu, proj.sigma, FAMILY[c_stat])

        p_over, p_under = float(np.mean(sims > line)), float(np.mean(sims < line))
        decided = max(p_over + p_under, 1e-9)
        dv_over, dv_under = devig(over_odds, under_odds)
        ev_over, ev_under = p_over * payout(over_odds) - p_under, p_under * payout(under_odds) - p_over
        is_over = ev_over >= ev_under
        p_w, p_l = (p_over, p_under) if is_over else (p_under, p_over)
        odds_side, ev = (over_odds, ev_over) if is_over else (under_odds, ev_under)
        p_cond, dv_side = p_w / decided, (dv_over if is_over else dv_under)

        return {
            "player_name": player_name, "team": team, "pos": pos, "mean": float(np.mean(sims)), "median": float(np.median(sims)),
            "best_side": f"{'Over' if is_over else 'Under'} {line:g}",
            "ev_pct": float(ev * 100.0), "edge_pp": float((p_cond - dv_side) * 100.0), "p_model": float(p_cond * 100.0), "p_market": float(dv_side * 100.0),
            "fair_odds": prob_to_american(p_cond), "kelly_pct": float(kelly_fraction(p_w, p_l, odds_side) * 100.0),
            "raw_avg": float(np.mean(vals)) if len(vals) else 0.0, "mu": float(proj.mu),
            "hit10": int(np.sum(vals > line)), "n10": len(vals), "hit5": int(np.sum(vals[-5:] > line)), "n5": len(vals[-5:]),
            "n_current": proj.n_current, "n_prior": proj.n_prior, "matchup": matchup, "warnings": warnings,
            "usage_mult": usage_mult, "inj": inj, "tiers": tiers,
            "base_proj": float(proj.mu / total_mult) if total_mult else float(proj.mu),
            "gamelog_details": [{"stat_value": float(v), "season": s, "x_label": f"W{int(w)}"}
                                for v, s, w in zip(vals, recent["SEASON_TAG"], recent["WEEK"])],
        }
    except Exception as e:
        return {"error": f"Evaluation error: {e}"}

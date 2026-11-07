"""NFL Monte Carlo prop engine: shrunk EWMA projection x real matchup, Gamma/Normal/NegBin sims, devigged edge, Kelly, 6-Point Assessment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nfl_matchups import DEFAULT_K, matchup_multiplier, matchup_table
from nfl_adjust import injury_layer, tier_splits
from sports_data import current_nfl_season, fetch_nfl_player_logs, load_schedule, load_weekly, next_game

SOURCES: Tuple[str, ...] = ("Model (EWMA mean)", "My mean", "My median")
DEVIG_METHODS: Tuple[str, ...] = ("Multiplicative",)
DEFAULTS: Tuple[str, float] = ("Aaron Rodgers", 244.5)

STATS: List[str] = [
    "Passing Yards", "Pass Completions", "Pass Attempts", "Passing TDs", "Rushing Yards",
    "Rushing Attempts", "Receiving Yards", "Receptions", "Interceptions",
]
WINDOW = 8             # recent current-season games feeding the EWMA
DECAY = 0.85           # EWMA decay per game of age
N_PRIOR = 4.0          # pseudo-games of last-season weight
N_CV_PRIOR = 4.0       # pseudo-games of weight on the stat's default spread
N_SIMS = 50_000
KELLY_FRACTION, KELLY_CAP = 0.25, 0.05

FAMILY: Dict[str, str] = {
    "PASSING YARDS": "gamma", "RUSHING YARDS": "gamma", "RECEIVING YARDS": "gamma",
    "PASS COMPLETIONS": "normal", "PASS ATTEMPTS": "normal", "RUSHING ATTEMPTS": "normal",
    "PASSING TDS": "count", "RECEPTIONS": "count", "INTERCEPTIONS": "count",
}

DEFAULT_CV: Dict[str, float] = {
    "PASSING YARDS": 0.25, "PASS COMPLETIONS": 0.22, "PASS ATTEMPTS": 0.18, "PASSING TDS": 0.65, "RUSHING YARDS": 0.55,
    "RUSHING ATTEMPTS": 0.35, "RECEIVING YARDS": 0.65, "RECEPTIONS": 0.50, "INTERCEPTIONS": 1.00,
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
    cvs = [(w, c) for w, c in ((float(n), _cv(cur_x)), (n0, _cv(pri_x))) if c is not None and w > 0] + [(N_CV_PRIOR, DEFAULT_CV[stat])]
    cv = max(sum(w * c for w, c in cvs) / sum(w for w, _ in cvs), 0.6 * DEFAULT_CV[stat])
    return Projection(mu=mu, sigma=cv * mu * float(np.sqrt(1.0 + 1.0 / (n + n0))), n_current=n, n_prior=len(pri_x), mult=mult)


def simulate(mu: float, sigma: float, family: str, n_sims: int = N_SIMS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    var = sigma ** 2
    if family == "gamma":
        return np.rint(rng.gamma(mu ** 2 / var, var / mu, n_sims))
    if family == "count":
        if var > mu * 1.0001:
            return rng.negative_binomial(mu ** 2 / (var - mu), mu / var, n_sims).astype(float)
        return rng.poisson(mu, n_sims).astype(float)
    return np.maximum(np.rint(rng.normal(mu, sigma, n_sims)), 0.0)


def implied_prob(odds: float) -> float:
    o = float(odds)
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def payout(odds: float) -> float:
    o = float(odds)
    return 100.0 / abs(o) if o < 0 else o / 100.0


def devig(over_odds: float, under_odds: float) -> Tuple[float, float]:
    po, pu = implied_prob(over_odds), implied_prob(under_odds)
    return po / (po + pu), pu / (po + pu)


def prob_to_american(p: float) -> int:
    p = float(np.clip(p, 0.01, 0.99))
    return int(round(-100.0 * p / (1.0 - p))) if p >= 0.5 else int(round(100.0 * (1.0 - p) / p))


def kelly_fraction(p_win: float, p_lose: float, odds: float) -> float:
    b = payout(odds)
    return float(np.clip((b * p_win - p_lose) / b * KELLY_FRACTION, 0.0, KELLY_CAP))


def evaluate_six_point_criteria(
    stat_category: str,
    line: float,
    is_over: bool,
    edge_pp: float,
    ev_pct: float,
    dv_side: float,
    odds_side: float,
    matchup: Optional[Dict[str, Any]],
    inj: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluates NFL prop against 6 criteria:
    1. Line High Enough (Baby Line)
    2. AI Edge
    3. Implied Chance
    4. Defensive Rank
    5. Injury
    6. Game Script & Rest
    """
    min_line_thresholds = {
        "PASSING YARDS": 150.0, "PASS COMPLETIONS": 12.0, "PASS ATTEMPTS": 20.0, "PASSING TDS": 0.5,
        "RUSHING YARDS": 20.0, "RUSHING ATTEMPTS": 6.0, "RECEIVING YARDS": 20.0, "RECEPTIONS": 2.5,
        "INTERCEPTIONS": 0.5,
    }
    stat_key = stat_category.upper().strip()
    min_req = min_line_thresholds.get(stat_key, 1.5)
    c1_baby_line = line >= min_req

    c2_ai_edge = (edge_pp >= 2.5) and (ev_pct > 0.0)
    c3_implied_chance = (dv_side >= 0.50) or (odds_side <= -105)

    if matchup and "rank" in matchup:
        rank = matchup["rank"]
        n_teams = matchup.get("n_teams", 32)
        c4_def_rank = (rank <= n_teams / 2) if is_over else (rank > n_teams / 2)
    else:
        c4_def_rank = True

    listed = (inj or {}).get("listed", [])
    has_high_risk_injuries = any(item.get("p_out", 0) > 0.8 for item in listed)
    c5_injury = (inj is None or inj.get("feed_ok", True)) and (not has_high_risk_injuries)

    rest_ok = True
    if matchup and matchup.get("rest") is not None and matchup.get("opp_rest") is not None:
        if matchup["rest"] < 6 or (matchup["opp_rest"] - matchup["rest"] >= 3):
            rest_ok = False
    c6_game_script = rest_ok

    checks = {
        "1. Line High Enough (Baby Line)": (c1_baby_line, f"Line ({line:g}) ≥ {min_req:g} stat threshold"),
        "2. AI Edge": (c2_ai_edge, f"Edge ({edge_pp:+.1f}pp) ≥ +2.5pp with positive EV ({ev_pct:+.1f}%)"),
        "3. Implied Chance": (c3_implied_chance, f"Market probability ({dv_side*100:.1f}%) / price ({odds_side:+d}) favors side"),
        "4. Defensive Rank": (c4_def_rank, f"Opponent matchup rank (#{matchup['rank'] if matchup else 'N/A'}) supports {'Over' if is_over else 'Under'}"),
        "5. Injury": (c5_injury, "No active high-risk injury flags disrupting projection stability"),
        "6. Game Script & Rest": (c6_game_script, "Clean schedule & rest spot (no short-week/rest disadvantage)"),
    }

    score = sum(1 for val, _ in checks.values() if val)
    if score == 6:
        verdict = "SHARP PLAY"
    elif score == 5:
        verdict = "LEAN"
    else:
        verdict = "PASS"

    return {
        "score": score,
        "max_score": 6,
        "verdict": verdict,
        "checks": checks,
    }


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
    k = (k_map or {}).get(stat, DEFAULT_K[stat])
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
        for o in (over_odds, under_odds):
            if abs(float(o)) < 100:
                return {"error": f"Odds {o:g} aren't valid American odds (must be <= -100 or >= +100)."}
        stat = stat_category.upper().strip()
        if stat not in FAMILY:
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
            matchup, note = resolve_matchup(team, pos, stat, season, opponent, weekly, matchup_k)
            if note:
                warnings.append(note)
        except Exception as e:
            warnings.append(f"Matchup data unavailable ({e}); no adjustment applied.")
        mult = matchup["mult"] if matchup else 1.0

        pid = str(logs["PID"].iloc[-1])
        inj: Optional[Dict[str, Any]] = None
        tiers: Optional[Dict[str, Any]] = None
        if weekly is not None:
            try:
                tiers = tier_splits(weekly, pid, pos, stat)
            except Exception as e:
                warnings.append(f"Defense-tier splits unavailable ({e}).")
            if adj_cfg and adj_cfg.get("injuries"):
                try:
                    inj = injury_layer(player_name, pid, team, stat, weekly, season, adj_cfg)
                    warnings.extend(inj["notes"])
                except Exception as e:
                    warnings.append(f"Injury adjustment unavailable ({e}); none applied.")
        usage_mult = inj["mult"] if inj else 1.0
        total_mult = mult * usage_mult

        proj = build_projection(cur[stat].to_numpy(float), pri[stat].to_numpy(float), stat, total_mult)
        recent = logs.tail(10)
        vals = recent[stat].to_numpy(float)
        mu = proj.mu
        if projection_source == "My mean" and len(vals):
            mu = max(float(np.mean(vals)) * total_mult, 0.05)
        elif projection_source == "My median" and len(vals):
            mu = max(float(np.median(vals)) * total_mult, 0.05)
        if mu != proj.mu:
            proj = Projection(mu, proj.sigma * mu / proj.mu, proj.n_current, proj.n_prior, total_mult)
        sims = simulate(proj.mu, proj.sigma, FAMILY[stat])

        p_over, p_under = float(np.mean(sims > line)), float(np.mean(sims < line))
        decided = max(p_over + p_under, 1e-9)
        dv_over, dv_under = devig(over_odds, under_odds)
        ev_over, ev_under = p_over * payout(over_odds) - p_under, p_under * payout(under_odds) - p_over
        is_over = ev_over >= ev_under
        p_w, p_l = (p_over, p_under) if is_over else (p_under, p_over)
        odds_side, ev = (over_odds, ev_over) if is_over else (under_odds, ev_under)
        p_cond, dv_side = p_w / decided, (dv_over if is_over else dv_under)

        six_point_eval = evaluate_six_point_criteria(
            stat_category=stat,
            line=line,
            is_over=is_over,
            edge_pp=(p_cond - dv_side) * 100.0,
            ev_pct=ev * 100.0,
            dv_side=dv_side,
            odds_side=odds_side,
            matchup=matchup,
            inj=inj,
        )

        if proj.n_current < 6:
            warnings.append(f"Small sample: {proj.n_current} games this season; projection leans on last season ({proj.n_prior} games).")
        if proj.n_prior == 0:
            warnings.append("No prior-season games found; no shrinkage applied.")
        return {
            "player_name": player_name, "team": team, "pos": pos, "mean": float(np.mean(sims)), "median": float(np.median(sims)),
            "verdict": six_point_eval["verdict"], "best_side": f"{'Over' if is_over else 'Under'} {line:g}",
            "ev_pct": ev * 100.0, "edge_pp": (p_cond - dv_side) * 100.0, "p_model": p_cond * 100.0, "p_market": dv_side * 100.0,
            "fair_odds": prob_to_american(p_cond), "kelly_pct": kelly_fraction(p_w, p_l, odds_side) * 100.0,
            "raw_avg": float(np.mean(vals)) if len(vals) else 0.0, "mu": proj.mu,
            "hit10": int(np.sum(vals > line)), "n10": len(vals), "hit5": int(np.sum(vals[-5:] > line)), "n5": len(vals[-5:]),
            "n_current": proj.n_current, "n_prior": proj.n_prior, "matchup": matchup, "warnings": warnings,
            "six_point_eval": six_point_eval,
            "projection_mean": float(np.mean(sims)), "projection_median": float(np.median(sims)), "edge_vs_market": (p_cond - dv_side) * 100.0,
            "adj_proj": proj.mu, "adjusted_projection": proj.mu, "matchup_mult": mult,
            "usage_mult": usage_mult, "inj": inj, "tiers": tiers,
            "base_proj": proj.mu / total_mult if total_mult else proj.mu, "projection_ewma": proj.mu / total_mult if total_mult else proj.mu,
            "kelly_pct_legacy": kelly_fraction(p_w, p_l, odds_side) * 100.0,
            "gamelog_details": [{"stat_value": float(v), "season": s, "x_label": f"{d}<br>{o}"}
                                for v, s, d, o in zip(vals, recent["SEASON_TAG"], recent["DATE"], recent["OPPONENT"])],
        }
    except Exception as e:
        return {"error": f"Evaluation error: {e}"}

"""NFL Monte Carlo prop engine: shrunk EWMA projection x real matchup, Gamma/Normal/NegBin sims, devigged edge, Kelly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from nfl_matchups import DEFAULT_K, matchup_multiplier, matchup_table
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
N_PRIOR = 4.0          # pseudo-games of last-season weight (roles change more in the NFL than the NBA)
N_CV_PRIOR = 4.0       # pseudo-games of weight on the stat's default spread
N_SIMS = 50_000
KELLY_FRACTION, KELLY_CAP = 0.25, 0.05
# Distribution family per stat: right-skewed yards -> Gamma; high-volume counts -> Normal; low counts -> NegBin/Poisson.
FAMILY: Dict[str, str] = {
    "PASSING YARDS": "gamma", "RUSHING YARDS": "gamma", "RECEIVING YARDS": "gamma",
    "PASS COMPLETIONS": "normal", "PASS ATTEMPTS": "normal", "RUSHING ATTEMPTS": "normal",
    "PASSING TDS": "count", "RECEPTIONS": "count", "INTERCEPTIONS": "count",
}
# Typical game-to-game coefficient of variation (assumptions used only to stabilize tiny samples).
DEFAULT_CV: Dict[str, float] = {
    "PASSING YARDS": 0.25, "PASS COMPLETIONS": 0.22, "PASS ATTEMPTS": 0.18, "PASSING TDS": 0.65, "RUSHING YARDS": 0.55,
    "RUSHING ATTEMPTS": 0.35, "RECEIVING YARDS": 0.65, "RECEPTIONS": 0.50, "INTERCEPTIONS": 1.00,
}


def stat_options() -> List[str]:
    """Stat labels shown in the UI."""
    return STATS


def default_season() -> int:
    """Current NFL season year."""
    return current_nfl_season()


@dataclass
class Projection:
    """Posterior per-game projection for one stat."""
    mu: float
    sigma: float
    n_current: int
    n_prior: int
    mult: float


def ewma_weights(n: int, decay: float = DECAY) -> np.ndarray:
    """Normalized weights for n games ordered oldest -> newest (newest heaviest)."""
    w = decay ** np.arange(n)[::-1]
    return w / w.sum()


def _cv(x: np.ndarray) -> Optional[float]:
    """Coefficient of variation, or None if the sample is too small/degenerate."""
    if len(x) < 3 or float(np.mean(x)) <= 0:
        return None
    return float(np.std(x, ddof=1) / np.mean(x))


def build_projection(cur_x: np.ndarray, pri_x: np.ndarray, stat: str, mult: float = 1.0) -> Projection:
    """
    mu    = [ (n*EWMA_cur + n0*mean_prior) / (n + n0) ] * matchup_mult
    sigma = max(pooled CV, 0.6*default CV) * mu * sqrt(1 + 1/(n + n0)),
            pooled CV = weighted mean of current CV (weight n), prior CV (weight n0) and the stat's default CV.
    """
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
    """Vectorized seeded draws: Gamma (rounded), Normal (rounded, >=0), or NegBin/Poisson for low counts."""
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
    """American odds -> implied probability (with vig)."""
    o = float(odds)
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def payout(odds: float) -> float:
    """Profit per 1 unit staked at American odds."""
    o = float(odds)
    return 100.0 / abs(o) if o < 0 else o / 100.0


def devig(over_odds: float, under_odds: float) -> Tuple[float, float]:
    """Multiplicative devig of a two-way market."""
    po, pu = implied_prob(over_odds), implied_prob(under_odds)
    return po / (po + pu), pu / (po + pu)


def prob_to_american(p: float) -> int:
    """Probability -> American odds (clipped to [1%, 99%])."""
    p = float(np.clip(p, 0.01, 0.99))
    return int(round(-100.0 * p / (1.0 - p))) if p >= 0.5 else int(round(100.0 * (1.0 - p) / p))


def kelly_fraction(p_win: float, p_lose: float, odds: float) -> float:
    """Fractional Kelly stake (share of bankroll), capped; 0 when EV <= 0."""
    b = payout(odds)
    return float(np.clip((b * p_win - p_lose) / b * KELLY_FRACTION, 0.0, KELLY_CAP))


def resolve_matchup(team: str, pos: str, stat: str, season: int, opponent: Optional[str], weekly: pd.DataFrame,
                    k_map: Optional[Dict[str, float]]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Opponent ('AUTO' = next scheduled game) -> rank/allowed/index/multiplier vs the player's position group."""
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
                  matchup_k: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Full evaluation: projection x matchup -> simulation -> devigged edge, EV, Kelly, hit rates."""
    try:
        for o in (over_odds, under_odds):
            if abs(float(o)) < 100:
                return {"error": f"Odds {o:g} aren't valid American odds (must be <= -100 or >= +100). "
                                 "If these came from the Odds API, request oddsFormat=american."}
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
        try:
            matchup, note = resolve_matchup(team, pos, stat, season, opponent, load_weekly(season, 1), matchup_k)
            if note:
                warnings.append(note)
        except Exception as e:
            warnings.append(f"Matchup data unavailable ({e}); no adjustment applied.")
        mult = matchup["mult"] if matchup else 1.0

        proj = build_projection(cur[stat].to_numpy(float), pri[stat].to_numpy(float), stat, mult)
        recent = logs.tail(10)
        vals = recent[stat].to_numpy(float)
        mu = proj.mu
        if projection_source == "My mean" and len(vals):
            mu = max(float(np.mean(vals)) * mult, 0.05)
        elif projection_source == "My median" and len(vals):
            mu = max(float(np.median(vals)) * mult, 0.05)
        if mu != proj.mu:
            proj = Projection(mu, proj.sigma * mu / proj.mu, proj.n_current, proj.n_prior, mult)
        sims = simulate(proj.mu, proj.sigma, FAMILY[stat])

        p_over, p_under = float(np.mean(sims > line)), float(np.mean(sims < line))
        decided = max(p_over + p_under, 1e-9)
        dv_over, dv_under = devig(over_odds, under_odds)
        ev_over, ev_under = p_over * payout(over_odds) - p_under, p_under * payout(under_odds) - p_over
        is_over = ev_over >= ev_under
        p_w, p_l = (p_over, p_under) if is_over else (p_under, p_over)
        odds_side, ev = (over_odds, ev_over) if is_over else (under_odds, ev_under)
        p_cond, dv_side = p_w / decided, (dv_over if is_over else dv_under)

        if proj.n_current < 6:
            warnings.append(f"Small sample: {proj.n_current} games this season; projection leans on last season ({proj.n_prior} games).")
        if proj.n_prior == 0:
            warnings.append("No prior-season games found; no shrinkage applied.")
        return {
            "player_name": player_name, "team": team, "pos": pos, "mean": float(np.mean(sims)), "median": float(np.median(sims)),
            "verdict": "PLAY" if ev > 0 else "PASS", "best_side": f"{'Over' if is_over else 'Under'} {line:g}",
            "ev_pct": ev * 100.0, "edge_pp": (p_cond - dv_side) * 100.0, "p_model": p_cond * 100.0, "p_market": dv_side * 100.0,
            "fair_odds": prob_to_american(p_cond), "kelly_pct": kelly_fraction(p_w, p_l, odds_side) * 100.0,
            "raw_avg": float(np.mean(vals)) if len(vals) else 0.0, "mu": proj.mu,
            "hit10": int(np.sum(vals > line)), "n10": len(vals), "hit5": int(np.sum(vals[-5:] > line)), "n5": len(vals[-5:]),
            "n_current": proj.n_current, "n_prior": proj.n_prior, "matchup": matchup, "warnings": warnings,
            "gamelog_details": [{"stat_value": float(v), "season": s, "x_label": f"{d}<br>{o}"}
                                for v, s, d, o in zip(vals, recent["SEASON_TAG"], recent["DATE"], recent["OPPONENT"])],
        }
    except Exception as e:
        return {"error": f"Evaluation error: {e}"}

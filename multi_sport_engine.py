"""Prop evaluation engine (NFL/NBA/MLB): game logs -> EWMA mean/median -> Monte Carlo pricing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

from nfl_fetcher import current_nfl_season, fetch_nfl_player_logs
from sports_data import (
    fetch_mlb_player_logs,
    fetch_nba_player_logs,
    mlb_season,
    nba_season_start,
)

PRIOR_STRENGTH_GAMES: float = 4.0  # pseudo-games of shrinkage toward the variance prior
PSEUDO_LINE_WINDOW: int = 8        # backtest proxy line = floor(median of last N) + 0.5
MIN_TRAIN: int = 5                 # minimum games of history before a backtest prediction
LOW_SAMPLE_GAMES: int = 5          # below this many games, warn about a small sample
SPORTS: Tuple[str, ...] = ("NFL", "NBA", "MLB")
SOURCES: Tuple[str, ...] = ("Model (EWMA mean)", "My mean", "My median")
DEVIG_METHODS: Tuple[str, ...] = ("Multiplicative", "Additive", "Power")
DEFAULTS: Dict[str, Tuple[str, float]] = {
    "NFL": ("Stefon Diggs", 5.5),
    "NBA": ("Jayson Tatum", 26.5),
    "MLB": ("Tarik Skubal", 6.5),
}


@dataclass(frozen=True)
class StatSpec:
    """Stat definition: data column, distribution family, variance prior, MLB group."""

    col: str
    family: str  # "count" (variance/mean ratio prior) or "continuous" (CV prior)
    prior: float
    group: str = ""  # MLB only: "pitching" or "hitting"


STATS: Dict[str, Dict[str, StatSpec]] = {
    "NFL": {
        "Receptions": StatSpec("receptions", "count", 1.3),
        "Receiving Yards": StatSpec("receiving_yards", "continuous", 0.70),
        "Targets": StatSpec("targets", "count", 1.3),
        "Rushing Attempts": StatSpec("carries", "count", 1.5),
        "Rushing Yards": StatSpec("rushing_yards", "continuous", 0.75),
        "Pass Completions": StatSpec("completions", "count", 2.0),
        "Passing Yards": StatSpec("passing_yards", "continuous", 0.30),
        "Passing TDs": StatSpec("passing_tds", "count", 1.0),
    },
    "NBA": {
        "Points": StatSpec("points", "count", 2.3),
        "Rebounds": StatSpec("rebounds", "count", 1.6),
        "Assists": StatSpec("assists", "count", 1.7),
        "3-Pointers Made": StatSpec("threes", "count", 1.2),
        "Pts + Reb + Ast": StatSpec("pts_reb_ast", "count", 2.6),
        "Steals": StatSpec("steals", "count", 1.1),
        "Blocks": StatSpec("blocks", "count", 1.2),
    },
    "MLB": {
        "Pitcher Strikeouts": StatSpec("p_strikeouts", "count", 1.5, "pitching"),
        "Pitcher Outs Recorded": StatSpec("p_outs", "count", 1.3, "pitching"),
        "Pitcher Earned Runs": StatSpec("p_earned_runs", "count", 1.8, "pitching"),
        "Pitcher Hits Allowed": StatSpec("p_hits_allowed", "count", 1.4, "pitching"),
        "Pitcher Walks Allowed": StatSpec("p_walks_allowed", "count", 1.2, "pitching"),
        "Batter Hits": StatSpec("b_hits", "count", 0.85, "hitting"),
        "Batter Total Bases": StatSpec("b_total_bases", "count", 1.9, "hitting"),
        "Batter Hits + Runs + RBIs": StatSpec("b_hrr", "count", 1.5, "hitting"),
        "Batter Home Runs": StatSpec("b_home_runs", "count", 0.95, "hitting"),
        "Batter RBIs": StatSpec("b_rbis", "count", 1.6, "hitting"),
        "Batter Runs": StatSpec("b_runs", "count", 1.1, "hitting"),
        "Batter Stolen Bases": StatSpec("b_stolen_bases", "count", 1.0, "hitting"),
        "Batter Strikeouts": StatSpec("b_strikeouts", "count", 0.95, "hitting"),
    },
}


def stat_options(league: str) -> List[str]:
    """Return the stat labels available for a league."""
    return list(STATS[league.upper()].keys())


def default_season(league: str) -> int:
    """Return the current season value for a league (NBA = season start year)."""
    return {"NFL": current_nfl_season, "NBA": nba_season_start, "MLB": mlb_season}[league.upper()]()


# --------------------------------------------------------------------------- #
# Odds utilities + devig
# --------------------------------------------------------------------------- #
def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    return 1.0 + american / 100.0 if american > 0 else 1.0 + 100.0 / abs(american)


def implied_prob(american: float) -> float:
    """Return the vig-inclusive implied probability of American odds."""
    return 1.0 / american_to_decimal(american)


def prob_to_american(p: float) -> float:
    """Convert a probability to fair American odds."""
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return -100.0 * p / (1.0 - p) if p >= 0.5 else 100.0 * (1.0 - p) / p


def devig_over_prob(over_odds: float, under_odds: float, method: str) -> float:
    """Return the vig-free Over probability via multiplicative, additive, or power devig."""
    q_o, q_u = implied_prob(over_odds), implied_prob(under_odds)
    total = q_o + q_u
    if method == "Additive":
        overround = (total - 1.0) / 2.0
        p_o = float(np.clip(q_o - overround, 0.001, 0.999))
        p_u = float(np.clip(q_u - overround, 0.001, 0.999))
        return p_o / (p_o + p_u)
    if method == "Power" and total > 1.0 + 1e-9:
        k = brentq(lambda e: q_o ** e + q_u ** e - 1.0, 1.0, 100.0)
        return float(q_o ** k)
    return q_o / total  # multiplicative (also the fallback)


# --------------------------------------------------------------------------- #
# Distribution + projection model
# --------------------------------------------------------------------------- #
@dataclass
class Projection:
    """Fitted projection: mean, variance, family, sample size, and dispersion parameter."""

    mean: float
    var: float
    family: str
    n_games: int
    disp: float  # count: variance/mean ratio; continuous: coefficient of variation


def var_from_mean(mu: float, family: str, disp: float) -> float:
    """Variance implied by a mean: ratio*mu for counts, (CV*mu)^2 for continuous stats."""
    return max(disp * mu if family == "count" else (disp * mu) ** 2, 1e-6)


def build_dist(mu: float, var: float, family: str) -> Any:
    """Return a frozen scipy distribution: Binomial / Poisson / NegBin (counts) or Gamma."""
    if family == "count":
        if var < mu * 0.999:  # under-dispersed -> Binomial
            n_trials = int(round(mu ** 2 / (mu - var)))
            n_trials = int(np.clip(max(n_trials, np.ceil(mu)), 1, 10_000))
            return stats.binom(n_trials, min(mu / n_trials, 1.0))
        if var <= mu * 1.0001:
            return stats.poisson(mu)
        r = mu ** 2 / (var - mu)
        return stats.nbinom(r, r / (r + mu))
    return stats.gamma(mu ** 2 / var, scale=var / mu)


def mid_cdf(dist: Any, x: float, family: str) -> float:
    """CDF at x; counts use the mid-point version P(X<x) + 0.5*P(X=x) for integer x."""
    if family != "count":
        return float(dist.cdf(x))
    if float(x).is_integer():
        return float(dist.cdf(x - 1) + 0.5 * dist.pmf(x))
    return float(dist.cdf(np.floor(x)))


def solve_mean_from_median(target: float, family: str, disp: float) -> float:
    """Find the mean whose distribution has the given median (Brent root-find on mid-CDF)."""

    def gap(mu: float) -> float:
        dist = build_dist(mu, var_from_mean(mu, family, disp), family)
        return mid_cdf(dist, target, family) - 0.5

    lo, hi = max(target * 0.05, 1e-3), max(target * 5.0, 10.0)
    try:
        return float(brentq(gap, lo, hi))
    except ValueError:
        return max(target, 1e-3)


def ewma_moments(x: np.ndarray, decay: float) -> Tuple[float, float]:
    """Return EWMA mean and reliability-weighted variance (newest game weighs most)."""
    n = len(x)
    w = decay ** np.arange(n - 1, -1, -1, dtype=float)
    w /= w.sum()
    mu = float(w @ x)
    denom = 1.0 - float(w @ w)
    var = float(w @ (x - mu) ** 2) / denom if n > 1 and denom > 1e-9 else mu
    return mu, max(var, 1e-6)


def weighted_median(x: np.ndarray, decay: float) -> float:
    """EWMA-weighted median: first sorted value where cumulative weight reaches 50%."""
    w = decay ** np.arange(len(x) - 1, -1, -1, dtype=float)
    order = np.argsort(x)
    cum = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cum, 0.5 * cum[-1])])


def build_projection(
    values: np.ndarray,
    decay: float,
    matchup_mult: float,
    spec: StatSpec,
    source: str,
    user_value: float,
) -> Projection:
    """Fit EWMA moments, shrink variance to the stat prior, then apply mean/median source."""
    n = len(values)
    mu0, s2 = ewma_moments(values, decay)
    mu0 = max(mu0, 1e-3)

    prior_var = spec.prior * mu0 if spec.family == "count" else (spec.prior * mu0) ** 2
    s2 = (n * s2 + PRIOR_STRENGTH_GAMES * prior_var) / (n + PRIOR_STRENGTH_GAMES)
    disp = s2 / mu0 if spec.family == "count" else float(np.sqrt(s2)) / mu0

    if source == "My mean":
        mu = max(user_value, 1e-3)
    elif source == "My median":
        mu = solve_mean_from_median(max(user_value, 0.0), spec.family, disp)
    else:
        mu = mu0 * matchup_mult
    mu = max(mu, 1e-3)
    return Projection(mu, var_from_mean(mu, spec.family, disp), spec.family, n, disp)


def simulate(proj: Projection, n_sims: int, seed: int) -> np.ndarray:
    """Draw vectorized Monte Carlo outcomes from the fitted distribution."""
    dist = build_dist(proj.mean, proj.var, proj.family)
    return np.asarray(dist.rvs(size=n_sims, random_state=seed), dtype=float)


# --------------------------------------------------------------------------- #
# Pricing (model/market blend + push-aware Kelly)
# --------------------------------------------------------------------------- #
def evaluate_side(
    sims: np.ndarray,
    line: float,
    side: str,
    american: float,
    market_prob: float,
    weight: float,
    kelly_fraction: float,
    kelly_cap: float,
    bankroll: float,
) -> Dict[str, float]:
    """Price one side using a model/market blended probability; push-aware EV and Kelly."""
    over = side == "Over"
    win = float(np.mean(sims > line) if over else np.mean(sims < line))
    lose = float(np.mean(sims < line) if over else np.mean(sims > line))
    push = float(np.mean(sims == line))

    decisive = win + lose
    model_p = win / decisive if decisive > 0 else 0.5
    blend_p = weight * model_p + (1.0 - weight) * market_prob
    win_b, lose_b = blend_p * decisive, (1.0 - blend_p) * decisive

    b = american_to_decimal(american) - 1.0
    ev = win_b * b - lose_b
    raw_kelly = ev / (b * decisive) if decisive > 0 else 0.0
    kelly = min(max(raw_kelly, 0.0) * kelly_fraction, kelly_cap)

    return {
        "Side": side,
        "Win %": win * 100,
        "Push %": push * 100,
        "Model %": model_p * 100,
        "Market %": market_prob * 100,
        "Blend %": blend_p * 100,
        "Fair Odds": prob_to_american(blend_p),
        "Book Odds": american,
        "Edge %": (blend_p - market_prob) * 100,
        "Model Edge %": (model_p - market_prob) * 100,
        "EV %": ev * 100,
        "Kelly %": kelly * 100,
        "Stake $": kelly * bankroll,
    }


# --------------------------------------------------------------------------- #
# Backtest (walk-forward, proxy lines)
# --------------------------------------------------------------------------- #
def pseudo_line(history: np.ndarray) -> float:
    """Proxy market line: floor of the recent median plus 0.5 (avoids pushes)."""
    return float(np.floor(np.median(history[-PSEUDO_LINE_WINDOW:])) + 0.5)


def backtest_row(
    values: np.ndarray, labels: np.ndarray, t: int, decay: float, max_games: int, spec: StatSpec
) -> Dict[str, Any]:
    """Predict game t from games before it only; return the prediction and outcome."""
    hist = values[max(0, t - max_games):t]
    line = pseudo_line(hist)
    proj = build_projection(hist, decay, 1.0, spec, SOURCES[0], 0.0)
    p_over = float(build_dist(proj.mean, proj.var, proj.family).sf(line))
    return {
        "Game": labels[t],
        "Proxy Line": line,
        "P(Over)": p_over,
        "Actual": float(values[t]),
        "Over": float(values[t] > line),
    }


def walk_forward(
    values: np.ndarray, labels: np.ndarray, decay: float, max_games: int, spec: StatSpec
) -> pd.DataFrame:
    """Run the walk-forward prediction for every game after the warm-up window."""
    rows = [
        backtest_row(values, labels, t, decay, max_games, spec)
        for t in range(MIN_TRAIN, len(values))
    ]
    return pd.DataFrame(rows)


def backtest_summary(bt: pd.DataFrame) -> Dict[str, float]:
    """Brier score, log loss, side hit rate, and predicted vs actual Over rate."""
    p = bt["P(Over)"].clip(1e-6, 1 - 1e-6).to_numpy()
    y = bt["Over"].to_numpy()
    decisive = p != 0.5
    hit = float(np.mean((p[decisive] > 0.5) == (y[decisive] == 1))) if decisive.any() else float("nan")
    return {
        "n": float(len(bt)),
        "brier": float(np.mean((p - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "hit": hit,
        "avg_pred": float(p.mean()),
        "actual": float(y.mean()),
    }


def calibration_table(bt: pd.DataFrame) -> pd.DataFrame:
    """Bucket predicted P(Over) and compare with the realized Over rate."""
    buckets = pd.cut(
        bt["P(Over)"], [0.0, 0.35, 0.5, 0.65, 1.0], include_lowest=True,
        labels=["0-35%", "35-50%", "50-65%", "65-100%"],
    )
    grouped = bt.groupby(buckets, observed=True).agg(
        Games=("Over", "size"), Predicted=("P(Over)", "mean"), Actual=("Over", "mean")
    )
    grouped.index = grouped.index.astype(str)
    grouped.index.name = "Bucket"
    return grouped.reset_index()


# --------------------------------------------------------------------------- #
# Data loading (common schema: player, season, label, opponent, <stat columns>)
# --------------------------------------------------------------------------- #
def _load_nfl(player: str, season: Optional[int]) -> Tuple[pd.DataFrame, Optional[str]]:
    """Fetch NFL logs via nfl_fetcher and map them to the common schema."""
    logs, note = fetch_nfl_player_logs(player, season)
    if logs.empty:
        return logs, note
    if "season_type" in logs.columns:
        logs = logs[logs["season_type"] == "REG"]
    logs = logs.assign(
        label="W" + logs["week"].astype(int).astype(str),
        player=logs["player_display_name"] if "player_display_name" in logs.columns else player,
        opponent=logs["opponent_team"] if "opponent_team" in logs.columns else "",
        source="nflverse",
    )
    return logs.reset_index(drop=True), note


def load_game_logs(
    league: str, player: str, season: Optional[int], group: str = ""
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Dispatch to the per-sport fetcher. Empty DataFrame = error text; else an optional note."""
    if league == "NFL":
        return _load_nfl(player, season)
    if league == "NBA":
        return fetch_nba_player_logs(player, season)
    return fetch_mlb_player_logs(player, group, season)


def _prepare_logs(
    league: str, player_name: str, spec: StatSpec, season: Optional[int]
) -> Tuple[Optional[pd.DataFrame], Optional[str], List[str]]:
    """Load logs, resolve multiple name matches, and drop games missing the stat."""
    warnings: List[str] = []
    logs, note = load_game_logs(league, player_name, season, spec.group)
    if logs.empty:
        return None, note or "No data returned.", warnings
    if note:
        warnings.append(note)
    if "player" in logs.columns and logs["player"].nunique() > 1:
        counts = logs["player"].value_counts()
        chosen = str(counts.idxmax())
        warnings.append(
            f"Multiple players matched ({', '.join(map(str, counts.index[:4]))}); using {chosen}. "
            "Enter the full name to be specific."
        )
        logs = logs[logs["player"] == chosen]
    if spec.col not in logs.columns:
        return None, f"'{spec.col}' is not available in this dataset.", warnings
    logs = logs.dropna(subset=[spec.col]).reset_index(drop=True)
    if logs.empty:
        return None, "No games with data this season yet for this player and stat.", warnings
    return logs, None, warnings


def _side_fields(prefix: str, r: Dict[str, float]) -> Dict[str, Any]:
    """Flatten one priced side into prefixed result keys (over_* / under_*)."""
    return {
        f"{prefix}_win_pct": r["Win %"],
        f"{prefix}_push_pct": r["Push %"],
        f"{prefix}_model_pct": r["Model %"],
        f"{prefix}_market_pct": r["Market %"],
        f"{prefix}_blend_pct": r["Blend %"],
        f"{prefix}_fair_odds": int(round(r["Fair Odds"])),
        f"{prefix}_edge": r["Edge %"],
        f"{prefix}_model_edge": r["Model Edge %"],
        f"{prefix}_ev": r["EV %"],
        f"{prefix}_kelly_pct": r["Kelly %"],
        f"{prefix}_stake": r["Stake $"],
    }


# --------------------------------------------------------------------------- #
# Public API used by the Streamlit app
# --------------------------------------------------------------------------- #
def evaluate_prop_multisport(
    league: str,
    player_name: str,
    stat_category: str,
    line: float,
    over_odds: float,
    under_odds: float,
    projection_source: str = SOURCES[0],
    custom_val: float = 0.0,
    ewma_decay: float = 0.80,
    max_recent_games: int = 10,
    matchup_multiplier: float = 1.0,
    season: Optional[int] = None,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    kelly_cap_pct: float = 3.0,
    devig_method: str = DEVIG_METHODS[0],
    model_weight: float = 0.5,
    n_sims: int = 50_000,
    seed: int = 42,
    edge_warn_pts: float = 8.0,
) -> Dict[str, Any]:
    """Evaluate one prop. Returns a result dict, or {"error": message} on any failure."""
    league = league.upper()
    if league not in STATS:
        return {"error": f"Unsupported league '{league}'."}
    if stat_category not in STATS[league]:
        return {"error": f"Stat '{stat_category}' is not supported for {league}."}
    spec = STATS[league][stat_category]

    try:
        logs_full, err, warnings = _prepare_logs(league, player_name, spec, season)
        if logs_full is None:
            return {"error": err}

        logs = logs_full.tail(int(max_recent_games)).reset_index(drop=True)
        values = logs[spec.col].to_numpy(dtype=float)
        labels = (
            logs["label"].astype(str) + " " + logs["opponent"].fillna("").astype(str)
        ).str.strip().tolist()
        shown_name = str(logs["player"].iloc[0]) if "player" in logs.columns else player_name

        proj = build_projection(
            values, ewma_decay, matchup_multiplier, spec, projection_source, custom_val
        )
        sims = simulate(proj, int(n_sims), int(seed))

        mkt_over = devig_over_prob(over_odds, under_odds, devig_method)
        cap = kelly_cap_pct / 100.0
        over = evaluate_side(
            sims, line, "Over", over_odds, mkt_over, model_weight, kelly_fraction, cap, bankroll
        )
        under = evaluate_side(
            sims, line, "Under", under_odds, 1.0 - mkt_over, model_weight, kelly_fraction, cap, bankroll
        )
        best = over if over["EV %"] >= under["EV %"] else under
        best_key = "over" if best is over else "under"

        if len(values) < LOW_SAMPLE_GAMES and not any("sample" in w.lower() for w in warnings):
            warnings.append(
                f"Small sample: only {len(values)} game(s) used. Treat the projection as low confidence."
            )
        if projection_source != SOURCES[0]:
            warnings.append(
                f"Using your {projection_source.replace('My ', '')} as the projection. Recent games "
                "only set the spread, and the matchup multiplier is not applied."
            )
        if best["Model Edge %"] > edge_warn_pts:
            warnings.append(
                f"Model edge of {best['Model Edge %']:.1f} pts over the market exceeds your "
                f"{edge_warn_pts:.1f}-pt threshold. Gaps this large are more often a model miss "
                "(role change, injury, small or stale sample) than a real edge."
            )

        sim_median = float(np.median(sims))
        result: Dict[str, Any] = {
            "player_name": shown_name,
            "league": league,
            "stat_category": stat_category,
            "line": float(line),
            "projection_mean": proj.mean,
            "projection_median": sim_median,
            "best_side": f"{best['Side']} {line:g}",
            "best_side_key": best_key,
            "fair_odds": int(round(best["Fair Odds"])),
            "edge_vs_market": best["Edge %"],
            "model_edge": best["Model Edge %"],
            "ev_pct": best["EV %"],
            "kelly_pct": best["Kelly %"],
            "kelly_stake_dollars": best["Stake $"],
            "simulations": sims,
            "recent_games": values.tolist(),
            "game_labels": labels,
            "family": spec.family,
            "data_source": str(logs["source"].iloc[-1]) if "source" in logs.columns else league,
            "n_games": int(len(values)),
            "variance": proj.var,
            "devig_method": devig_method,
            "model_weight": float(model_weight),
            "market_over_prob": mkt_over,
            "recent_weighted_median": weighted_median(values, ewma_decay),
            "recent_simple_median": float(np.median(values)),
            "recent_simple_mean": float(values.mean()),
            "warnings": warnings,
        }
        result.update(_side_fields("over", over))
        result.update(_side_fields("under", under))
        return result
    except Exception as exc:
        return {"error": f"Evaluation failed: {exc}"}


def run_backtest(
    league: str,
    player_name: str,
    stat_category: str,
    season: Optional[int] = None,
    ewma_decay: float = 0.80,
    max_recent_games: int = 10,
    n_games: int = 30,
) -> Dict[str, Any]:
    """Walk-forward backtest on one player's logs. Returns summary/tables or {"error": ...}."""
    league = league.upper()
    if league not in STATS or stat_category not in STATS[league]:
        return {"error": "Unsupported league or stat."}
    spec = STATS[league][stat_category]
    try:
        logs, err, _ = _prepare_logs(league, player_name, spec, season)
        if logs is None:
            return {"error": err}
        sub = logs.tail(int(n_games))
        values = sub[spec.col].to_numpy(dtype=float)
        labels = sub["label"].astype(str).to_numpy()
        if len(values) < MIN_TRAIN + 3:
            return {
                "error": f"Need at least {MIN_TRAIN + 3} games for a backtest; "
                f"only {len(values)} available."
            }
        bt = walk_forward(values, labels, ewma_decay, int(max_recent_games), spec)
        return {
            "summary": backtest_summary(bt),
            "calibration": calibration_table(bt),
            "table": bt,
            "n_games": int(len(values)),
        }
    except Exception as exc:
        return {"error": f"Backtest failed: {exc}"}

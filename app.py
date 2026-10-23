"""Multi-sport (NFL/NBA/MLB) prop evaluator: EWMA mean/median, blended pricing, backtest."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from scipy.optimize import brentq

from nfl_fetcher import current_nfl_season, fetch_nfl_player_logs
from sports_data import (
    fetch_mlb_player_logs,
    fetch_nba_player_logs,
    mlb_season,
    nba_season_start,
)

try:  # Optional: Sleeper injury/depth info (NFL only) if sleeper_client.py exists.
    from sleeper_client import find_player

    HAS_SLEEPER = True
except Exception:
    HAS_SLEEPER = False

PRIOR_STRENGTH_GAMES: float = 4.0  # pseudo-games of shrinkage toward the variance prior
PSEUDO_LINE_WINDOW: int = 8        # backtest proxy line = floor(median of last N) + 0.5
MIN_TRAIN: int = 5                 # minimum games of history before a backtest prediction
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


def fmt_american(x: float) -> str:
    """Format American odds with an explicit sign."""
    return f"{x:+.0f}"


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
# Pricing (model/market blend + Kelly with pushes)
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

    dec = american_to_decimal(american)
    b = dec - 1.0
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
    buckets = pd.cut(bt["P(Over)"], [0.0, 0.35, 0.5, 0.65, 1.0], include_lowest=True)
    grouped = bt.groupby(buckets, observed=True).agg(
        Games=("Over", "size"), Predicted=("P(Over)", "mean"), Actual=("Over", "mean")
    )
    grouped.index = grouped.index.astype(str)
    grouped.index.name = "Bucket"
    return grouped.reset_index()


# --------------------------------------------------------------------------- #
# Data loading (common schema: player, season, label, opponent, <stat columns>)
# --------------------------------------------------------------------------- #
def load_nfl(player: str, season: int) -> Tuple[pd.DataFrame, Optional[str]]:
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
    )
    return logs.reset_index(drop=True), note


@st.cache_data(ttl=1800, show_spinner="Fetching game logs...")
def load_logs(sport: str, name: str, season: int, group: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Cached dispatcher to the per-sport fetchers."""
    if sport == "NFL":
        return load_nfl(name, season)
    if sport == "NBA":
        return fetch_nba_player_logs(name, season)
    return fetch_mlb_player_logs(name, group, season)


@st.cache_data(ttl=3600)
def sleeper_status(name: str) -> Optional[str]:
    """Return a one-line Sleeper injury/depth summary (NFL), or None if unavailable."""
    if not HAS_SLEEPER:
        return None
    try:
        row = find_player(name)
        if row.empty:
            return None
        r = row.iloc[0]
        inj = r["injury_status"] if pd.notna(r["injury_status"]) else "No designation"
        return f"Sleeper: {r['full_name']} ({r['team']}, {r['position']}) | Injury: {inj}"
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def outcome_chart(sims: np.ndarray, line: float, family: str, label: str) -> go.Figure:
    """Plot the simulated outcome distribution with the book line and median marked."""
    fig = go.Figure()
    if family == "count":
        counts = np.bincount(sims.astype(int))
        x = np.arange(len(counts))
        colors = np.where(x > line, "#22c55e", np.where(x < line, "#ef4444", "#9ca3af"))
        fig.add_bar(x=x, y=counts / len(sims), marker_color=colors.tolist())
    else:
        fig.add_histogram(x=sims, nbinsx=50, histnorm="probability", marker_color="#3b82f6")
    fig.add_vline(x=line, line_dash="dash", line_color="white", annotation_text=f"Line {line}")
    fig.add_vline(
        x=float(np.median(sims)), line_dash="dot", line_color="#f59e0b",
        annotation_text="Median", annotation_position="bottom right",
    )
    fig.update_layout(
        title=f"Simulated {label} distribution",
        xaxis_title=label, yaxis_title="Probability", showlegend=False,
    )
    return fig


def logs_chart(logs: pd.DataFrame, col: str, line: float, label: str) -> go.Figure:
    """Bar chart of recent game results versus the book line."""
    colors = np.where(logs[col] > line, "#22c55e", "#ef4444")
    fig = go.Figure(go.Bar(x=logs["label"].astype(str), y=logs[col], marker_color=colors.tolist()))
    fig.add_hline(y=line, line_dash="dash", line_color="white")
    fig.update_layout(title=f"Recent {label}", xaxis_title="Game", yaxis_title=label)
    return fig


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def render_backtest(logs_full: pd.DataFrame, spec: StatSpec, decay: float, max_games: int) -> None:
    """Render the walk-forward backtest expander for the current player and stat."""
    with st.expander("Backtest (walk-forward on this player's logs)"):
        st.caption(
            "Each game is predicted using only earlier games, against a proxy line "
            f"(floor of the median of the previous {PSEUDO_LINE_WINDOW} games + 0.5), not real "
            "market lines. This checks whether the probabilities are calibrated, not whether "
            "the bets would have made money. One player's logs are a small sample, so treat "
            "the results as a rough sanity check."
        )
        bt_games = st.slider("Games to test over (most recent)", 10, 60, 30)
        if not st.checkbox("Run backtest"):
            return
        sub = logs_full.tail(bt_games)
        values = sub[spec.col].to_numpy(dtype=float)
        labels = sub["label"].astype(str).to_numpy()
        if len(values) < MIN_TRAIN + 3:
            st.info(f"Need at least {MIN_TRAIN + 3} games for a backtest; only {len(values)} available.")
            return

        bt = walk_forward(values, labels, decay, max_games, spec)
        s = backtest_summary(bt)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Games tested", f"{int(s['n'])}")
        c2.metric("Brier (0.250 = coin flip)", f"{s['brier']:.3f}")
        c3.metric("Log loss (0.693 = coin flip)", f"{s['logloss']:.3f}")
        c4.metric("Side hit rate", f"{s['hit']:.1%}" if not np.isnan(s["hit"]) else "n/a")
        c5.metric("Avg predicted vs actual Over", f"{s['avg_pred']:.0%} / {s['actual']:.0%}")
        st.dataframe(calibration_table(bt), use_container_width=True, hide_index=True)
        st.dataframe(
            bt.style.format({"Proxy Line": "{:.1f}", "P(Over)": "{:.1%}", "Actual": "{:.1f}", "Over": "{:.0f}"}),
            use_container_width=True, hide_index=True,
        )


def main() -> None:
    """Render the multi-sport Streamlit prop evaluator."""
    st.set_page_config(page_title="Prop Evaluator", layout="wide")
    st.title("Prop Evaluator")

    sport = st.sidebar.radio("Sport", SPORTS, horizontal=True)
    specs = STATS[sport]
    default_player, default_line = DEFAULTS[sport]
    default_season = {
        "NFL": current_nfl_season(),
        "NBA": nba_season_start(),
        "MLB": mlb_season(),
    }[sport]

    with st.sidebar.form(f"inputs_{sport}"):
        st.header("Prop")
        player = st.text_input("Player", value=default_player, key=f"player_{sport}")
        season = st.number_input(
            "Season" if sport != "NBA" else "Season start year (2026 = 2026-27)",
            1999, 2100, default_season, step=1, key=f"season_{sport}",
        )
        stat_label = st.selectbox("Stat", list(specs.keys()), key=f"stat_{sport}")
        line = st.number_input("Line", value=default_line, step=0.5, key=f"line_{sport}")
        over_odds = st.number_input("Over odds (American)", value=-110, step=5, key=f"over_{sport}")
        under_odds = st.number_input("Under odds (American)", value=-110, step=5, key=f"under_{sport}")

        st.header("Model")
        source = st.selectbox("Projection source", SOURCES)
        user_value = st.number_input(
            "Your mean / median (used only if 'My mean' or 'My median')",
            min_value=0.0, value=float(default_line), step=0.1, key=f"uval_{sport}",
        )
        decay = st.slider("EWMA decay (lower = more recent weight)", 0.50, 1.00, 0.80, 0.01)
        max_games = st.slider("Max recent games", 3, 25, 10)
        matchup = st.slider("Matchup multiplier (Model source only)", 0.70, 1.30, 1.00, 0.01)
        n_sims = st.select_slider("Simulations", [10_000, 25_000, 50_000, 100_000], value=50_000)
        seed = st.number_input("Seed", value=42, step=1)

        st.header("Pricing")
        devig_method = st.selectbox("Devig method", DEVIG_METHODS)
        weight = st.slider("Model weight (1.0 = trust model fully)", 0.0, 1.0, 0.5, 0.05)
        edge_flag = st.slider("Warn if model edge exceeds (pts)", 3.0, 20.0, 8.0, 0.5)

        st.header("Bankroll")
        bankroll = st.number_input("Bankroll ($)", value=1000.0, step=50.0)
        kelly_frac = st.slider("Kelly fraction", 0.05, 1.00, 0.25, 0.05)
        kelly_cap = st.slider("Max allocation (% of bankroll)", 0.5, 10.0, 3.0, 0.5) / 100.0

        if st.form_submit_button("Run Simulation"):
            st.session_state["ran_sport"] = sport

    if st.session_state.get("ran_sport") != sport:
        st.info(f"Enter a {sport} prop in the sidebar and click **Run Simulation**.")
        st.stop()

    spec = specs[stat_label]
    logs, note = load_logs(sport, player, int(season), spec.group)
    if logs.empty:
        st.error(note or "No data returned.")
        st.stop()
    if note:
        st.warning(note)

    if "player" in logs.columns and logs["player"].nunique() > 1:
        pick = st.selectbox("Multiple players matched", sorted(logs["player"].unique()))
        logs = logs[logs["player"] == pick]
    shown_name = str(logs["player"].iloc[0]) if "player" in logs.columns else player

    if spec.col not in logs.columns:
        st.error(f"'{spec.col}' is not available in this dataset.")
        st.stop()
    logs_full = logs.dropna(subset=[spec.col]).reset_index(drop=True)
    logs = logs_full.tail(max_games).reset_index(drop=True)
    if len(logs) < 2:
        st.error("Need at least 2 games with data to build a projection.")
        st.stop()

    if sport == "NFL":
        status = sleeper_status(player)
        if status:
            st.caption(status)

    values = logs[spec.col].to_numpy(dtype=float)
    proj = build_projection(values, decay, matchup, spec, source, user_value)
    sims = simulate(proj, int(n_sims), int(seed))
    sim_median = float(np.median(sims))

    mkt_over = devig_over_prob(over_odds, under_odds, devig_method)
    results = pd.DataFrame(
        [
            evaluate_side(sims, line, "Over", over_odds, mkt_over, weight, kelly_frac, kelly_cap, bankroll),
            evaluate_side(sims, line, "Under", under_odds, 1 - mkt_over, weight, kelly_frac, kelly_cap, bankroll),
        ]
    )
    best = results.loc[results["EV %"].idxmax()]

    st.subheader(f"{shown_name} | {stat_label} | {sport}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Projection (mean)", f"{proj.mean:.2f}")
    m2.metric("Projection (median)", f"{sim_median:.2f}")
    m3.metric("Best Side", f"{best['Side']} {line}")
    m4.metric("Fair Odds (blended)", fmt_american(best["Fair Odds"]))
    m5.metric("Edge vs market", f"{best['Edge %']:+.1f}%")
    m6.metric("EV", f"{best['EV %']:+.1f}%")
    st.metric("Kelly Stake", f"${best['Stake $']:.2f} ({best['Kelly %']:.2f}% of bankroll)")
    st.caption(
        f"{best['Side']}: model {best['Model %']:.1f}% | market ({devig_method} devig) "
        f"{best['Market %']:.1f}% | blended {best['Blend %']:.1f}% at model weight {weight:.2f}. "
        f"Recent games ({len(values)}): EWMA-weighted median {weighted_median(values, decay):.2f} | "
        f"simple median {float(np.median(values)):.2f} | simple mean {float(values.mean()):.2f}"
    )
    if best["Model Edge %"] > edge_flag:
        st.warning(
            f"Model edge of {best['Model Edge %']:.1f} pts over the market exceeds your "
            f"{edge_flag:.1f}-pt threshold. Gaps this large are more often a model miss "
            "(role change, injury, small or stale sample) than a real edge. Check the inputs "
            "before betting."
        )

    display = results.copy()
    display["Fair Odds"] = display["Fair Odds"].map(fmt_american)
    display["Book Odds"] = display["Book Odds"].map(fmt_american)
    st.dataframe(
        display.style.format(
            {"Win %": "{:.1f}", "Push %": "{:.1f}", "Model %": "{:.1f}", "Market %": "{:.1f}",
             "Blend %": "{:.1f}", "Edge %": "{:+.1f}", "Model Edge %": "{:+.1f}",
             "EV %": "{:+.1f}", "Kelly %": "{:.2f}", "Stake $": "${:.2f}"}
        ),
        use_container_width=True, hide_index=True,
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(outcome_chart(sims, line, spec.family, stat_label), use_container_width=True)
    c2.plotly_chart(logs_chart(logs, spec.col, line, stat_label), use_container_width=True)

    render_backtest(logs_full, spec, decay, max_games)

    with st.expander("Model details"):
        dist = "Binomial / Poisson / Negative Binomial" if spec.family == "count" else "Gamma"
        st.write(
            f"Source: {source} | Games used: {proj.n_games} | Mean: {proj.mean:.2f} | "
            f"Sim median: {sim_median:.2f} | Variance: {proj.var:.2f} | Distribution: {dist} | "
            f"Devigged market Over prob ({devig_method}): {mkt_over:.1%}"
        )
        cols = [c for c in ("season", "label", "opponent", spec.col) if c in logs.columns]
        st.dataframe(logs[cols], use_container_width=True)


if __name__ == "__main__":
    main()

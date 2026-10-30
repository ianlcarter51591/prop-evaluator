"""Live NFL game-script adjustment for player props (ESPN public endpoints, no API key).

Math (all assumptions are named constants you can tune or backtest):
  1. Game script lean   : lean = tanh(dev / (MARGIN_SD * sqrt(r)))
        dev = score margin + spread * (1 - r)   (spread < 0 means your team was the favorite)
        r   = fraction of regulation left; lean in (-1, 1), + = leading, - = trailing.
        The sqrt(r) scaling makes a 7-point lead matter far more in Q4 than in Q1.
  2. Remaining volume   : mu_rem = mu_full * r * (1 + beta_stat * lean)
        beta < 0 for pass-driven stats (trailing -> more throws), beta > 0 for rushing.
  3. Remaining spread   : sd_rem = cv_stat * mu_full * sqrt(r) * (1 + beta_stat * lean)
        (variance accrues roughly linearly with time).
  4. Final stat         : current + Remaining, Remaining ~ Gamma (yards) or NegBin/Poisson (counts).
  5. P(over), P(under), P(push) -> EV at the offered prices, devigged edge, 1/4 Kelly.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import requests
import streamlit as st

ESPN = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
GAME_SECONDS = 3600.0
MARGIN_SD = 13.5          # std dev of an NFL final margin (points)
PLAY_EV_PCT = 2.0         # EV% needed to upgrade LEAN -> PLAY
N_SIMS = 50_000
KELLY_FRACTION = 0.25
KELLY_CAP_PCT = 5.0

# nflverse / books -> one canonical abbreviation (ESPN uses WSH, LAR, JAX...)
_ABBR = {"WSH": "WAS", "LA": "LAR", "JAC": "JAX", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
         "SD": "LAC", "STL": "LAR", "OAK": "LV"}


@dataclass(frozen=True)
class StatSpec:
    """How one prop stat responds to game script and where it lives in the ESPN box score."""

    group: str       # ESPN box score group: passing / rushing / receiving
    field: str       # att | cmp | yds | td | car | rec
    beta: float      # volume sensitivity to lean (lean > 0 = leading)
    cv: float        # full-game coefficient of variation
    is_yards: bool   # Gamma (True) vs count distribution (False)


SPECS: Dict[str, StatSpec] = {
    "pass_att": StatSpec("passing", "att", -0.15, 0.17, False),
    "pass_cmp": StatSpec("passing", "cmp", -0.15, 0.22, False),
    "pass_yds": StatSpec("passing", "yds", -0.08, 0.28, True),
    "pass_td": StatSpec("passing", "td", -0.03, 0.70, False),
    "rush_att": StatSpec("rushing", "car", 0.20, 0.35, False),
    "rush_yds": StatSpec("rushing", "yds", 0.15, 0.55, True),
    "rush_td": StatSpec("rushing", "td", 0.10, 1.00, False),
    "rec_cnt": StatSpec("receiving", "rec", -0.08, 0.50, False),
    "rec_yds": StatSpec("receiving", "yds", -0.08, 0.75, True),
    "rec_td": StatSpec("receiving", "td", -0.03, 1.00, False),
}
_LABEL = {"att": "C/ATT", "cmp": "C/ATT", "yds": "YDS", "td": "TD", "car": "CAR", "rec": "REC"}


@dataclass(frozen=True)
class GameState:
    """Live snapshot of the player's game from the player's team perspective."""

    event_id: str
    state: str            # pre | in | post
    detail: str           # e.g. "Q3 4:21" or "Halftime"
    remaining_s: float
    team: str
    opp: str
    team_score: int
    opp_score: int
    home: bool

    @property
    def margin(self) -> int:
        """Score margin: positive = player's team leads."""
        return self.team_score - self.opp_score

    @property
    def remaining_frac(self) -> float:
        """Fraction of regulation time left in [0, 1]."""
        return float(np.clip(self.remaining_s / GAME_SECONDS, 0.0, 1.0))


def canon(abbr: str) -> str:
    """Canonical team abbreviation."""
    a = str(abbr).upper().strip()
    return _ABBR.get(a, a)


def spec_for(stat_category: str) -> Optional[StatSpec]:
    """Map a stat label (e.g. 'Passing Yards') to a StatSpec; None if unsupported (combos, INTs, longest...)."""
    s = stat_category.lower()
    if any(t in s for t in ("interception", "long", "+", "&", "combined", "sack", "fantasy", "anytime")):
        return None
    if "reception" in s:
        return SPECS["rec_cnt"]
    if "rec" in s:
        return SPECS["rec_td"] if "td" in s else SPECS["rec_yds"] if ("yd" in s or "yard" in s) else None
    for fam, pre in (("pass", "pass"), ("rush", "rush")):
        if fam in s:
            if "td" in s:
                return SPECS[f"{pre}_td"]
            if "yd" in s or "yard" in s:
                return SPECS[f"{pre}_yds"]
            if pre == "pass" and "comp" in s:
                return SPECS["pass_cmp"]
            if "att" in s or "car" in s:
                return SPECS[f"{pre}_att"]
    return None


@st.cache_data(ttl=20, show_spinner=False)
def _get_json(url: str) -> Dict[str, Any]:
    """GET JSON with a 20s cache so a refresh button gives near-live data without hammering ESPN."""
    r = requests.get(url, timeout=8, headers={"User-Agent": "prop-evaluator"})
    r.raise_for_status()
    return r.json()


def refresh() -> None:
    """Drop the cached ESPN responses (only this module's cache)."""
    _get_json.clear()


def _clock_seconds(display: str) -> float:
    """'4:21' -> 261.0 ; '0.0' -> 0.0."""
    m = re.match(r"^(\d+):(\d+)", str(display))
    if m:
        return int(m.group(1)) * 60.0 + int(m.group(2))
    try:
        return float(display)
    except (TypeError, ValueError):
        return 0.0


def _remaining_seconds(state: str, period: int, clock_s: float) -> float:
    """Regulation seconds left (overtime counts only its own clock)."""
    if state == "pre":
        return GAME_SECONDS
    if state == "post":
        return 0.0
    if period <= 4:
        return (4 - period) * 900.0 + clock_s
    return min(clock_s, 600.0)


def find_game(team: str) -> Optional[GameState]:
    """Find the team's game on ESPN's current NFL scoreboard; None if absent or the request fails."""
    try:
        data = _get_json(f"{ESPN}/scoreboard")
    except Exception:
        return None
    want = canon(team)
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        for mine, other in (("home", "away"), ("away", "home")):
            if mine not in sides or other not in sides:
                continue
            if canon(sides[mine]["team"]["abbreviation"]) != want:
                continue
            st_ = ev["status"]
            state = st_["type"]["state"]
            period = int(st_.get("period", 0))
            clock_s = _clock_seconds(st_.get("displayClock", "0:00"))
            detail = st_["type"].get("shortDetail", "")
            return GameState(
                event_id=str(ev["id"]), state=state, detail=detail,
                remaining_s=_remaining_seconds(state, period, clock_s),
                team=want, opp=canon(sides[other]["team"]["abbreviation"]),
                team_score=int(float(sides[mine].get("score", 0) or 0)),
                opp_score=int(float(sides[other].get("score", 0) or 0)),
                home=(mine == "home"),
            )
    return None


def _norm_name(name: str) -> str:
    """Lowercase, strip punctuation and suffixes for name matching."""
    n = re.sub(r"[^a-z ]", "", name.lower().replace(".", ""))
    return re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n).strip()


def _same_player(a: str, b: str) -> bool:
    """Exact normalized match, or same last name and same first initial."""
    x, y = _norm_name(a), _norm_name(b)
    if x == y:
        return True
    xs, ys = x.split(), y.split()
    return bool(xs and ys and xs[-1] == ys[-1] and xs[0][0] == ys[0][0])


def fetch_player_stat(event_id: str, player: str, spec: StatSpec) -> Optional[float]:
    """Player's current value for the stat from the ESPN box score; None if not found / request failed."""
    try:
        data = _get_json(f"{ESPN}/summary?event={event_id}")
    except Exception:
        return None
    for team_block in data.get("boxscore", {}).get("players", []):
        for grp in team_block.get("statistics", []):
            if grp.get("name") != spec.group:
                continue
            labels = grp.get("labels", [])
            want = _LABEL[spec.field]
            if want not in labels:
                continue
            idx = labels.index(want)
            for ath in grp.get("athletes", []):
                if not _same_player(ath["athlete"]["displayName"], player):
                    continue
                raw = str(ath["stats"][idx])
                if want == "C/ATT":
                    parts = raw.split("/")
                    return float(parts[0] if spec.field == "cmp" else parts[1])
                try:
                    return float(raw)
                except ValueError:
                    return None
    return None


def script_lean(margin: float, spread: float, remaining_frac: float) -> float:
    """tanh-squashed game-script lean in (-1, 1); + = leading, - = trailing (see module docstring)."""
    r = max(remaining_frac, 0.02)
    dev = margin + spread * (1.0 - remaining_frac)
    return float(np.tanh(dev / (MARGIN_SD * np.sqrt(r))))


def _dec(american: float) -> float:
    """American -> decimal odds."""
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def _sim_remaining(mu: float, sd: float, is_yards: bool, n: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized draws of the remaining production."""
    if mu <= 1e-9:
        return np.zeros(n)
    sd = max(sd, 1e-6)
    if is_yards:
        return rng.gamma((mu / sd) ** 2, sd ** 2 / mu, n)
    if sd ** 2 > mu * 1.001:
        k = mu ** 2 / (sd ** 2 - mu)
        return rng.negative_binomial(k, k / (k + mu), n).astype(float)
    return rng.poisson(mu, n).astype(float)


def evaluate_live(mu_full: float, spec: StatSpec, game: GameState, spread: float, current: float,
                  line: float, over_odds: float, under_odds: float, seed: int = 7) -> Dict[str, Any]:
    """Re-price the prop given score, clock, and what the player has already done."""
    r, lean = game.remaining_frac, script_lean(game.margin, spread, game.remaining_frac)
    mult = float(np.clip(1.0 + spec.beta * lean, 0.5, 1.6))
    mu_rem = mu_full * r * mult
    sd_rem = spec.cv * mu_full * np.sqrt(r) * mult
    final = current + _sim_remaining(mu_rem, sd_rem, spec.is_yards, N_SIMS, np.random.default_rng(seed))
    p_over, p_under = float((final > line).mean()), float((final < line).mean())
    d_o, d_u = _dec(over_odds), _dec(under_odds)
    ev_o, ev_u = p_over * (d_o - 1) - p_under, p_under * (d_u - 1) - p_over
    imp_o, imp_u = 1 / d_o, 1 / d_u
    mkt_over = imp_o / (imp_o + imp_u)
    over_best = ev_o >= ev_u
    ev, d = (ev_o, d_o) if over_best else (ev_u, d_u)
    p_side = p_over if over_best else p_under
    p_opp = p_under if over_best else p_over
    mkt_side = mkt_over if over_best else 1 - mkt_over
    fair_p = p_side / max(p_side + p_opp, 1e-9)
    fair_dec = 1 / max(fair_p, 1e-9)
    fair_am = int(round(-100 * fair_p / (1 - fair_p))) if fair_p >= 0.5 else int(round(100 * (1 - fair_p) / fair_p))
    kelly = float(np.clip(KELLY_FRACTION * 100 * ev / (d - 1), 0.0, KELLY_CAP_PCT)) if ev > 0 else 0.0
    return {
        "lean": lean, "mult": mult, "r": r, "mu_rem": mu_rem, "current": current, "final_proj": current + mu_rem,
        "p_over": p_over, "p_under": p_under, "side": "Over" if over_best else "Under",
        "ev_pct": ev * 100, "edge_pp": (p_side - mkt_side) * 100, "fair_odds": fair_am, "fair_dec": fair_dec,
        "kelly_pct": kelly, "verdict": "PLAY" if ev * 100 >= PLAY_EV_PCT else "LEAN (PASS)",
    }

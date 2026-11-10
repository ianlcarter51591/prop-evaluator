"""NFL Monte Carlo Prop Engine & Dual-Directional 6-Point Assessment System.

Integrates EWMA baseline projections, distribution simulations (Gamma, Normal, NegBin),
two-way devigging, Kelly sizing, injury/matchup adjustments, game script/weather checks,
and dynamic Dual-Directional 6-Point criteria evaluation for both Over and Under bets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from nfl_adjust import injury_layer, tier_splits
from nfl_matchups import DEFAULT_K, _defense_games, matchup_multiplier, matchup_table, run_backtests
from sports_data import ABBR_CANON, clean_name, clean_series, current_nfl_season, fetch_nfl_player_logs, load_schedule, load_weekly, next_game

# --- Configuration & Thresholds -----------------------------------------------
DEFAULTS: Tuple[str, float] = ("Patrick Mahomes", 265.5)
SOURCES: Tuple[str, ...] = ("Model (EWMA mean)", "My mean", "My median")
DEVIG_METHODS: Tuple[str, ...] = ("Multiplicative",)

STATS: List[str] = [
    "Passing Yards", "Pass Completions", "Pass Attempts", "Passing TDs", "Interceptions",
    "Rushing Yards", "Rushing Attempts", "Receiving Yards", "Receptions",
    "Kicking Points", "Field Goals Made", "Anytime Touchdown Scorer",
]

WINDOW = 8             # Recent current-season games feeding EWMA
DECAY = 0.85           # EWMA decay per game of age
N_PRIOR = 4.0          # Pseudo-games of last-season weight
N_CV_PRIOR = 4.0       # Pseudo-games weight on stat's default spread
N_SIMS = 50_000
KELLY_FRACTION, KELLY_CAP = 0.25, 0.05

FAMILY: Dict[str, str] = {
    "PASSING YARDS": "gamma", "RUSHING YARDS": "gamma", "RECEIVING YARDS": "gamma", "KICKING POINTS": "gamma",
    "PASS COMPLETIONS": "normal", "PASS ATTEMPTS": "normal", "RUSHING ATTEMPTS": "normal",
    "PASSING TDS": "count", "RECEPTIONS": "count", "INTERCEPTIONS": "count",
    "FIELD GOALS": "count", "TOUCHDOWNS": "count",
}

DEFAULT_CV: Dict[str, float] = {
    "PASSING YARDS": 0.25, "PASS COMPLETIONS": 0.22, "PASS ATTEMPTS": 0.18, "PASSING TDS": 0.65,
    "RUSHING YARDS": 0.55, "RUSHING ATTEMPTS": 0.35, "RECEIVING YARDS": 0.65, "RECEPTIONS": 0.50,
    "INTERCEPTIONS": 1.00, "KICKING POINTS": 0.35, "FIELD GOALS": 0.65, "TOUCHDOWNS": 0.85,
}

MIN_LINE: Dict[str, float] = {
    "PASSING YARDS": 175.5, "PASS COMPLETIONS": 14.5, "PASS ATTEMPTS": 24.5, "PASSING TDS": 1.5,
    "RUSHING YARDS": 35.5, "RUSHING ATTEMPTS": 8.5, "RECEIVING YARDS": 35.5, "RECEPTIONS": 3.5,
    "INTERCEPTIONS": 0.5, "KICKING POINTS": 6.5, "FIELD GOALS": 1.5, "TOUCHDOWNS": 0.5,
}

MIN_EDGE_ABS: Dict[str, float] = {
    "PASSING YARDS": 12.0, "PASS COMPLETIONS": 1.5, "PASS ATTEMPTS": 2.0, "PASSING TDS": 0.25,
    "RUSHING YARDS": 6.0, "RUSHING ATTEMPTS": 1.5, "RECEIVING YARDS": 6.0, "RECEPTIONS": 0.75,
    "INTERCEPTIONS": 0.15, "KICKING POINTS": 1.0, "FIELD GOALS": 0.35, "TOUCHDOWNS": 0.15,
}

MIN_EDGE_PCT = 0.06
IMPLIED_MIN = 0.50
RANK_CUTOFF = 16            # Half of 32-team league
RECENT_GAMES = 4            # Recent defense filter window
RECENT_MIN_GAMES = 2
BLOWOUT_SPREAD = 10.0
WIND_MPH, GUST_MPH, PRECIP_MM, SNOW_CM, COLD_F = 15.0, 25.0, 3.0, 0.5, 20.0
WEATHER_HOURS = 3
SHARP_SCORE, LEAN_SCORE = 6, 5

ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"

STADIUMS: Dict[str, Tuple[float, float, str]] = {
    "ARI": (33.5276, -112.2626, "retractable"), "ATL": (33.7554, -84.4010, "dome"),
    "BAL": (39.2780, -76.6227, "outdoor"), "BUF": (42.7738, -78.7870, "outdoor"),
    "CAR": (35.2258, -80.8528, "outdoor"), "CHI": (41.8623, -87.6167, "outdoor"),
    "CIN": (39.0955, -84.5161, "outdoor"), "CLE": (41.5061, -81.6995, "outdoor"),
    "DAL": (32.7473, -97.0945, "retractable"), "DEN": (39.7439, -105.0201, "outdoor"),
    "DET": (42.3400, -83.0456, "dome"), "GB": (44.5013, -88.0622, "outdoor"),
    "HOU": (29.6847, -95.4107, "retractable"), "IND": (39.7601, -86.1639, "retractable"),
    "JAX": (30.3239, -81.6373, "outdoor"), "KC": (39.0489, -94.4839, "outdoor"),
    "LA": (33.9535, -118.3392, "canopy"), "LAC": (33.9535, -118.3392, "canopy"),
    "LV": (36.0909, -115.1833, "dome"), "MIA": (25.9580, -80.2389, "outdoor"),
    "MIN": (44.9735, -93.2575, "dome"), "NE": (42.0909, -71.2643, "outdoor"),
    "NO": (29.9511, -90.0812, "dome"), "NYG": (40.8135, -74.0745, "outdoor"),
    "NYJ": (40.8135, -74.0745, "outdoor"), "PHI": (39.9008, -75.1675, "outdoor"),
    "PIT": (40.4468, -80.0158, "outdoor"), "SEA": (47.5952, -122.3316, "outdoor"),
    "SF": (37.4030, -121.9700, "outdoor"), "TB": (27.9759, -82.5033, "outdoor"),
    "TEN": (36.1665, -86.7713, "outdoor"), "WAS": (38.9076, -76.8645, "outdoor"),
}

TAG_COL = "Six-Point"
NICKNAMES: Dict[str, str] = {
    "ARI": "cardinals", "ATL": "falcons", "BAL": "ravens", "BUF": "bills", "CAR": "panthers", "CHI": "bears",
    "CIN": "bengals", "CLE": "browns", "DAL": "cowboys", "DEN": "broncos", "DET": "lions", "GB": "packers",
    "HOU": "texans", "IND": "colts", "JAX": "jaguars", "KC": "chiefs", "LA": "rams", "LAC": "chargers",
    "LV": "raiders", "MIA": "dolphins", "MIN": "vikings", "NE": "patriots", "NO": "saints", "NYG": "giants",
    "NYJ": "jets", "PHI": "eagles", "PIT": "steelers", "SEA": "seahawks", "SF": "49ers", "TB": "buccaneers",
    "TEN": "titans", "WAS": "commanders",
}


# --- Dataclasses ---
@dataclass
class Projection:
    mu: float
    sigma: float
    n_current: int
    n_prior: int
    mult: float


@dataclass(frozen=True)
class Check:
    key: str
    label: str
    passed: bool
    detail: str
    verified: bool = True


@dataclass(frozen=True)
class SixPointResult:
    side: str
    score: int
    tier: str
    checks: Tuple[Check, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "side": self.side,
            "score": self.score,
            "tier": self.tier,
            "checks": [c.__dict__.copy() for c in self.checks],
        }


@dataclass(frozen=True)
class GameInfo:
    home: str
    away: str
    kickoff_iso: str
    spread: Optional[float]
    neutral: bool
    indoor: Optional[bool]


@dataclass(frozen=True)
class Weather:
    verified: bool
    bad: bool
    detail: str


# --- Core Helper Functions ---
def stat_options() -> List[str]:
    return STATS


def default_season() -> int:
    return current_nfl_season()


def canon_stat(stat: str) -> str:
    """Canonicalize stat category to standard nflverse/threshold keys."""
    s = str(stat or "").upper().strip().replace("_", " ")
    s = re.sub(r"^PLAYER\s+", "", s)
    if "KICK" in s and ("POINT" in s or "PTS" in s): return "KICKING POINTS"
    if "FIELD" in s or "FG" in s: return "FIELD GOALS"
    if "TOUCHDOWN" in s or "TD" in s:
        if "PASS" in s: return "PASSING TDS"
        return "TOUCHDOWNS"
    if "PASS" in s and "YARD" in s: return "PASSING YARDS"
    if "PASS" in s and "COMP" in s: return "PASS COMPLETIONS"
    if "PASS" in s and "ATT" in s: return "PASS ATTEMPTS"
    if "PASS" in s and ("INT" in s or "INTERCEPTION" in s): return "INTERCEPTIONS"
    if "RUSH" in s and "YARD" in s: return "RUSHING YARDS"
    if "RUSH" in s and "ATT" in s: return "RUSHING ATTEMPTS"
    if ("REC" in s or "RECEIVING" in s) and "YARD" in s: return "RECEIVING YARDS"
    if "REC" in s or "RECEPTION" in s: return "RECEPTIONS"
    return s


def canon(abbr: Any) -> str:
    """Upper-case team code mapped to the nflverse convention."""
    a = str(abbr or "").upper().strip()
    return ABBR_CANON.get(a, a)


# --- Projection & Simulation Engine ---
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
    default_cv = DEFAULT_CV.get(stat, 0.50)
    cvs = [(w, c) for w, c in ((float(n), _cv(cur_x)), (n0, _cv(pri_x))) if c is not None and w > 0] + [(N_CV_PRIOR, default_cv)]
    cv = max(sum(w * c for w, c in cvs) / sum(w for w, _ in cvs), 0.6 * default_cv)
    return Projection(mu=mu, sigma=cv * mu * float(np.sqrt(1.0 + 1.0 / (n + n0))), n_current=n, n_prior=len(pri_x), mult=mult)


def simulate(mu: float, sigma: float, family: str, n_sims: int = N_SIMS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    var = max(sigma ** 2, 1e-4)
    if family == "gamma":
        shape = max(mu ** 2 / var, 0.5)
        scale = var / mu
        return np.rint(rng.gamma(shape=shape, scale=scale, size=n_sims))
    if family == "count":
        if var > mu * 1.0001:
            r = max(mu ** 2 / (var - mu), 0.5)
            p = float(r / (r + mu))
            return rng.negative_binomial(n=r, p=p, size=n_sims).astype(float)
        return rng.poisson(lam=mu, size=n_sims).astype(float)
    return np.maximum(np.rint(rng.normal(loc=mu, scale=sigma, size=n_sims)), 0.0)


# --- Devigging & Kelly ---
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


# --- Dynamic Dual-Directional 6-Point Assessment ---
def tier_for_side(side: str, score: int) -> str:
    if score >= SHARP_SCORE:
        return f"SHARP {side.upper()}"
    if score == LEAN_SCORE:
        return f"LEAN {side.upper()}"
    return "PASS / FADE"


def check_line(stat: str, line: float, side: str = "Over") -> Check:
    c_stat = canon_stat(stat)
    need = MIN_LINE.get(c_stat, MIN_LINE.get(stat, 0.0))
    ok = bool(line >= need) if side == "Over" else bool(line >= need * 0.5)
    detail = f"Line {line:g} vs minimum {need:g} for this stat" if side == "Over" else f"Line {line:g} is high enough for viable Under"
    return Check("baby_line", f"1. Line Safety ({side})", ok, detail + ("" if ok else ": too low/risky"))


def check_edge(side: str, stat: str, line: float, proj: float) -> Check:
    c_stat = canon_stat(stat)
    margin = proj - line if side == "Over" else line - proj
    need = max(MIN_EDGE_ABS.get(c_stat, MIN_EDGE_ABS.get(stat, 0.0)), MIN_EDGE_PCT * abs(line))
    ok = bool(margin >= need)
    return Check("ai_edge", f"2. AI {side} Edge", ok,
                 f"Projection {proj:.1f} is {margin:+.1f} {'above' if side == 'Over' else 'below'} "
                 f"the line in your direction; needs at least +{need:.1f}")


def check_implied(side: str, p_market: float) -> Check:
    ok = bool(p_market > IMPLIED_MIN)
    return Check("implied_chance", f"3. Implied Market {side}", ok,
                 f"Devigged price gives {side} {p_market:.1%} (needs > {IMPLIED_MIN:.0%}); single book devig")


def rank_supports(side: str, rank: int, n_teams: int, cutoff: int = RANK_CUTOFF) -> bool:
    return rank <= cutoff if side == "Over" else rank > (n_teams - cutoff)


def check_defense(side: str, season_rank: Optional[int], season_n: Optional[int],
                  recent_rank: Optional[int], recent_n: Optional[int], pos: str = "") -> Check:
    label = f"4. Defensive Rank ({side})"
    if None in (season_rank, season_n, recent_rank, recent_n):
        have = "season" if season_rank is not None else "recent" if recent_rank is not None else "neither"
        return Check("defensive_rank", label, False,
                     f"Not verified ({have} rank available): pick an opponent and load weekly data", verified=False)
    ok = rank_supports(side, season_rank, season_n) and rank_supports(side, recent_rank, recent_n)
    want = f"{RANK_CUTOFF} softest" if side == "Over" else f"{RANK_CUTOFF} toughest"
    return Check("defensive_rank", label, bool(ok),
                 f"Vs {pos or 'position'}s: season #{season_rank} of {season_n}, last {RECENT_GAMES} games #{recent_rank} of {recent_n} "
                 f"(1 = softest); {side} needs both in the {want}")


def flagged_players(inj: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    if inj.get("self_status"):
        out.append(f"the player ({inj['self_status']})")
    out += [f"{x['name']} ({x['status']})" for x in inj.get("listed", []) if x.get("p_out", 0.0) > 0]
    if inj.get("qb_flag"):
        out.append(str(inj["qb_flag"]))
    return out


def check_injury(inj: Optional[Mapping[str, Any]]) -> Check:
    label = "5. Injury & Role Stability"
    if not inj or not inj.get("checked") or not inj.get("feed_ok", False):
        return Check("injury", label, False, "Not verified: injury feed unavailable or adjustment off", verified=False)
    flagged = flagged_players(inj)
    return Check("injury", label, not flagged, "No active injury flags on team" if not flagged else "Flagged: " + ", ".join(flagged))


def check_game_script(spread: Optional[float], weather: Optional[Weather]) -> Check:
    label = "6. Game Script Spot"
    if spread is None or weather is None or not weather.verified:
        parts = []
        if spread is None:
            parts.append("spread unavailable")
        if weather is None or not weather.verified:
            parts.append(weather.detail if weather else "weather unavailable")
        return Check("game_script", label, False, "Not verified: " + "; ".join(parts), verified=False)
    blow = abs(spread) >= BLOWOUT_SPREAD
    ok = (not blow) and (not weather.bad)
    spread_txt = f"Spread {abs(spread):g}" + (f" (blowout risk at {BLOWOUT_SPREAD:g}+)" if blow else "")
    return Check("game_script", label, bool(ok), f"{spread_txt}; {weather.detail}")


def score_prop_side(side: str, stat: str, line: float, proj: float, p_market: float,
                    season_rank: Optional[int], season_n: Optional[int], recent_rank: Optional[int],
                    recent_n: Optional[int], pos: str, inj: Optional[Mapping[str, Any]],
                    spread: Optional[float], weather: Optional[Weather]) -> SixPointResult:
    checks = (
        check_line(stat, line, side),
        check_edge(side, stat, line, proj),
        check_implied(side, p_market),
        check_defense(side, season_rank, season_n, recent_rank, recent_n, pos),
        check_injury(inj),
        check_game_script(spread, weather),
    )
    score = int(np.sum([c.passed for c in checks]))
    return SixPointResult(side=side, score=score, tier=tier_for_side(side, score), checks=checks)


# --- Weather & Game Info Scrapers ---
def recent_defense_rank(weekly: pd.DataFrame, stat: str, pos: str, season: int, opponent: str,
                        n_recent: int = RECENT_GAMES, min_games: int = RECENT_MIN_GAMES
                        ) -> Tuple[Optional[int], Optional[int]]:
    c_stat = canon_stat(stat)
    g = _defense_games(weekly, c_stat, pos)
    g = g[g["SEASON"] == season].sort_values(["OPP", "WEEK"])
    g = g[g.groupby("OPP").cumcount(ascending=False) < n_recent]
    agg = g.groupby("OPP")["V"].agg(["mean", "size"])
    agg = agg[agg["size"] >= min_games]
    opp = canon(opponent)
    if opp not in agg.index:
        return None, None
    rank = agg["mean"].rank(ascending=False, method="min")
    return int(rank[opp]), int(len(agg))


def _spread_from_odds(odds: Any) -> Optional[float]:
    for o in odds or []:
        sp = o.get("spread")
        if isinstance(sp, (int, float)) and not isinstance(sp, bool):
            return abs(float(sp))
        det = str(o.get("details") or "")
        if re.search(r"\b(EVEN|PK|PICK)\b", det, re.IGNORECASE):
            return 0.0
        m = re.search(r"([+-]\d+(?:\.\d+)?)\s*$", det)
        if m:
            return abs(float(m[1]))
    return None


def parse_scoreboard_game(payload: Mapping[str, Any], team: str, opponent: Optional[str] = None) -> Optional[GameInfo]:
    t, o = canon(team), canon(opponent) if opponent else None
    for ev in payload.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = {c.get("homeAway"): canon((c.get("team") or {}).get("abbreviation")) for c in comp.get("competitors", [])}
        if t not in teams.values() or (o and o not in teams.values()):
            continue
        return GameInfo(home=teams.get("home", ""), away=teams.get("away", ""),
                        kickoff_iso=str(comp.get("date") or ev.get("date") or ""),
                        spread=_spread_from_odds(comp.get("odds")), neutral=bool(comp.get("neutralSite")),
                        indoor=(comp.get("venue") or {}).get("indoor"))
    return None


def summarize_forecast(payload: Mapping[str, Any], kickoff_iso: str, hours: int = WEATHER_HOURS) -> Optional[Dict[str, float]]:
    h = payload.get("hourly") or {}
    if not h.get("time"):
        return None
    start = pd.Timestamp(kickoff_iso).tz_convert("UTC").floor("h")
    times = pd.to_datetime(pd.Series(h["time"]), utc=True)
    mask = ((times >= start) & (times < start + pd.Timedelta(hours=hours))).to_numpy()
    if not mask.any():
        return None
    col = lambda k: np.nan_to_num(np.asarray(h.get(k) or [0.0] * len(times), dtype=float)[mask])
    return {"wind": float(col("wind_speed_10m").max()), "gust": float(col("wind_gusts_10m").max()),
            "precip": float(col("precipitation").sum()), "snow": float(col("snowfall").sum()),
            "temp": float(col("temperature_2m").min())}


def judge_weather(w: Mapping[str, float]) -> Weather:
    problems = [txt for bad, txt in (
        (w["wind"] >= WIND_MPH or w["gust"] >= GUST_MPH, f"wind {w['wind']:.0f} mph (gusts {w['gust']:.0f})"),
        (w["precip"] >= PRECIP_MM, f"rain {w['precip']:.1f} mm"),
        (w["snow"] >= SNOW_CM, f"snow {w['snow']:.1f} cm"),
        (w["temp"] <= COLD_F, f"{w['temp']:.0f}°F"),
    ) if bad]
    base = f"forecast wind {w['wind']:.0f} mph, {w['temp']:.0f}°F, precip {w['precip']:.1f} mm"
    return Weather(True, bool(problems), ("bad weather: " + ", ".join(problems)) if problems else "weather " + base)


def game_weather(info: Optional[GameInfo], fetch_forecast_fn: Callable[[float, float], Mapping[str, Any]]) -> Weather:
    if info is None:
        return Weather(False, False, "game not found on ESPN scoreboard, weather not checked")
    if info.neutral:
        return Weather(False, False, "neutral-site game, weather not checked")
    lat_lon_roof = STADIUMS.get(canon(info.home))
    if lat_lon_roof is None:
        return Weather(False, False, f"unknown stadium for {info.home}, weather not checked")
    lat, lon, roof = lat_lon_roof
    if roof != "outdoor" or info.indoor:
        return Weather(True, False, f"{roof if roof != 'outdoor' else 'indoor'} stadium, no weather risk")
    try:
        summary = summarize_forecast(fetch_forecast_fn(lat, lon), info.kickoff_iso)
    except Exception as e:
        return Weather(False, False, f"forecast unavailable ({e})")
    if summary is None:
        return Weather(False, False, "no forecast hours for kickoff (beyond 16-day window?)")
    return judge_weather(summary)


def fetch_game_info(team: str, opponent: Optional[str], gameday: Optional[str]) -> Optional[GameInfo]:
    day = pd.Timestamp(gameday) if gameday else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    end = day if gameday else day + pd.Timedelta(days=14)
    dates = day.strftime("%Y%m%d") if gameday else f"{day:%Y%m%d}-{end:%Y%m%d}"
    resp = requests.get(ESPN_SCOREBOARD, params={"dates": dates, "limit": 100}, timeout=8)
    resp.raise_for_status()
    return parse_scoreboard_game(resp.json(), team, opponent)


def fetch_forecast(lat: float, lon: float) -> Dict[str, Any]:
    params = {"latitude": lat, "longitude": lon, "forecast_days": 16, "timezone": "UTC",
              "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m",
              "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "mm"}
    resp = requests.get(OPEN_METEO, params=params, timeout=8)
    resp.raise_for_status()
    return resp.json()


def assess(res: Mapping[str, Any], stat_category: str, line: float, weekly: Optional[pd.DataFrame], season: int,
           get_game: Callable[[str, Optional[str], Optional[str]], Optional[GameInfo]] = fetch_game_info,
           get_forecast_fn: Callable[[float, float], Mapping[str, Any]] = fetch_forecast) -> Dict[str, Any]:
    stat = canon_stat(stat_category)
    best = "Over" if str(res.get("best_side", "")).startswith("Over") else "Under"
    p_best = float(res.get("p_market", 50.0)) / 100.0
    m = res.get("matchup") or {}
    recent_rank = recent_n = None
    if m and weekly is not None:
        recent_rank, recent_n = recent_defense_rank(weekly, stat, m.get("pos", ""), season, m.get("opponent", ""))
    try:
        info = get_game(str(res.get("team", "")), m.get("opponent"), m.get("gameday"))
    except Exception:
        info = None
    weather = game_weather(info, get_forecast_fn)
    out: Dict[str, Any] = {"best": best}
    for side, p in (("Over", p_best if best == "Over" else 1.0 - p_best), ("Under", p_best if best == "Under" else 1.0 - p_best)):
        out[side] = score_prop_side(side, stat, float(line), float(res.get("mu", line)), p,
                                    m.get("rank"), m.get("n_teams"), recent_rank, recent_n,
                                    str(m.get("pos", "")), res.get("inj"),
                                    info.spread if info else None, weather).as_dict()
    return out


# --- Matchup Resolver ---
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
    k = (k_map or {}).get(stat, DEFAULT_K.get(stat, 0.35))
    return {
        "opponent": opp, "pos": pos, "week": info.get("week"), "home": info.get("home"), "gameday": info.get("gameday"),
        "rest": info.get("rest"), "opp_rest": info.get("opp_rest"), "rank": int(row["RANK"]), "n_teams": int(len(table)),
        "allowed": float(row["EST"]), "current_allowed": float(row["CUR"]) if pd.notna(row["CUR"]) else None,
        "league": float(row["LEAGUE"]), "index": float(row["INDEX"]), "games": int(row["GAMES"]),
        "k": float(k), "calibrated": bool(k_map and stat in k_map), "mult": matchup_multiplier(float(row["INDEX"]), float(k)),
        "board": table.reset_index().rename(columns={"OPP": "TEAM"})[["RANK", "TEAM", "EST", "CUR", "INDEX", "GAMES"]],
    }, note


# --- Main Prop Evaluator Function ---
def evaluate_prop(player_name: str, stat_category: str, line: float, over_odds: float, under_odds: float,
                  projection_source: str = "Model (EWMA mean)", season: int = 2026, opponent: Optional[str] = None,
                  matchup_k: Optional[Dict[str, float]] = None, adj_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        line = float(line)
        over_odds = float(over_odds)
        under_odds = float(under_odds)
        if abs(over_odds) < 100 or abs(under_odds) < 100:
            return {"error": "Odds must be valid American odds (<= -100 or >= +100)."}

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
            "player_name": player_name, "team": team, "pos": pos,
            "mean": float(np.mean(sims)), "median": float(np.median(sims)),
            "best_side": f"{'Over' if is_over else 'Under'} {line:g}",
            "ev_pct": float(ev * 100.0), "edge_pp": float((p_cond - dv_side) * 100.0),
            "p_model": float(p_cond * 100.0), "p_market": float(dv_side * 100.0),
            "fair_odds": prob_to_american(p_cond),
            "kelly_pct": float(kelly_fraction(p_w, p_l, odds_side) * 100.0),
            "raw_avg": float(np.mean(vals)) if len(vals) else 0.0, "mu": float(proj.mu),
            "hit10": int(np.sum(vals > line)), "n10": len(vals),
            "hit5": int(np.sum(vals[-5:] > line)), "n5": len(vals[-5:]),
            "n_current": proj.n_current, "n_prior": proj.n_prior,
            "matchup": matchup, "warnings": warnings, "usage_mult": usage_mult,
            "inj": inj, "tiers": tiers,
            "base_proj": float(proj.mu / total_mult) if total_mult else float(proj.mu),
            "gamelog_details": [{"stat_value": float(v), "season": s, "x_label": f"W{int(w)}"}
                                for v, s, w in zip(vals, recent["SEASON_TAG"], recent["WEEK"])],
        }
    except Exception as e:
        return {"error": f"Evaluation error: {e}"}


# --- UI Formatting & Dataframe Helpers ---
def game_matches(label: str, team: Optional[str], opponent: Optional[str]) -> bool:
    nicks = [NICKNAMES.get(canon(t)) for t in (team, opponent)]
    return all(nicks) and all(n in str(label).lower() for n in nicks)


def teams_in_label(label: str) -> List[str]:
    low = str(label).lower()
    return [code for code, nick in NICKNAMES.items() if nick in low]


def opponent_from_label(label: str, team: Optional[str]) -> Optional[str]:

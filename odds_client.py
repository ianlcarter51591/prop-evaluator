"""The Odds API v4 client: games, player-prop boards, line shopping, and no-vig consensus."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

BASE_URL: str = "https://api.the-odds-api.com/v4"
TIMEOUT: int = 30
DISPLAY_TZ: str = "America/New_York"

SPORT_KEYS: Dict[str, str] = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
}

# App stat label -> Odds API market key (labels match STATS in app.py).
MARKETS: Dict[str, Dict[str, str]] = {
    "NFL": {
        "Receptions": "player_receptions",
        "Receiving Yards": "player_reception_yds",
        "Rushing Attempts": "player_rush_attempts",
        "Rushing Yards": "player_rush_yds",
        "Pass Completions": "player_pass_completions",
        "Passing Yards": "player_pass_yds",
        "Passing TDs": "player_pass_tds",
    },
    "NBA": {
        "Points": "player_points",
        "Rebounds": "player_rebounds",
        "Assists": "player_assists",
        "3-Pointers Made": "player_threes",
        "Pts + Reb + Ast": "player_points_rebounds_assists",
        "Steals": "player_steals",
        "Blocks": "player_blocks",
    },
    "MLB": {
        "Pitcher Strikeouts": "pitcher_strikeouts",
        "Pitcher Outs Recorded": "pitcher_outs",
        "Pitcher Earned Runs": "pitcher_earned_runs",
        "Pitcher Hits Allowed": "pitcher_hits_allowed",
        "Pitcher Walks Allowed": "pitcher_walks",
        "Batter Hits": "batter_hits",
        "Batter Total Bases": "batter_total_bases",
        "Batter Hits + Runs + RBIs": "batter_hits_runs_rbis",
        "Batter Home Runs": "batter_home_runs",
        "Batter RBIs": "batter_rbis",
        "Batter Runs": "batter_runs_scored",
        "Batter Stolen Bases": "batter_stolen_bases",
        "Batter Strikeouts": "batter_strikeouts",
    },
}

# Bookmaker keys: williamhill_us is Caesars (paid plans only); pinnacle is the EU sharp book.
DEFAULT_BOOKS: str = "draftkings,fanduel,betmgm,williamhill_us,pinnacle"

BOARD_COLUMNS: Tuple[str, ...] = (
    "player", "book", "book_title", "side", "line", "price", "updated",
)
SUMMARY_COLUMNS: Tuple[str, ...] = (
    "Player", "Line", "Line Range", "Books", "Best Over", "Over Book",
    "Best Under", "Under Book", "No-vig Over %",
)


class OddsApiError(Exception):
    """Raised for any Odds API failure, with a message that is safe to show on screen."""


@dataclass(frozen=True)
class Quota:
    """Credit usage reported in the response headers."""

    remaining: Optional[int] = None
    used: Optional[int] = None
    last: Optional[int] = None


def _to_int(value: Any) -> Optional[int]:
    """Parse a header value to int, or None if missing or malformed."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _get(path: str, api_key: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Quota]:
    """GET an Odds API path; return (parsed JSON, quota). The API key never appears in errors."""
    if not api_key:
        raise OddsApiError("Enter your Odds API key first.")
    try:
        resp = requests.get(
            f"{BASE_URL}{path}",
            params={"apiKey": api_key, **(params or {})},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise OddsApiError(f"Network error: {str(exc).replace(api_key, '***')}") from None

    quota = Quota(
        _to_int(resp.headers.get("x-requests-remaining")),
        _to_int(resp.headers.get("x-requests-used")),
        _to_int(resp.headers.get("x-requests-last")),
    )
    if resp.status_code >= 400:
        try:
            detail = str(resp.json().get("message", ""))
        except ValueError:
            detail = resp.text[:200]
        detail = detail.replace(api_key, "***")
        if resp.status_code == 401:
            raise OddsApiError(f"Key or credit problem (HTTP 401): {detail}")
        if resp.status_code == 429:
            raise OddsApiError("Rate limited (HTTP 429). Wait a few seconds and try again.")
        raise OddsApiError(f"Odds API error (HTTP {resp.status_code}): {detail}")
    return resp.json(), quota


def parse_books(text: str) -> Tuple[str, ...]:
    """Split a comma-separated bookmaker-key string into a clean tuple."""
    return tuple(b.strip().lower() for b in text.split(",") if b.strip())


def estimate_cost(n_markets: int, books: Sequence[str]) -> int:
    """Credits for one event-odds call: markets x regions (every 10 bookmakers = 1 region)."""
    return n_markets * max(1, math.ceil(len(books) / 10))


def list_events(sport: str, api_key: str) -> Tuple[pd.DataFrame, Quota]:
    """List upcoming/live games for a sport (free endpoint, costs no credits)."""
    data, quota = _get(f"/sports/{SPORT_KEYS[sport]}/events", api_key, {"dateFormat": "iso"})
    cols = ["id", "label", "away_team", "home_team", "start"]
    if not data:
        return pd.DataFrame(columns=cols), quota
    df = pd.DataFrame(data)
    start = pd.to_datetime(df["commence_time"], utc=True).dt.tz_convert(DISPLAY_TZ)
    df["start"] = start
    df["label"] = (
        df["away_team"] + " @ " + df["home_team"] + " | " + start.dt.strftime("%a %m/%d %I:%M %p %Z")
    )
    return df.sort_values("start").reset_index(drop=True)[cols], quota


def parse_board(event: Dict[str, Any], market: str) -> pd.DataFrame:
    """Flatten an event-odds response into one row per (player, book, side)."""
    rows = [
        {
            "player": o.get("description"),
            "book": b.get("key"),
            "book_title": b.get("title") or b.get("key"),
            "side": o.get("name"),
            "line": o.get("point"),
            "price": o.get("price"),
            "updated": m.get("last_update"),
        }
        for b in event.get("bookmakers", [])
        for m in b.get("markets", [])
        if m.get("key") == market
        for o in m.get("outcomes", [])
    ]
    df = pd.DataFrame(rows, columns=list(BOARD_COLUMNS))
    df = df[df["side"].isin(["Over", "Under"])].dropna(subset=["player", "line", "price"])
    df = df.assign(line=df["line"].astype(float), price=df["price"].astype(float))
    return df.reset_index(drop=True)


def fetch_prop_board(
    sport: str, event_id: str, market: str, books: Sequence[str], api_key: str
) -> Tuple[pd.DataFrame, Quota]:
    """Fetch one market for one game across the chosen books (American odds)."""
    params: Dict[str, Any] = {
        "regions": "us",
        "markets": market,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if books:
        params["bookmakers"] = ",".join(books)
    data, quota = _get(f"/sports/{SPORT_KEYS[sport]}/events/{event_id}/odds", api_key, params)
    return parse_board(data, market), quota


def american_to_implied(price: np.ndarray) -> np.ndarray:
    """Vectorized American odds -> vig-inclusive implied probability."""
    p = np.asarray(price, dtype=float)
    mag = np.abs(p)
    return np.where(p > 0, 100.0 / (mag + 100.0), mag / (mag + 100.0))


def player_offers(board: pd.DataFrame, player: str) -> pd.DataFrame:
    """One row per (book, line) where both sides are posted, with no-vig Over % and hold."""
    cols = ["Book", "Line", "Over", "Under", "No-vig Over %", "Hold %"]
    sub = board[board["player"] == player]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    wide = sub.pivot_table(
        index=["book_title", "line"], columns="side", values="price", aggfunc="max"
    ).reset_index()
    if not {"Over", "Under"} <= set(wide.columns):
        return pd.DataFrame(columns=cols)
    wide = wide.dropna(subset=["Over", "Under"])
    if wide.empty:
        return pd.DataFrame(columns=cols)
    q_o = american_to_implied(wide["Over"].to_numpy())
    q_u = american_to_implied(wide["Under"].to_numpy())
    out = pd.DataFrame(
        {
            "Book": wide["book_title"].to_numpy(),
            "Line": wide["line"].to_numpy(),
            "Over": wide["Over"].to_numpy(),
            "Under": wide["Under"].to_numpy(),
            "No-vig Over %": q_o / (q_o + q_u) * 100.0,
            "Hold %": (q_o + q_u - 1.0) * 100.0,
        }
    )
    return out.sort_values(["Book", "Line"]).reset_index(drop=True)


def summarize_board(board: pd.DataFrame) -> pd.DataFrame:
    """Per-player summary: consensus line, best Over/Under price there, no-vig Over %."""
    if board.empty:
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))

    consensus = board.groupby("player")["line"].agg(lambda s: float(s.mode().median()))
    df = board.join(consensus.rename("consensus"), on="player")
    at = df[np.isclose(df["line"], df["consensus"])]

    out = pd.DataFrame(index=consensus.index)
    out["Line"] = consensus
    out["Line Range"] = board.groupby("player")["line"].agg(
        lambda s: f"{s.min():g}" if s.min() == s.max() else f"{s.min():g}-{s.max():g}"
    )
    out["Books"] = at.groupby("player")["book"].nunique()

    for side in ("Over", "Under"):
        sub = at[at["side"] == side]
        if sub.empty:
            out[f"Best {side}"], out[f"{side} Book"] = np.nan, None
            continue
        best = sub.loc[sub.groupby("player")["price"].idxmax()].set_index("player")
        out[f"Best {side}"] = best["price"]
        out[f"{side} Book"] = best["book_title"]

    wide = at.pivot_table(index=["player", "book"], columns="side", values="price", aggfunc="max")
    if {"Over", "Under"} <= set(wide.columns):
        wide = wide.dropna(subset=["Over", "Under"])
        q_o = american_to_implied(wide["Over"].to_numpy())
        q_u = american_to_implied(wide["Under"].to_numpy())
        nv = pd.Series(q_o / (q_o + q_u) * 100.0, index=wide.index).groupby(level="player").mean()
        out["No-vig Over %"] = nv
    else:
        out["No-vig Over %"] = np.nan

    out = out.reset_index().rename(columns={"player": "Player"})
    return out[list(SUMMARY_COLUMNS)].sort_values("Player").reset_index(drop=True)

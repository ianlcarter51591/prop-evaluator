"""NFL weekly player-log fetcher backed by nflverse release files."""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Optional, Tuple

import pandas as pd

# New location first, legacy location as fallback.
URL_TEMPLATES: Tuple[str, ...] = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "stats_player/stats_player_week_{season}.csv",
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats_{season}.csv",
)


def current_nfl_season(today: Optional[date] = None) -> int:
    """Return the NFL season year (Mar-Dec -> that year, Jan-Feb -> prior year)."""
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1


@lru_cache(maxsize=8)
def _load_season(season: int) -> pd.DataFrame:
    """Download one season of weekly player stats, trying each URL in turn."""
    last_err: Optional[Exception] = None
    for template in URL_TEMPLATES:
        try:
            df = pd.read_csv(template.format(season=season), low_memory=False)
            # Normalize schema differences between old and new releases.
            if "recent_team" not in df.columns and "team" in df.columns:
                df["recent_team"] = df["team"]
            return df
        except Exception as exc:  # 404s, network errors, parse errors
            last_err = exc
    raise RuntimeError(f"No stats file available for {season}: {last_err}")


def _match_player(df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Case-insensitive match on display name, short name (S.Diggs), or substring."""
    query = player_name.strip().lower()
    parts = query.split()
    short_form = f"{parts[0][0]}.{parts[-1]}" if len(parts) >= 2 else query

    mask = pd.Series(False, index=df.index)
    if "player_display_name" in df.columns:
        disp = df["player_display_name"].fillna("").str.lower()
        mask |= (disp == query) | disp.str.contains(query, regex=False)
    if "player_name" in df.columns:
        short = df["player_name"].fillna("").str.lower()
        mask |= (short == query) | (short == short_form) | short.str.contains(query, regex=False)
    return df[mask].copy()


def fetch_nfl_player_logs(
    player_name: str, target_season: Optional[int] = None
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Return (weekly logs, error). Falls back one season if the target isn't published."""
    try:
        season = target_season or current_nfl_season()
        note: Optional[str] = None
        try:
            df = _load_season(season)
        except RuntimeError:
            season -= 1
            df = _load_season(season)
            note = f"Target season not published yet; using {season}."

        matched = _match_player(df, player_name)
        if "season" in matched.columns:
            matched = matched[matched["season"] == season]
        if matched.empty:
            return pd.DataFrame(), f"NFL player '{player_name}' not found in {season} data."

        stat_col = "receptions" if "receptions" in matched.columns else "passing_yards"
        matched = (
            matched.dropna(subset=[stat_col])
            .sort_values("week")
            .reset_index(drop=True)
        )
        return matched, note
    except Exception as exc:
        return pd.DataFrame(), f"Fetcher Error: {exc}"

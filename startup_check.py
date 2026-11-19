"""Startup self-check: names the exact file or name that is missing instead of a redacted ImportError.

Streamlit Cloud hides ImportError text. app.py calls problems() first and shows each line on screen.
"""
from __future__ import annotations

import importlib
import inspect
from typing import Dict, List, Tuple

# Import order matters: a module is tried only after the ones it depends on.
REQUIRED: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("nfl_matchups", ("DEFAULT_K", "matchup_multiplier", "matchup_table", "run_backtests", "_defense_games")),
    ("sports_data", ("NFL_TEAMS", "clean_name", "clean_series", "current_nfl_season", "fetch_nfl_player_logs",
                     "load_schedule", "load_weekly", "next_game")),
    ("nfl_adjust", ("injury_layer", "tier_splits", "load_injuries")),
    ("odds_credits", ("record", "record_headers", "fetch_cached", "credits_status", "friendly_error", "cache_note")),
    ("six_point_core", ("SportSpec", "choose_best", "assemble", "tag_text", "add_six_point_column", "style_tags", "find_espn_spread", "spread_from_espn_odds")),
    ("nfl_six_point", ("assess", "add_six_point_column", "style_tags", "game_matches", "opponent_from_label", "label_gameday")),
    ("multi_sport_engine", ("DEFAULTS", "SOURCES", "default_season", "evaluate_prop", "stat_options")),
    ("odds_api_client", ("MARKETS", "OddsApiError", "fetch_prop_board", "list_events", "player_offers", "summarize_board")),
    ("live_game", ("scoreboard", "headshot_url", "evaluate_live")),
    ("nfl_headshots", ("fetch_roster", "player_photo")),
    ("cloud_store", ("configure", "load_text", "save_text", "flush", "status")),
    ("validation", ("by_score", "by_criterion", "calibration", "brier", "verdict")),
    ("parlay", ("Leg", "evaluate", "legs_from_snapshots")),
    ("vegas_env", ("find_espn_lines", "implied_team_total", "team_environment")),
    ("nfl_vegas", ("DEFAULT_K", "vegas_layer", "load_lines", "run_backtests")),
    ("nfl_snapshots", ("make_row", "upsert", "grade", "normalize", "by_tier", "by_pin")),
    ("pin_check", ("split", "table", "for_player", "cross", "assess", "convert_over")),
    ("nfl_bet_log", ("make_bet_row", "add_bet", "auto_grade", "summarize_log", "merge_logs", "apply_board_as_closing")),
    ("favorites", ("legs", "option_table", "set_flags", "toggle", "find_id", "star_many", "mark_placed", "snap_id_for_leg", "tag_best_picks")),
    ("best_picks", ("render", "make_row", "rank_picks", "selected_row", "cap_per_game", "sharp_picks")),
    ("prop_review", ("fit_blend", "blend", "render_blend_banner", "render_calibration", "record_close", "compute_clv", "clv_summary", "render_clv", "render_daily", "day_counts", "graded_days")),
    ("shared_ui", ("render_storage_status", "render_validation", "render_parlay", "render_favorites", "render_star_button", "clickable_table", "render_tracked_place_bet")),
)


LOCAL_MODULES = {m for m, _ in REQUIRED}

# Arguments app.py passes: catches an old engine paired with the new app.
SIGNATURES: Dict[str, Dict[str, Tuple[str, ...]]] = {"multi_sport_engine": {"evaluate_prop": ("gameday", "six_point")}}


def problems() -> List[str]:
    """One human-readable line per broken or missing module / name. Empty list = all good."""
    out: List[str] = []
    for mod, names in REQUIRED:
        try:
            m = importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name == mod:
                out.append(f"{mod}.py is missing from the app folder.")
            elif e.name in LOCAL_MODULES:
                out.append(f"{e.name}.py is missing from the app folder (needed by {mod}.py).")
            else:
                out.append(f"{mod}.py needs the package '{e.name}', which isn't installed (add it to requirements.txt).")
            continue
        except Exception as e:
            out.append(f"{mod}.py failed to import: {type(e).__name__}: {e}")
            continue
        missing = [n for n in names if not hasattr(m, n)]
        if missing:
            out.append(f"{mod}.py is an older or different version: it has no {', '.join(missing)}. Replace it with the matching file.")
        for fn, params in SIGNATURES.get(mod, {}).items():
            try:
                have = set(inspect.signature(getattr(m, fn)).parameters)
            except (AttributeError, TypeError, ValueError):
                continue
            lacking = [p for p in params if p not in have]
            if lacking:
                out.append(f"{mod}.py is an older version: {fn}() has no {', '.join(lacking)} option that app.py uses. Replace it with the new {mod}.py.")
    return out

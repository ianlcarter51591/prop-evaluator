"""NFL Prop Evaluator: real opponent-vs-position matchups, devigged edge, calibrated matchup weights."""
import json
import uuid
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from multi_sport_engine import (
    BLOWOUT_SPREAD, DEFAULTS, RECENT_GAMES, SOURCES, GameInfo, add_six_point_column, assess,
    default_season, evaluate_prop, fetch_forecast, fetch_game_info, game_matches, label_gameday,
    opponent_from_label, stat_options, style_tags,
)
from nfl_matchups import DEFAULT_K, run_backtests
from nfl_headshots import fetch_roster, player_photo
from odds_api_client import (
    MARKETS,
    OddsApiError,
    fetch_prop_board,
    list_events,
    player_offers,
    summarize_board,
)
from sports_data import NFL_TEAMS, load_weekly
import live_game

st.set_page_config(
    page_title="NFL Prop Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# High-Contrast Powder Blue UI Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    .vision-card, 
    [data-testid="stVerticalBlockBorderWrapper"], 
    [data-testid="stExpander"], 
    .vision-header {
        background: #b9e6ff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 14px rgba(0, 117, 255, 0.08) !important;
    }
    .vision-card { padding: 20px !important; margin-bottom: 20px !important; }

    [data-testid="stExpander"] summary, 
    [data-testid="stExpander"] summary * {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 16px !important;
    }

    .vision-title {
        color: #0369a1 !important;
        font-size: 26px;
        font-weight: 900;
        margin: 0;
    }

    .vision-sub {
        color: #0f172a !important;
        font-size: 13px;
        font-weight: 600;
        margin-top: 4px;
    }

    [data-testid="stMetric"], .metric-box {
        background: #b9e6ff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        text-align: center;
    }
    [data-testid="stMetricValue"], .metric-value {
        color: #0369a1 !important;
        font-size: 24px !important;
        font-weight: 900 !important;
    }
    [data-testid="stMetricLabel"], .metric-label {
        color: #0f172a !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    .metric-value-green {
        color: #15803d !important;
        font-size: 24px !important;
        font-weight: 900 !important;
    }

    label, p, span, div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #0f172a !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 2px solid #0284c7 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] * {
        color: #ffffff !important;
    }

    .stButton>button, [data-testid="stFormSubmitButton"]>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        font-weight: 900 !important;
        border-radius: 10px !important;
        border: None !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3) !important;
        width: 100% !important;
    }
    .stButton>button *, [data-testid="stFormSubmitButton"]>button * {
        color: #ffffff !important;
        font-weight: 900 !important;
    }

    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

AUTO_OPP = "Auto (next game)"
K_PATH = "matchup_k.json"


def _secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def _to_american(x: float) -> int:
    x = float(x)
    if abs(x) >= 100:
        return int(round(x))
    if x > 1.0:
        return int(round((x - 1.0) * 100.0)) if x >= 2.0 else int(round(-100.0 / (x - 1.0)))
    return -110


def load_k_file() -> Optional[Dict[str, float]]:
    try:
        if os.path.exists(K_PATH):
            with open(K_PATH) as fh:
                return json.load(fh)
    except Exception:
        pass
    return None


def save_k_file(k: Optional[Dict[str, float]]) -> None:
    try:
        if k is None:
            if os.path.exists(K_PATH):
                os.remove(K_PATH)
        else:
            with open(K_PATH, "w") as fh:
                json.dump(k, fh)
    except Exception:
        pass


def sync_populate_to_widgets(stat_category: str, player: str, line: float, over: float, under: float) -> None:
    ss = st.session_state
    ss["input_player"], ss["input_stat"], ss["input_line"] = player, stat_category, float(line)
    ss["input_over"], ss["input_under"] = _to_american(over), _to_american(under)
    ss["input_opp"] = AUTO_OPP
    ss["pending_eval"] = True


def _adj_cfg() -> Dict[str, Any]:
    ss = st.session_state
    split = lambda key: [x.strip() for x in str(ss[key]).split(",") if x.strip()]
    return {"injuries": ss["adj_inj"], "absorb_scale": float(ss["adj_absorb"]),
            "manual_out": split("adj_out"), "manual_q": split("adj_q"), "manual_in": split("adj_in")}


def scan_board(summary: pd.DataFrame, stat_category: str, game_label: str, progress: Any = None) -> Dict[str, Any]:
    ss = st.session_state
    out: Dict[str, Any] = {}
    try:
        weekly: Optional[pd.DataFrame] = load_weekly(default_season(), 1)
    except Exception:
        weekly = None
    gameday = label_gameday(game_label)
    n = len(summary)

    def run(name: str, r: pd.Series, opponent: str) -> Dict[str, Any]:
        return evaluate_prop(
            player_name=name,
            stat_category=stat_category,
            line=float(r["Line"]),
            over_odds=_to_american(r["Over"]),
            under_odds=_to_american(r["Under"]),
            projection_source=ss.get("input_source", SOURCES[0]),
            season=default_season(),
            opponent=opponent,
            matchup_k=ss.get("matchup_k"),
            adj_cfg=_adj_cfg(),
        )

    for k, (_, r) in enumerate(summary.iterrows()):
        name = str(r["Player"])
        if r[["Line", "Over", "Under"]].isna().any():
            out[name] = "incomplete odds"
            continue
        if progress is not None:
            progress.progress(k / max(n, 1), text=f"Scanning {name} ({k + 1}/{n})")
        try:
            res = run(name, r, "AUTO")
            if "error" in res:
                out[name] = str(res["error"])[:70]
                continue
            if not game_matches(game_label, res.get("team"), (res.get("matchup") or {}).get("opponent")):
                opp = opponent_from_label(game_label, res.get("team"))
                if opp is None:
                    out[name] = f"{res.get('team')} isn't in this game"
                    continue
                res = run(name, r, opp)
                if "error" in res:
                    out[name] = str(res["error"])[:70]
                    continue
            if res.get("matchup") and gameday:
                res["matchup"] = {**res["matchup"], "gameday": gameday}
            out[name] = assess(res, stat_category, float(r["Line"]), weekly, default_season(), cached_game_info, cached_forecast)
        except Exception as e:
            out[name] = f"error: {str(e)[:60]}"
    if progress is not None:
        progress.progress(1.0, text="Scan complete")
    return out


def render_live_panel() -> None:
    ss = st.session

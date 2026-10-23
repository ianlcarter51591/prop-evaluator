"""Prop Evaluator: Live DraftKings Lines via The Odds API."""
import os
from typing import Any, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from multi_sport_engine import (
    DEFAULTS,
    SPORTS,
    default_season,
    evaluate_prop_multisport,
    stat_options,
)

from odds_api_client import (
    DEFAULT_BOOKS,
    MARKETS,
    OddsApiError,
    fetch_prop_board,
    list_events,
    parse_books,
    player_offers,
    summarize_board,
)

st.set_page_config(
    page_title="Prop Evaluator | Live DraftKings Lines",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(0, 117, 255, 0.12) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(111, 66, 193, 0.12) 0%, transparent 40%),
                    #060B26 !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
        display: none !important;
    }
    .stMetric {
        background: rgba(15, 21, 53, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 14px 18px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def _secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()

def load_offer_into_model(sport: str, player: str, line: float, over: float, under: float) -> None:
    ss = st.session_state
    ss[f"player_{sport}"] = player
    ss[f"line_{sport}"] = float(line)
    ss[f"over_{sport}"] = int(round(over))
    ss[f"under_{sport}"] = int(round(under))
    ss["ran_sport"] = sport

def render_live_panel(sport: str) -> None:
    ss = st.session_state
    api_key = _secret("ODDS_API_KEY")
    with st.expander("📡 Fetch Live DraftKings Lines (The Odds API)", expanded=True):
        if not api_key:
            st.warning("Please configure `ODDS_API_KEY` in Streamlit Secrets.")
            return

        live_stat = st.selectbox("Select Prop Market", list(MARKETS[sport].keys()), key=f"live_stat_{sport}")

        if st.button("Step 1: Load Upcoming Games", key=f"load_games_{sport}"):
            try:
                events, _ = list_events(sport, api_key)
                ss[f"events_{sport}"] = events
            except OddsApiError as exc:
                st.error(str(exc))

        events = ss.get(f"events_{sport}")
        if events is not None and not events.empty:
            game_idx = st.selectbox(
                "Step 2: Select Game",
                range(len(events)),
                format_func=lambda i: str(events.loc[i, "label"]),
                key=f"live_game_{sport}",
            )
            if st.button(f"Step 3: Fetch {live_stat} Odds", key=f"fetch_board_{sport}"):
                try:
                    event_id = str(events.loc[game_idx, "id"])
                    board, _ = fetch_prop_board(sport, event_id, live_stat, ("draftkings",), api_key)
                    ss[f"board_{sport}"] = board
                except OddsApiError as exc:
                    st.error(str(exc))

        board = ss.get(f"board_{sport}")
        if board is not None and not board.empty:
            summary = summarize_board(board)
            st.dataframe(summary, use_container_width=True)
            player_pick = st.selectbox("Step 4: Choose Player to Evaluate", summary["Player"].tolist(), key=f"pick_{sport}")
            offers = player_offers(board, player_pick)
            if not offers.empty:
                row = offers.iloc[0]
                if st.button(f"⚡ Populate {player_pick} ({row['Line']} | Over {row['Over']} / Under {row['Under']})"):
                    load_offer_into_model(sport, player_pick, row["Line"], row["Over"], row["Under"])
                    st.rerun()

def main() -> None:
    st.title("⚡ NFL & NBA Prop Evaluator")
    sport = st.radio("Sport", list(SPORTS), horizontal=True, key="sport")

    render_live_panel(sport)

    ss = st.session_state
    def_player, def_line = DEFAULTS[sport]
    
    with st.form(f"form_{sport}"):
        c1, c2, c3 = st.columns([2, 1, 1])
        player_name = c1.text_input("Player Name", value=ss.get(f"player_{sport}", def_player))
        stat_category = c2.selectbox("Stat Category", stat_options(sport))
        line_val = c3.number_input("Line", value=float(ss.get(f"line_{sport}", def_line)), step=0.5)

        c4, c5 = st.columns(2)
        over_odds = c4.number_input("Over Odds", value=int(ss.get(f"over_{sport}", -110)))
        under_odds = c5.number_input("Under Odds", value=int(ss.get(f"under_{sport}", -110)))

        submitted = st.form_submit_button("⚡ Evaluate Prop")

    if submitted or ss.get("ran_sport") == sport:
        res = evaluate_prop_multisport(
            league=sport,
            player_name=player_name,
            stat_category=stat_category,
            line=line_val,
            over_odds=over_odds,
            under_odds=under_odds,
        )

        if "error" in res:
            st.error(res["error"])
        else:
            st.markdown(f"## {res['player_name']} | {stat_category}")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("MEAN", f"{res['projection_mean']:.2f}")
            m2.metric("MEDIAN", f"{res['projection_median']:.2f}")
            m3.metric("BEST SIDE", res["best_side"])
            m4.metric("FAIR ODDS", f"{res['fair_odds']:+d}")
            m5.metric("EDGE VS MKT", f"{res['edge_vs_market']:+.1f}%")
            m6.metric("EV", f"{res['ev_pct']:+.1f}%")

if __name__ == "__main__":
    main()

"""Prop Evaluator (Vision UI): Clean NFL & NBA Engine."""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from multi_sport_engine import (
    DEFAULTS,
    DEVIG_METHODS,
    SOURCES,
    SPORTS,
    default_season,
    evaluate_prop_multisport,
    stat_options,
)

try:
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
    HAS_ODDS = True
except Exception:
    HAS_ODDS = False

st.set_page_config(
    page_title="Prop Evaluator | NFL & NBA",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GREEN, RED, GRAY = "#00E676", "#FF5252", "#9CA3AF"

VISION_UI_CSS = """
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

.vision-card, [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] {
    background: rgba(15, 21, 53, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}
.vision-card { padding: 20px !important; margin-bottom: 20px !important; }

[data-testid="stMetric"] {
    background: rgba(15, 21, 53, 0.65) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #A0AEC0 !important; font-size: 0.82rem !important; text-transform: uppercase !important; }

.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #0075FF 0%, #0052D4 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    box-shadow: 0 4px 15px rgba(0, 117, 255, 0.4) !important;
    width: 100% !important;
}

div[data-baseweb="input"], div[data-baseweb="select"] {
    background: rgba(15, 21, 53, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}
div[data-baseweb="input"] input { color: #FFFFFF !important; }
</style>
"""
st.markdown(VISION_UI_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def cached_evaluate(**kwargs: Any) -> Dict[str, Any]:
    return evaluate_prop_multisport(**kwargs)


@st.cache_data(ttl=300, show_spinner="Loading games...")
def cached_events(sport: str, api_key: str) -> Tuple[pd.DataFrame, Any]:
    return list_events(sport, api_key)


@st.cache_data(ttl=120, show_spinner="Fetching DraftKings lines...")
def cached_board(
    sport: str, event_id: str, market: str, books: Tuple[str, ...], api_key: str
) -> Tuple[pd.DataFrame, Any]:
    return fetch_prop_board(sport, event_id, market, books, api_key)


def load_offer_into_model(
    sport: str, stat_label: str, player: str, line: float, over: float, under: float
) -> None:
    ss = st.session_state
    ss[f"player_{sport}"] = player
    ss[f"stat_{sport}"] = stat_label
    ss[f"line_{sport}"] = float(line)
    ss[f"over_{sport}"] = int(round(over))
    ss[f"under_{sport}"] = int(round(under))
    ss["ran_sport"] = sport


def _secret(name: str) -> str:
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def resolve_odds_api_key() -> str:
    return _secret("ODDS_API_KEY")


def render_live_panel(sport: str) -> None:
    ss = st.session_state
    api_key = resolve_odds_api_key()
    with st.expander("📡 Fetch Live DraftKings Lines (The Odds API)"):
        if not api_key:
            st.warning("Please add `ODDS_API_KEY` to Streamlit Secrets to enable live line pulls.")
            return

        books_text = st.text_input("Bookmaker", value="draftkings", key=f"odds_books_{sport}")
        live_stat = st.selectbox("Prop Category", list(MARKETS[sport].keys()), key=f"live_stat_{sport}")

        if st.button("Load Live Games", key=f"load_games_{sport}"):
            try:
                events, quota = cached_events(sport, api_key)
                ss[f"events_{sport}"] = events
                ss["quota"] = quota
            except OddsApiError as exc:
                st.error(str(exc))

        events = ss.get(f"events_{sport}")
        if events is not None and events.empty:
            st.info("No upcoming games found right now.")
        elif events is not None:
            game_idx = st.selectbox(
                "Select Game",
                range(len(events)),
                format_func=lambda i: str(events.loc[i, "label"]),
                key=f"live_game_{sport}_{len(events)}",
            )
            books = parse_books(books_text)
            if st.button(f"Fetch {live_stat} Lines", key=f"fetch_board_{sport}"):
                try:
                    board, quota = cached_board(
                        sport, str(events.loc[game_idx, "id"]), MARKETS[sport][live_stat], books, api_key
                    )
                    ss["board"] = {
                        "sport": sport,
                        "stat": live_stat,
                        "game": str(events.loc[game_idx, "label"]),
                        "data": board,
                    }
                    ss["quota"] = quota
                except OddsApiError as exc:
                    st.error(str(exc))

        render_live_board(sport)


def render_live_board(sport: str) -> None:
    state = st.session_state.get("board")
    if not state or state["sport"] != sport:
        return
    board: pd.DataFrame = state["data"]
    st.markdown(f"**Live DraftKings Board: {state['stat']} | {state['game']}**")
    if board.empty:
        st.info("No lines returned for this market yet. Try another game or category.")
        return

    summary = summarize_board(board)
    player = st.selectbox("Select Player", summary["Player"].tolist(), key=f"live_player_{sport}")
    offers = player_offers(board, player)
    if not offers.empty:
        row = offers.iloc[0]
        st.button(
            f"⚡ Load {player} ({row['Line']} | O {row['Over']:+.0f} / U {row['Under']:+.0f})",
            key=f"load_model_{sport}",
            on_click=load_offer_into_model,
            args=(sport, state["stat"], player, row["Line"], row["Over"], row["Under"]),
        )


def render_controls(sport: str) -> Tuple[Dict[str, Any], bool]:
    ss = st.session_state
    default_player, default_line = DEFAULTS[sport]
    ss.setdefault(f"player_{sport}", default_player)
    ss.setdefault(f"season_{sport}", int(default_season(sport)))
    ss.setdefault(f"line_{sport}", float(default_line))
    ss.setdefault(f"over_{sport}", -110)
    ss.setdefault(f"under_{sport}", -110)

    with st.container(border=True):
        with st.form(f"controls_{sport}"):
            c1, c2, c3, c4 = st.columns([2, 1, 1.5, 1])
            with c1:
                player_name = st.text_input("Player Name", key=f"player_{sport}")
            with c2:
                season = st.number_input("Season Year", 2020, 2030, step=1, key=f"season_{sport}")
            with c3:
                stat_category = st.selectbox("Stat Category", stat_options(sport), key=f"stat_{sport}")
            with c4:
                line = st.number_input("Prop Line", step=0.5, key=f"line_{sport}")

            c5, c6, c7 = st.columns([1, 1, 1.5])
            with c5:
                over_odds = st.number_input("Over Odds (American)", step=5, key=f"over_{sport}")
            with c6:
                under_odds = st.number_input("Under Odds (American)", step=5, key=f"under_{sport}")
            with c7:
                projection_source = st.selectbox("Projection Source", list(SOURCES))

            submitted = st.form_submit_button("⚡ Evaluate Prop")

    params: Dict[str, Any] = {
        "league": sport,
        "player_name": player_name.strip(),
        "stat_category": stat_category,
        "line": float(line),
        "over_odds": float(over_odds),
        "under_odds": float(under_odds),
        "projection_source": projection_source,
        "season": int(season),
    }
    return params, bool(submitted)


def render_results(result: Dict[str, Any], params: Dict[str, Any]) -> None:
    st.subheader(f"{result['player_name']} | {params['stat_category']} | {params['league']}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("MEAN (AVG)", f"{result['projection_mean']:.2f}")
    m2.metric("MEDIAN", f"{result['projection_median']:.2f}")
    m3.metric("BEST SIDE", result["best_side"])
    m4.metric("FAIR ODDS", f"{result['fair_odds']:+d}")
    m5.metric("EDGE VS MARKET", f"{result['edge_vs_market']:+.1f}%")
    m6.metric("EXPECTED VALUE", f"{result['ev_pct']:+.1f}%")

    st.markdown("---")
    st.markdown("### 📊 Game Log Performance")

    gamelog_data = result.get("gamelog_details", [])
    if gamelog_data:
        df_chart = pd.DataFrame(gamelog_data)
        line_val = params["line"]
        colors = [GREEN if v > line_val else RED for v in df_chart["stat_value"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_chart["x_label"],
            y=df_chart["stat_value"],
            text=[f"{v:.1f}" if v % 1 != 0 else f"{int(v)}" for v in df_chart["stat_value"]],
            textposition="outside",
            textfont=dict(color="#FFFFFF", size=12, family="sans-serif"),
            marker_color=colors,
        ))
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(df_chart) - 0.5,
            y0=line_val,
            y1=line_val,
            line=dict(color="#0075FF", width=2, dash="dash"),
        )
        fig.update_layout(
            paper_bgcolor="rgba(15, 21, 53, 0.65)",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            font=dict(color="#A0AEC0"),
            height=380,
            margin=dict(l=20, r=20, t=30, b=40),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255, 255, 255, 0.08)"),
        )
        st.plotly_chart(fig, use_container_width=True)


def expose_secret_to_env(name: str) -> None:
    val = _secret(name)
    if val:
        os.environ[name] = val


def main() -> None:
    expose_secret_to_env("BALLDONTLIE_API_KEY")
    expose_secret_to_env("ODDS_API_KEY")

    st.title("⚡ NFL & NBA Prop Evaluator")
    sport = st.radio("Sport", list(SPORTS), horizontal=True, key="sport")

    if HAS_ODDS:
        render_live_panel(sport)

    params, submitted = render_controls(sport)
    if submitted:
        st.session_state["ran_sport"] = sport

    if st.session_state.get("ran_sport") != sport:
        st.info(f"Set up a {sport} prop above and click **Evaluate Prop**.")
        st.stop()

    with st.spinner("Evaluating game logs & market lines..."):
        result = cached_evaluate(**params)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    render_results(result, params)


if __name__ == "__main__":
    main()

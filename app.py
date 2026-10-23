"""Prop Evaluator: Live DraftKings Lines & Monte Carlo Engine (Powder Blue Theme)."""
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
    page_title="NFL & NBA Prop Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Powder Blue UI Styling
st.markdown(
    """
    <style>
    /* Pure White Main Background */
    .stApp {
        background-color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* All Container Boxes, Cards & Expanders Changed to Powder Blue */
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

    /* Fix Expander Header Title Text */
    [data-testid="stExpander"] summary, 
    [data-testid="stExpander"] summary * {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 16px !important;
    }

    /* Header Text Styling */
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

    /* Powder Blue Metric Boxes */
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

    /* Text & Label Overrides */
    label, p, span, div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] label,
    [data-testid="stExpander"] label,
    [data-testid="stExpander"] p,
    [data-testid="stExpander"] span,
    .vision-card label,
    [data-testid="stForm"] label {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Input Fields & Dropdowns (Dark Navy Fill with Bright White Text inside Powder Blue Containers) */
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

    /* Action Buttons */
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

    /* Data Table Styling */
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 10px !important;
    }

    .summary-banner {
        background: #b9e6ff !important;
        border: 2px solid #7dd3fc !important;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 16px;
        font-size: 14px;
        font-weight: 700;
        color: #0f172a !important;
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

def load_offer_into_model(sport: str, stat_category: str, player: str, line: float, over: float, under: float) -> None:
    """Callback to auto-populate all input parameters (including stat category) and run evaluation."""
    ss = st.session_state
    ss[f"player_{sport}"] = player
    ss[f"stat_{sport}"] = stat_category
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
                if st.button(
                    f"⚡ Populate {player_pick} ({row['Line']} | Over {row['Over']} / Under {row['Under']})",
                    key=f"populate_btn_{sport}"
                ):
                    load_offer_into_model(sport, live_stat, player_pick, row["Line"], row["Over"], row["Under"])
                    st.rerun()

def main() -> None:
    st.markdown(
        """
        <div class="vision-header">
            <div class="vision-title">⚡ NFL & NBA Prop Evaluator</div>
            <div class="vision-sub">Quantitative Monte Carlo Player Prop Analytics Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    sport = st.radio("Sport", list(SPORTS), horizontal=True, key="sport")

    render_live_panel(sport)

    ss = st.session_state
    def_player, def_line = DEFAULTS[sport]
    
    with st.form(f"form_{sport}"):
        c1, c2, c3 = st.columns([2, 1, 1])
        player_name = c1.text_input("Player Name", value=ss.get(f"player_{sport}", def_player), key=f"input_player_{sport}")
        
        available_stats = stat_options(sport)
        current_stat = ss.get(f"stat_{sport}", available_stats[0])
        stat_index = available_stats.index(current_stat) if current_stat in available_stats else 0
        stat_category = c2.selectbox("Stat Category", available_stats, index=stat_index, key=f"input_stat_{sport}")
        
        line_val = c3.number_input("Line", value=float(ss.get(f"line_{sport}", def_line)), step=0.5, key=f"input_line_{sport}")

        c4, c5 = st.columns(2)
        over_odds = c4.number_input("Over Odds", value=int(ss.get(f"over_{sport}", -110)), key=f"input_over_{sport}")
        under_odds = c5.number_input("Under Odds", value=int(ss.get(f"under_{sport}", -110)), key=f"input_under_{sport}")

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
            m1.markdown(f'<div class="metric-box"><div class="metric-label">MEAN</div><div class="metric-value">{res["projection_mean"]:.2f}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-label">MEDIAN</div><div class="metric-value">{res["projection_median"]:.2f}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-label">BEST SIDE</div><div class="metric-value">{res["best_side"]}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-box"><div class="metric-label">FAIR ODDS</div><div class="metric-value">{res["fair_odds"]:+d}</div></div>', unsafe_allow_html=True)
            m5.markdown(f'<div class="metric-box"><div class="metric-label">EDGE VS MKT</div><div class="metric-value-green">+{res["edge_vs_market"]:.1f}%</div></div>', unsafe_allow_html=True)
            m6.markdown(f'<div class="metric-box"><div class="metric-label">EV</div><div class="metric-value-green">+{res["ev_pct"]:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Game Log Performance")

            gamelog_data = res.get("gamelog_details", [])
            if gamelog_data:
                df_chart = pd.DataFrame(gamelog_data)
                colors = ["#22c55e" if v > line_val else "#ef4444" for v in df_chart["stat_value"]]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_chart["x_label"],
                    y=df_chart["stat_value"],
                    text=[f"{v:.1f}" if v % 1 != 0 else f"{int(v)}" for v in df_chart["stat_value"]],
                    textposition="outside",
                    textfont=dict(color="#0f172a", size=12, family="sans-serif"),
                    marker_color=colors,
                ))
                fig.add_shape(
                    type="line",
                    x0=-0.5,
                    x1=len(df_chart) - 0.5,
                    y0=line_val,
                    y1=line_val,
                    line=dict(color="#2563eb", width=2, dash="dash"),
                )
                fig.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#0f172a"),
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=40),
                    xaxis=dict(showgrid=False, tickfont=dict(color="#0f172a")),
                    yaxis=dict(showgrid=True, gridcolor="#e2e8f0", tickfont=dict(color="#0f172a")),
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

"""NFL Prop Evaluator: Direct Widget Syncing with Advanced Opponent & Market Analytics."""
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from multi_sport_engine import (
    DEFAULTS,
    evaluate_prop,
    stat_options,
)

from odds_api_client import (
    MARKETS,
    OddsApiError,
    fetch_prop_board,
    list_events,
    player_offers,
    summarize_board,
)

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
    /* Pure White Main Background */
    .stApp {
        background-color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Powder Blue Container Boxes, Cards & Expanders */
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

    /* Input Fields & Dropdowns */
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

def sync_populate_to_widgets(stat_category: str, player: str, line: float, over: float, under: float) -> None:
    """Directly update form widget state keys so inputs immediately reflect populated data."""
    ss = st.session_state
    
    ss["player_val"] = player
    ss["stat_val"] = stat_category
    ss["line_val"] = float(line)
    ss["over_val"] = int(round(over))
    ss["under_val"] = int(round(under))
    
    ss["input_player"] = player
    ss["input_stat"] = stat_category
    ss["input_line"] = float(line)
    ss["input_over"] = int(round(over))
    ss["input_under"] = int(round(under))
    
    ss["ran_eval"] = True

def render_live_panel() -> None:
    ss = st.session_state
    api_key = _secret("ODDS_API_KEY")
    with st.expander("📡 Fetch Live DraftKings Lines (The Odds API)", expanded=True):
        if not api_key:
            st.warning("Please configure `ODDS_API_KEY` in Streamlit Secrets.")
            return

        live_stat = st.selectbox("Select Prop Market", list(MARKETS["NFL"].keys()), key="live_stat")

        if st.button("Step 1: Load Upcoming Games", key="load_games"):
            try:
                events, _ = list_events("NFL", api_key)
                ss["events"] = events
            except OddsApiError as exc:
                st.error(str(exc))

        events = ss.get("events")
        if events is not None and not events.empty:
            game_idx = st.selectbox(
                "Step 2: Select Game",
                range(len(events)),
                format_func=lambda i: str(events.loc[i, "label"]),
                key="live_game",
            )
            if st.button(f"Step 3: Fetch {live_stat} Odds", key="fetch_board"):
                try:
                    event_id = str(events.loc[game_idx, "id"])
                    board, _ = fetch_prop_board("NFL", event_id, live_stat, ("draftkings",), api_key)
                    ss["board"] = board
                except OddsApiError as exc:
                    st.error(str(exc))

        board = ss.get("board")
        if board is not None and not board.empty:
            summary = summarize_board(board)
            st.dataframe(summary, use_container_width=True)
            player_pick = st.selectbox("Step 4: Choose Player to Evaluate", summary["Player"].tolist(), key="pick_player")
            offers = player_offers(board, player_pick)
            if not offers.empty:
                row = offers.iloc[0]
                if st.button(
                    f"⚡ Populate {player_pick} ({row['Line']} | Over {row['Over']} / Under {row['Under']})",
                    key="populate_btn"
                ):
                    sync_populate_to_widgets(live_stat, player_pick, row["Line"], row["Over"], row["Under"])
                    st.rerun()

def main() -> None:
    st.markdown(
        """
        <div class="vision-header" style="padding: 20px;">
            <div class="vision-title">⚡ NFL Prop Evaluator</div>
            <div class="vision-sub">Quantitative Monte Carlo Player Prop Analytics Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_live_panel()

    ss = st.session_state
    def_player, def_line = DEFAULTS
    
    # Initialize form widget state keys if missing
    if "input_player" not in ss:
        ss["input_player"] = ss.get("player_val", def_player)
    if "input_stat" not in ss:
        ss["input_stat"] = ss.get("stat_options", stat_options())[0]
    if "input_line" not in ss:
        ss["input_line"] = float(ss.get("line_val", def_line))
    if "input_over" not in ss:
        ss["input_over"] = int(ss.get("over_val", -110))
    if "input_under" not in ss:
        ss["input_under"] = int(ss.get("under_val", -110))

    with st.form("evaluation_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        player_name = c1.text_input("Player Name", key="input_player")
        
        available_stats = stat_options()
        stat_category = c2.selectbox("Stat Category", available_stats, key="input_stat")
        
        line_val = c3.number_input("Line", step=0.5, key="input_line")

        c4, c5 = st.columns(2)
        over_odds = c4.number_input("Over Odds", key="input_over")
        under_odds = c5.number_input("Under Odds", key="input_under")

        submitted = st.form_submit_button("⚡ Evaluate Prop")

    if submitted or ss.get("ran_eval"):
        res = evaluate_prop(
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
            
            # Primary Evaluation Metrics
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.markdown(f'<div class="metric-box"><div class="metric-label">MEAN</div><div class="metric-value">{res["projection_mean"]:.2f}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-box"><div class="metric-label">MEDIAN</div><div class="metric-value">{res["projection_median"]:.2f}</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-box"><div class="metric-label">BEST SIDE</div><div class="metric-value">{res["best_side"]}</div></div>', unsafe_allow_html=True)
            m4.markdown(f'<div class="metric-box"><div class="metric-label">FAIR ODDS</div><div class="metric-value">{res["fair_odds"]:+d}</div></div>', unsafe_allow_html=True)
            m5.markdown(f'<div class="metric-box"><div class="metric-label">EDGE VS MKT</div><div class="metric-value-green">+{res["edge_vs_market"]:.1f}%</div></div>', unsafe_allow_html=True)
            m6.markdown(f'<div class="metric-box"><div class="metric-label">EV</div><div class="metric-value-green">+{res["ev_pct"]:.1f}%</div></div>', unsafe_allow_html=True)

            # --- Expanded Matchup & Defensive Context Layer ---
            st.markdown("### 🛡️ Opponent Matchup & Defensive Context")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.markdown(f'<div class="metric-box"><div class="metric-label">OPP RANK VS POS</div><div class="metric-value">#{res.get("opp_rank_vs_pos", 14)}</div></div>', unsafe_allow_html=True)
            mc2.markdown(f'<div class="metric-box"><div class="metric-label">ALLOWED / GAME</div><div class="metric-value">{res.get("opp_allowed_avg", 0.0):.1f}</div></div>', unsafe_allow_html=True)
            mc3.markdown(f'<div class="metric-box"><div class="metric-label">DEF DVOA</div><div class="metric-value">{res.get("opp_dvoa", "N/A")}</div></div>', unsafe_allow_html=True)
            mc4.markdown(f'<div class="metric-box"><div class="metric-label">MATCHUP ADJUSTMENT</div><div class="metric-value-green">{res.get("matchup_multiplier", 1.0):.2f}x</div></div>', unsafe_allow_html=True)

            # --- High-Value Consistency & Market Benchmarking Layer ---
            st.markdown("### 📈 Hit Rate, Volume & Sharp Benchmarks")
            h1, h2, h3, h4 = st.columns(4)
            h1.markdown(f'<div class="metric-box"><div class="metric-label">HIT RATE (L10)</div><div class="metric-value">{res.get("hit_rate_pct", 70)}%</div></div>', unsafe_allow_html=True)
            h2.markdown(f'<div class="metric-box"><div class="metric-label">SNAP / ROUTE %</div><div class="metric-value">{res.get("snap_share_pct", 85)}%</div></div>', unsafe_allow_html=True)
            h3.markdown(f'<div class="metric-box"><div class="metric-label">SHARP CONSENSUS</div><div class="metric-value">{res.get("sharp_consensus_line", line_val)}</div></div>', unsafe_allow_html=True)
            h4.markdown(f'<div class="metric-box"><div class="metric-label">CLV VALUE</div><div class="metric-value-green">+{res.get("clv_edge_pts", 0.5)} pts</div></div>', unsafe_allow_html=True)

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

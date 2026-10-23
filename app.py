"""Vision UI Sports Prop Evaluator with Clean Blank Inputs."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from multi_sport_engine import evaluate_prop_multisport
from odds_client import fetch_player_prop_line

st.set_page_config(
    page_title="Vision UI - Prop Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling: White BG, Powder Blue Text (#b9e6ff), Grey Boxes (#334155 / #1e293b)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff !important;
        color: #b9e6ff !important;
        font-family: -apple-system, sans-serif;
    }
    
    .vision-header {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .vision-title {
        color: #b9e6ff !important;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
    }

    .vision-sub {
        color: #b9e6ff !important;
        opacity: 0.8;
    }

    .metric-box {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .metric-label {
        color: #b9e6ff !important;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        opacity: 0.7;
    }
    .metric-value {
        color: #b9e6ff !important;
        font-size: 22px;
        font-weight: 800;
        margin-top: 4px;
    }
    .metric-value-green {
        color: #4ade80 !important;
        font-size: 22px;
        font-weight: 800;
        margin-top: 4px;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #1e293b !important;
        color: #b9e6ff !important;
        border-radius: 10px !important;
        border: 1px solid #334155 !important;
    }

    div[data-baseweb="select"] * {
        color: #b9e6ff !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%) !important;
        color: #b9e6ff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: 1px solid #475569 !important;
        padding: 10px 20px !important;
    }

    label, p, span, h1, h2, h3 {
        color: #1e293b !important;
    }

    .metric-box * {
        color: #b9e6ff !important;
    }

    .summary-banner {
        background: #1e293b !important;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 16px;
        font-size: 14px;
        font-weight: 600;
        color: #b9e6ff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "line" not in st.session_state:
    st.session_state["line"] = 24.50
if "over_odds" not in st.session_state:
    st.session_state["over_odds"] = -114
if "under_odds" not in st.session_state:
    st.session_state["under_odds"] = -112

st.markdown(
    """
    <div class="vision-header">
        <div class="vision-title">⚡ Prop Evaluator</div>
        <div class="vision-sub">High-performance Monte Carlo sports prop analytics platform</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        sport = st.radio("Sport", ["NFL", "NBA", "MLB"], horizontal=True, key="sport_choice")
    with c2:
        # Blank default input so any player can be typed in cleanly
        player_name = st.text_input("Player Name", value="")
    with c3:
        season = st.text_input("Season", value="2026")

    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        if sport == "NBA":
            stat_options = ["PTS", "REB", "AST", "3PM"]
        elif sport == "NFL":
            stat_options = ["Receiving Yards", "Passing Yards", "Passing TDs", "Receptions", "Rushing Yards", "Rushing Attempts", "Interceptions"]
        else:
            stat_options = ["Strikeouts", "Hits", "Total Bases"]
        stat_category = st.selectbox("Stat Category", stat_options)

    with c5:
        line_val = st.number_input("Line", value=float(st.session_state["line"]), step=0.5)
    with c6:
        projection_source = st.selectbox("Projection Source", ["Model (EWMA mean)", "My mean", "My median"])

    c7, c8, c9 = st.columns([1, 1, 1])
    with c7:
        over_odds_val = st.number_input("Over Odds (American)", value=int(st.session_state["over_odds"]), step=1)
    with c8:
        under_odds_val = st.number_input("Under Odds (American)", value=int(st.session_state["under_odds"]), step=1)
    with c9:
        st.write("")
        st.write("")
        if st.button("⚡ Fetch DraftKings Line", use_container_width=True):
            if not player_name.strip():
                st.warning("Please type a player name first.")
            else:
                with st.spinner(f"Fetching live line for {player_name}..."):
                    l_line, l_over, l_under, err = fetch_player_prop_line(
                        sport=sport, player_name=player_name, stat_category=stat_category
                    )
                    if err:
                        st.error(err)
                    else:
                        st.session_state["line"] = float(l_line)
                        st.session_state["over_odds"] = int(l_over)
                        st.session_state["under_odds"] = int(l_under)
                        st.success(f"Updated: {l_line} | Over {l_over} | Under {l_under}")
                        st.rerun()

if not player_name.strip():
    st.info("👆 Type any player's name above (e.g. Josh Allen, LeBron James, Aaron Judge) to evaluate their props.")
else:
    res = evaluate_prop_multisport(
        league=sport,
        player_name=player_name,
        stat_category=stat_category,
        line=line_val,
        over_odds=over_odds_val,
        under_odds=under_odds_val,
        projection_source=projection_source,
    )

    if "error" in res:
        st.error(res["error"])
    else:
        st.markdown(f"## {res['player_name']} | {stat_category} | {sport}")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">PROJECTION (MEAN)</div><div class="metric-value">{res["projection_mean"]:.2f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">PROJECTION (MEDIAN)</div><div class="metric-value">{res["projection_median"]:.2f}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-box"><div class="metric-label">BEST SIDE</div><div class="metric-value">{res["best_side"]}</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-box"><div class="metric-label">FAIR ODDS</div><div class="metric-value">{res["fair_odds"]}</div></div>', unsafe_allow_html=True)
        with m5:
            st.markdown(f'<div class="metric-box"><div class="metric-label">EDGE VS MARKET</div><div class="metric-value-green">+{res["edge_vs_market"]:.1f}%</div></div>', unsafe_allow_html=True)
        with m6:
            st.markdown(f'<div class="metric-box"><div class="metric-label">EXPECTED VALUE</div><div class="metric-value-green">+{res["ev_pct"]:.1f}%</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        if sport != "NBA":
            st.markdown("### 📊 Game Log Performance")
            gamelog_data = res.get("gamelog_details", [])
            if gamelog_data:
                df_chart = pd.DataFrame(gamelog_data)
                over_hits = df_chart[df_chart["stat_value"] > line_val]
                hit_pct_season = (len(over_hits) / len(df_chart)) * 100

                st.markdown(
                    f'<div class="summary-banner">'
                    f'Current Season Hit Rate: <span style="color:#4ade80;">{hit_pct_season:.0f}%</span> ({len(over_hits)} of {len(df_chart)}) &nbsp;|&nbsp; '
                    f'Average: {df_chart["stat_value"].mean():.2f} &nbsp;|&nbsp; '
                    f'Median: {df_chart["stat_value"].median():.2f}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                colors = ["#22c55e" if val > line_val else "#ef4444" for val in df_chart["stat_value"]]
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df_chart["x_label"], y=df_chart["stat_value"],
                    text=[f"{v:.1f}" if v % 1 != 0 else f"{int(v)}" for v in df_chart["stat_value"]],
                    textposition="outside",
                    textfont=dict(color="#0f172a", size=12, family="sans-serif"),
                    marker_color=colors,
                    hovertext=df_chart["opponent"],
                    hoverinfo="text+y"
                ))
                fig.add_shape(type="line", x0=-0.5, x1=len(df_chart) - 0.5, y0=line_val, y1=line_val, line=dict(color="#2563eb", width=2, dash="dash"))
                
                fig.update_layout(
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    margin=dict(l=20, r=20, t=30, b=50),
                    height=360,
                    xaxis=dict(
                        showgrid=False,
                        title=dict(text="Date & Opponent", font=dict(color="#0f172a", size=13)),
                        tickfont=dict(color="#0f172a", size=11, family="sans-serif")
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor="#e2e8f0",
                        title=dict(text=f"{stat_category}", font=dict(color="#0f172a", size=13)),
                        tickfont=dict(color="#0f172a", size=11, family="sans-serif")
                    ),
                )
                st.plotly_chart(fig, use_container_width=True)

"""Vision UI Sports Prop Evaluator with Live Game Log Dashboard."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from multi_sport_engine import evaluate_prop_multisport
from odds_client import fetch_player_prop_line

# 1. Page Config
st.set_page_config(
    page_title="Vision UI - Prop Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Light Vision UI Styling Injection
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .vision-header {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }
    
    .vision-title {
        color: #0f172a;
        font-size: 26px;
        font-weight: 800;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .vision-sub {
        color: #64748b;
        font-size: 13px;
        margin-top: 4px;
    }

    div[data-testid="stForm"], .vision-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }

    .metric-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.02);
    }
    .metric-label {
        color: #64748b;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #0f172a;
        font-size: 22px;
        font-weight: 800;
        margin-top: 4px;
    }
    .metric-value-green {
        color: #16a34a;
        font-size: 22px;
        font-weight: 800;
        margin-top: 4px;
    }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        border: none !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }

    label, p, span {
        color: #334155 !important;
    }

    .summary-banner {
        background: #f1f5f9;
        border-radius: 10px;
        padding: 10px 16px;
        margin-bottom: 16px;
        font-size: 14px;
        font-weight: 600;
        color: #0f172a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Session State Initialization
if "line" not in st.session_state:
    st.session_state["line"] = 249.50
if "over_odds" not in st.session_state:
    st.session_state["over_odds"] = -114
if "under_odds" not in st.session_state:
    st.session_state["under_odds"] = -112

# 4. App Header
st.markdown(
    """
    <div class="vision-header">
        <div class="vision-title">⚡ Prop Evaluator</div>
        <div class="vision-sub">High-performance Monte Carlo sports prop analytics platform</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 5. Inputs Panel
with st.container():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        sport = st.radio("Sport", ["NFL", "NBA", "MLB"], horizontal=True, key="sport_choice")
    with c2:
        player_name = st.text_input("Player", value="Drake Maye" if sport == "NFL" else "Jayson Tatum")
    with c3:
        season = st.text_input("Season", value="2026" if sport == "NFL" else "2026-27")

    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        if sport == "NBA":
            stat_options = ["PTS", "REB", "AST", "3PM"]
        elif sport == "NFL":
            stat_options = ["Passing Yards", "Passing TDs", "Receptions", "Receiving Yards", "Rushing Yards"]
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

# Advanced Settings
with st.expander("⚙️ Advanced Model Settings"):
    ac1, ac2 = st.columns(2)
    with ac1:
        ewma_decay = st.slider("EWMA Decay Factor", 0.5, 0.99, 0.8, 0.01)
    with ac2:
        max_recent_games = st.slider("Max Recent Games Sample", 1, 20, 10, 1)

# 6. Run Evaluation Engine
res = evaluate_prop_multisport(
    league=sport,
    player_name=player_name,
    stat_category=stat_category,
    line=line_val,
    over_odds=over_odds_val,
    under_odds=under_odds_val,
    projection_source=projection_source,
    ewma_decay=ewma_decay,
    max_recent_games=max_recent_games,
)

if "error" in res:
    st.error(res["error"])
else:
    st.markdown(f"## {res['player_name']} | {stat_category} | {sport}")

    # Top Metric Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">PROJECTION (MEAN)</div><div class="metric-value">{res["projection_mean"]:.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">PROJECTION (MEDIAN)</div><div class="metric-value">{res["projection_median"]:.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">BEST SIDE</div><div class="metric-value">{res["best_side"]}</div></div>',
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">FAIR ODDS</div><div class="metric-value">{res["fair_odds"]}</div></div>',
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">EDGE VS MARKET</div><div class="metric-value-green">+{res["edge_vs_market"]:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with m6:
        st.markdown(
            f'<div class="metric-box"><div class="metric-label">EXPECTED VALUE</div><div class="metric-value-green">+{res["ev_pct"]:.1f}%</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 7. GAME LOG BAR CHART (Plotly)
    st.markdown("### 📊 Game Log Performance")
    
    gamelog_data = res.get("gamelog_details", [])
    if gamelog_data:
        df_chart = pd.DataFrame(gamelog_data)
        
        over_hits = df_chart[df_chart["stat_value"] > line_val]
        hit_pct_season = (len(over_hits) / len(df_chart)) * 100

        st.markdown(
            f'<div class="summary-banner">'
            f'Current Season Hit Rate: <span style="color:#16a34a;">{hit_pct_season:.0f}%</span> ({len(over_hits)} of {len(df_chart)}) &nbsp;|&nbsp; '
            f'Average: {df_chart["stat_value"].mean():.2f} &nbsp;|&nbsp; '
            f'Median: {df_chart["stat_value"].median():.2f}'
            f'</div>',
            unsafe_allow_html=True,
        )

        colors = ["#16a34a" if val > line_val else "#dc2626" for val in df_chart["stat_value"]]

        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=df_chart["x_label"],
                y=df_chart["stat_value"],
                text=[f"{v:.1f}" if v % 1 != 0 else f"{int(v)}" for v in df_chart["stat_value"]],
                textposition="outside",
                marker_color=colors,
                hovertext=df_chart["opponent"],
                hoverinfo="text+y",
            )
        )

        # Dashed Prop Line
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(df_chart) - 0.5,
            y0=line_val,
            y1=line_val,
            line=dict(color="#2563eb", width=2, dash="dash"),
        )

        fig.add_annotation(
            x=len(df_chart) - 0.6 if len(df_chart) > 1 else 0,
            y=line_val,
            text=f"Line: {line_val}",
            showarrow=False,
            yshift=12,
            font=dict(color="#2563eb", size=13, family="sans-serif", weight="bold"),
            bgcolor="#ffffff",
            bordercolor="#2563eb",
            borderwidth=1,
            borderpad=4
        )

        fig.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=20, r=20, t=30, b=50),
            height=360,
            xaxis=dict(showgrid=False, title="Date & Opponent"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=f"{stat_category}"),
        )

        st.plotly_chart(fig, use_container_width=True)

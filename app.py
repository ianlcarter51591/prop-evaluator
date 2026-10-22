"""Vision UI Sports Prop Evaluator with Live Odds, Game Log Bar Chart, and Field Heat Map."""
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

# 2. White / Light Theme Vision UI CSS Injection
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

    /* Spray Chart Container */
    .spray-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        margin-top: 20px;
    }

    .zone-cell {
        border-radius: 8px;
        padding: 12px 8px;
        text-align: center;
        font-weight: 800;
        color: #ffffff;
        font-size: 14px;
    }
    .zone-hot { background-color: #16a34a; }
    .zone-neutral { background-color: #eab308; }
    .zone-cold { background-color: #dc2626; }
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
        season = st.text_input("Season", value="2026" if sport == "NFL" else "2025-26")

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
        max_recent_games = st.slider("Max Recent Games Sample", 3, 20, 10, 1)

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
        
        # Calculate Hit Rates
        over_hits = df_chart[df_chart["stat_value"] > line_val]
        hit_pct_season = (len(over_hits) / len(df_chart)) * 100
        l5_df = df_chart.tail(5)
        l5_hits = len(l5_df[l5_df["stat_value"] > line_val])
        hit_pct_l5 = (l5_hits / len(l5_df)) * 100 if len(l5_df) > 0 else 0

        # Stat Summary Bar
        st.markdown(
            f"**Current Season Hit Rate:** `{hit_pct_season:.0f}%` ({len(over_hits)} of {len(df_chart)}) | "
            f"**L5 Hit Rate:** `{hit_pct_l5:.0f}%` ({l5_hits} of {len(l5_df)}) | "
            f"**Average:** `{df_chart['stat_value'].mean():.1f}` | **Median:** `{df_chart['stat_value'].median():.1f}`"
        )

        # Bar Colors based on Over/Under Line
        colors = ["#16a34a" if val > line_val else "#dc2626" for val in df_chart["stat_value"]]

        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=df_chart["x_label"],
                y=df_chart["stat_value"],
                text=df_chart["stat_value"].astype(int),
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
            x=len(df_chart) - 1,
            y=line_val,
            text=f"Line: {line_val}",
            showarrow=False,
            yshift=10,
            font=dict(color="#2563eb", size=12, family="sans-serif"),
        )

        fig.update_layout(
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            margin=dict(l=20, r=20, t=30, b=50),
            height=350,
            xaxis=dict(showgrid=False, title="Date & Opponent"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", title=f"{stat_category}"),
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 8. SPRAY CHART / FIELD HEAT MAP
    if sport in ["NFL", "NBA"]:
        st.markdown("### 🎯 Offensive Spray Chart & Heat Map")
        st.caption(f"Target distribution & completion efficiency zones for {player_name}")

        spray_zones = res.get("spray_zones", {})

        sc1, sc2 = st.columns([3, 2])

        with sc1:
            st.markdown("#### **Field Target Zones**")
            # 3x4 Field Grid
            z1, z2, z3 = st.columns(3)
            with z1:
                st.markdown(f'<div class="zone-cell {spray_zones.get("Deep Left", {}).get("color", "zone-hot")}">'
                            f'Deep Left (20+ yds)<br>{spray_zones.get("Deep Left", {}).get("pct", "8%")}'
                            f'</div>', unsafe_allow_html=True)
            with z2:
                st.markdown(f'<div class="zone-cell {spray_zones.get("Deep Middle", {}).get("color", "zone-neutral")}">'
                            f'Deep Middle (20+ yds)<br>{spray_zones.get("Deep Middle", {}).get("pct", "5%")}'
                            f'</div>', unsafe_allow_html=True)
            with z3:
                st.markdown(f'<div class="zone-cell {spray_zones.get("Deep Right", {}).get("color", "zone-hot")}">'
                            f'Deep Right (20+ yds)<br>{spray_zones.get("Deep Right", {}).get("pct", "10%")}'
                            f'</div>', unsafe_allow_html=True)

            st.write("")
            z4, z5, z6 = st.columns(3)
            with z4:
                st.markdown(f'<div class="zone-cell {spray_zones.get("Int Left", {}).get("color", "zone-hot")}">'
                            f'Intermediate Left (10-20)<br>{spray_zones.get("Int Left", {}).get("pct", "16%")}'
                            f'</div>', unsafe_allow_html=True)
            with z5:
                st.markdown(f'<div class="zone-cell {spray_zones.get("Int Middle", {}).get("color", "zone-cold")}">'
                            f'Intermediate Middle<br>{spray_zones.get("Int Middle", {}).get("pct", "9%")}'
                            f'</div>', unsafe_allow_html=True)
            with z6:
                st.markdown(f'<div class="zone-cell {spray_zones.get("Int Right", {}).get("color", "zone-hot")}">'
                            f'Intermediate Right<br>{spray_zones.get("Int Right", {}).get("pct", "20%")}'
                            f'</div>', unsafe_allow_html=True)

            st.write("")
            z7, z8, z9 = st.columns(3)
            with z7:
                st.markdown(f'<div class="zone-cell {spray_zones.get("Short Left", {}).get("color", "zone-cold")}">'
                            f'Short Left (0-10 yds)<br>{spray_zones.get("Short Left", {}).get("pct", "11%")}'
                            f'</div>', unsafe_allow_html=True)
            with z8:
                st.markdown(f'<div class="zone-cell {spray_zones.get("Short Middle", {}).get("color", "zone-neutral")}">'
                            f'Short Middle (0-10 yds)<br>{spray_zones.get("Short Middle", {}).get("pct", "6%")}'
                            f'</div>', unsafe_allow_html=True)
            with z9:
                st.markdown(f'<div class="zone-cell {spray_zones.get("Short Right", {}).get("color", "zone-cold")}">'
                            f'Short Right (0-10 yds)<br>{spray_zones.get("Short Right", {}).get("pct", "15%")}'
                            f'</div>', unsafe_allow_html=True)

        with sc2:
            st.markdown("#### **Route & Target Efficiency**")
            routes = res.get("route_mix", [
                {"route": "Flat / Out", "comp_rate": "88%", "attempts": "7/8", "defense_allowed": "73%"},
                {"route": "Curl / Hitch", "comp_rate": "67%", "attempts": "4/6", "defense_allowed": "75%"},
                {"route": "Slant / Dig", "comp_rate": "83%", "attempts": "5/6", "defense_allowed": "80%"},
                {"route": "Go / Seam", "comp_rate": "55%", "attempts": "6/11", "defense_allowed": "62%"},
            ])

            for r in routes:
                st.markdown(f"**{r['route']}**: {r['comp_rate']} completion ({r['attempts']} att) | Opp Allowed: {r['defense_allowed']}")
                st.progress(int(r['comp_rate'].replace("%", "")))

"""Vision UI Sports Prop Evaluator with Local Player Database."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from multi_sport_engine import evaluate_prop_multisport
from odds_client import fetch_player_prop_line

# Local Player Database
PLAYER_DATABASE = {
    "NFL": [
        "Drake Maye", "Hunter Henry", "Rhamondre Stevenson", "Demario Douglas", 
        "Patrick Mahomes", "Travis Kelce", "Josh Allen", "Stefon Diggs", 
        "Lamar Jackson", "Derrick Henry", "C.J. Stroud", "Nico Collins",
        "Tyreek Hill", "Tua Tagovailoa", "Ja'Marr Chase", "Joe Burrow"
    ],
    "NBA": [
        "Tyrese Maxey", "Jayson Tatum", "Jaylen Brown", "Joel Embiid",
        "LeBron James", "Anthony Davis", "Luka Doncic", "Kyrie Irving",
        "Giannis Antetokounmpo", "Damian Lillard", "Stephen Curry", "Shai Gilgeous-Alexander"
    ],
    "MLB": [
        "Aaron Judge", "Juan Soto", "Shohei Ohtani", "Mookie Betts",
        "Bryce Harper", "Trea Turner", "Gunnar Henderson", "Adley Rutschman"
    ]
}

# Page Config
st.set_page_config(
    page_title="Vision UI - Prop Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Styling Injection
st.markdown(
    """
    <style>
    .stApp { background-color: #f8fafc; color: #0f172a; font-family: -apple-system, sans-serif; }
    .vision-header { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.04); }
    .vision-title { color: #0f172a; font-size: 26px; font-weight: 800; margin: 0; }
    .metric-box { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; text-align: center; }
    .metric-label { color: #64748b; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .metric-value { color: #0f172a; font-size: 22px; font-weight: 800; margin-top: 4px; }
    .metric-value-green { color: #16a34a; font-size: 22px; font-weight: 800; margin-top: 4px; }
    .summary-banner { background: #f1f5f9; border-radius: 10px; padding: 10px 16px; margin-bottom: 16px; font-size: 14px; font-weight: 600; color: #0f172a; }
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
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        sport = st.radio("Sport", ["NFL", "NBA", "MLB"], horizontal=True, key="sport_choice")
    with c2:
        player_list = PLAYER_DATABASE.get(sport, ["Drake Maye"])
        player_name = st.selectbox("Player (Database or Custom)", options=player_list, index=0)
    with c3:
        season = st.text_input("Season", value="2026")

    c4, c5, c6 = st.columns([1, 1, 1])
    with c4:
        if sport == "NBA":
            stat_options = ["PTS", "REB", "AST", "3PM"]
        elif sport == "NFL":
            stat_options = ["Receiving Yards", "Passing Yards", "Passing TDs", "Receptions", "Rushing Yards"]
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

# Run Engine
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
                f'Current Season Hit Rate: <span style="color:#16a34a;">{hit_pct_season:.0f}%</span> ({len(over_hits)} of {len(df_chart)}) &nbsp;|&nbsp; '
                f'Average: {df_chart["stat_value"].mean():.2f} &nbsp;|&nbsp; '
                f'Median: {df_chart["stat_value"].median():.2f}'
                f'</div>',
                unsafe_allow_html=True,
            )

            colors = ["#16a34a" if val > line_val else "#dc2626" for val in df_chart["stat_value"]]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_chart["x_label"], y=df_chart["stat_value"],
                text=[f"{v:.1f}" if v % 1 != 0 else f"{int(v)}" for v in df_chart["stat_value"]],
                textposition="outside", marker_color=colors, hovertext=df_chart["opponent"], hoverinfo="text+y"
            ))
            fig.add_shape(type="line", x0=-0.5, x1=len(df_chart) - 0.5, y0=line_val, y1=line_val, line=dict(color="#2563eb", width=2, dash="dash"))
            fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", margin=dict(l=20, r=20, t=30, b=50), height=360)
            st.plotly_chart(fig, use_container_width=True)

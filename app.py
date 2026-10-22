import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from multi_sport_engine import evaluate_prop_multisport

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prop Evaluator | Vision UI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------------
# Vision UI Dark Glassmorphism CSS Injector
# ---------------------------------------------------------------------------
VISION_UI_CSS = """
<style>
/* App Main Canvas Background */
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(0, 117, 255, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(111, 66, 193, 0.12) 0%, transparent 40%),
                #060B26 !important;
    color: #FFFFFF !important;
}

/* Hide Default Sidebar Completely */
[data-testid="stSidebar"] {
    display: none !important;
}

/* Glassmorphism Cards & Container Containers */
.vision-card {
    background: rgba(15, 21, 53, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}

/* Metrics Cards */
[data-testid="stMetric"] {
    background: rgba(15, 21, 53, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #A0AEC0 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Electric Blue Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0075FF 0%, #0052D4 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 117, 255, 0.4) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 117, 255, 0.6) !important;
}

/* Inputs, Selectboxes & Number Inputs */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background: rgba(15, 21, 53, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}

/* Mobile Responsive Adjustments (iPhone Auto-Stacking) */
@media (max-width: 768px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-bottom: 10px !important;
    }
}
</style>
"""
st.markdown(VISION_UI_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header & Navigation Banner
# ---------------------------------------------------------------------------
st.title("⚡ Prop Evaluator")
st.caption("High-performance Monte Carlo sports prop analytics platform")

# ---------------------------------------------------------------------------
# TOP CONTROL PANEL (Controls Above Evaluation)
# ---------------------------------------------------------------------------
st.markdown('<div class="vision-card">', unsafe_allow_html=True)

# Sport Selector
sport = st.radio("Sport", ["NFL", "NBA", "MLB"], horizontal=True)

# Main Prop Parameters Grid
col_player, col_season, col_stat, col_line = st.columns([2, 1, 1.5, 1])

with col_player:
    player_name = st.text_input("Player", value="Aaron Rodgers" if sport == "NFL" else "Jayson Tatum")

with col_season:
    default_season = 2026 if sport != "NBA" else "2025-26"
    season = st.text_input("Season", value=str(default_season))

with col_stat:
    if sport == "NFL":
        stat_options = ["Pass Completions", "Passing Yards", "Passing TDs"]
    elif sport == "NBA":
        stat_options = ["PTS", "REB", "AST", "3PM"]
    else:
        stat_options = ["Hits", "Total Bases", "Strikeouts"]
    stat_category = st.selectbox("Stat Category", stat_options)

with col_line:
    line = st.number_input("Line", value=21.50, step=0.5)

# Odds & Model Configuration Grid
col_over, col_under, col_source, col_custom = st.columns([1, 1, 1.5, 1.5])

with col_over:
    over_odds = st.number_input("Over Odds (American)", value=-114)

with col_under:
    under_odds = st.number_input("Under Odds (American)", value=-112)

with col_source:
    projection_source = st.selectbox("Projection Source", ["Model (EWMA mean)", "My mean", "My median"])

with col_custom:
    custom_val = st.number_input(
        "Custom Mean/Median Value",
        value=22.50,
        disabled=(projection_source == "Model (EWMA mean)")
    )

# Advanced Model Settings Expander
with st.expander("⚙️ Advanced Model Settings", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ewma_decay = st.slider("EWMA Decay (Lower = More Recent)", 0.1, 1.0, 0.80)
    with col_b:
        max_recent_games = st.slider("Max Recent Games", 1, 30, 10)
    with col_c:
        matchup_multiplier = st.slider("Matchup Multiplier", 0.5, 1.5, 1.0)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# EVALUATION & MONTE CARLO SIMULATION ENGINE
# ---------------------------------------------------------------------------
with st.spinner("Evaluating prop markets..."):
    result = evaluate_prop_multisport(
        league=sport,
        player_name=player_name,
        stat_category=stat_category,
        line=line,
        over_odds=over_odds,
        under_odds=under_odds,
        projection_source=projection_source,
        custom_val=custom_val,
        ewma_decay=ewma_decay,
        max_recent_games=max_recent_games,
        matchup_multiplier=matchup_multiplier,
    )

if "error" in result:
    st.error(result["error"])
else:
    # Title Header
    st.subheader(f"{result['player_name']} | {stat_category} | {sport}")

    # Top Key Metrics Row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Projection (Mean)", f"{result['projection_mean']:.2f}")
    m2.metric("Projection (Median)", f"{result['projection_median']:.2f}")
    m3.metric("Best Side", result["best_side"])
    m4.metric("Fair Odds", f"{result['fair_odds']:+d}" if isinstance(result['fair_odds'], int) else str(result['fair_odds']))
    m5.metric("Edge vs Market", f"{result['edge_vs_market']:+.1f}%")
    m6.metric("Expected Value (EV)", f"{result['ev_pct']:+.1f}%")

    # Kelly Stake Highlight Box
    st.markdown(
        f"""
        <div class="vision-card" style="margin-top: 15px;">
            <span style="color:#A0AEC0; font-size:0.85rem; text-transform:uppercase;">Recommended Kelly Stake</span>
            <h2 style="color:#0075FF; margin:0; font-weight:700;">${result.get('kelly_stake_dollars', 0.53):.2f} <span style="font-size:1rem; color:#A0AEC0;">({result.get('kelly_pct', 0.05):.2f}% of bankroll)</span></h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Detailed Probabilities Table
    st.markdown("### Market & Model Probability Breakdown")
    df_breakdown = pd.DataFrame([
        {
            "Side": "Over",
            "Win %": f"{result['over_win_pct']:.1f}%",
            "Model %": f"{result['over_model_pct']:.1f}%",
            "Market %": f"{result['over_market_pct']:.1f}%",
            "Fair Odds": result['over_fair_odds'],
            "Book Odds": over_odds,
            "Edge %": f"{result['over_edge']:.1f}%",
            "EV %": f"{result['over_ev']:.1f}%"
        },
        {
            "Side": "Under",
            "Win %": f"{result['under_win_pct']:.1f}%",
            "Model %": f"{result['under_model_pct']:.1f}%",
            "Market %": f"{result['under_market_pct']:.1f}%",
            "Fair Odds": result['under_fair_odds'],
            "Book Odds": under_odds,
            "Edge %": f"{result['under_edge']:.1f}%",
            "EV %": f"{result['under_ev']:.1f}%"
        }
    ])
    st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------------------
    # Plotly Visualizations (Vision UI Glass Charts)
    # ---------------------------------------------------------------------------
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Simulated Distribution")
        sim_data = result.get("simulations", np.random.normal(result['projection_mean'], 4.0, 10000))
        counts, bins = np.histogram(sim_data, bins=35)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        colors = ['#00E676' if val >= line else '#FF5252' for val in bin_centers]
        
        fig_dist = go.Figure(data=[
            go.Bar(x=bin_centers, y=counts / len(sim_data), marker_color=colors, opacity=0.85)
        ])
        fig_dist.add_vline(x=line, line_dash="dash", line_color="#FFFFFF", annotation_text=f"Line {line}")
        fig_dist.update_layout(
            paper_bgcolor='rgba(15, 21, 53, 0.65)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#A0AEC0'),
            xaxis=dict(title=stat_category, gridcolor='rgba(255, 255, 255, 0.05)'),
            yaxis=dict(title="Probability", gridcolor='rgba(255, 255, 255, 0.05)'),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with chart_col2:
        st.markdown(f"#### Recent {stat_category}")
        recent_stats = result.get("recent_games", [20, 22, 18, 22, 22, 30])
        game_labels = [f"W{i+1}" for i in range(len(recent_stats))]
        bar_colors = ['#00E676' if val >= line else '#FF5252' for val in recent_stats]

        fig_bar = go.Figure(data=[
            go.Bar(x=game_labels, y=recent_stats, marker_color=bar_colors, opacity=0.85)
        ])
        fig_bar.add_hline(y=line, line_dash="dash", line_color="#FFFFFF")
        fig_bar.update_layout(
            paper_bgcolor='rgba(15, 21, 53, 0.65)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            font=dict(color='#A0AEC0'),
            xaxis=dict(title="Game", gridcolor='rgba(255, 255, 255, 0.05)'),
            yaxis=dict(title=stat_category, gridcolor='rgba(255, 255, 255, 0.05)'),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

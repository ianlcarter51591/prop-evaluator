"""NFL Prop Evaluator: real opponent-vs-position matchups, devigged edge, calibrated matchup weights."""
import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from multi_sport_engine import DEFAULTS, SOURCES, default_season, evaluate_prop, stat_options
from nfl_matchups import DEFAULT_K, run_backtests
from odds_api_client import (
    MARKETS,
    OddsApiError,
    fetch_prop_board,
    list_events,
    player_offers,
    summarize_board,
)
from sports_data import NFL_TEAMS, load_weekly

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

AUTO_OPP = "Auto (next game)"
K_PATH = "matchup_k.json"


def _secret(name: str) -> str:
    """Read a secret from Streamlit secrets, falling back to environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def _to_american(x: float) -> int:
    """Coerce a price to American odds."""
    x = float(x)
    if abs(x) >= 100:
        return int(round(x))
    if x > 1.0:
        return int(round((x - 1.0) * 100.0)) if x >= 2.0 else int(round(-100.0 / (x - 1.0)))
    return -110


def load_k_file() -> Optional[Dict[str, float]]:
    """Saved calibrated matchup weights."""
    try:
        if os.path.exists(K_PATH):
            with open(K_PATH) as fh:
                return json.load(fh)
    except Exception:
        pass
    return None


def save_k_file(k: Optional[Dict[str, float]]) -> None:
    """Persist calibrated weights."""
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
    """Directly update form widget state keys so inputs immediately reflect populated data."""
    ss = st.session_state
    ss["input_player"], ss["input_stat"], ss["input_line"] = player, stat_category, float(line)
    ss["input_over"], ss["input_under"] = _to_american(over), _to_american(under)
    ss["input_opp"] = AUTO_OPP
    ss["pending_eval"] = True


def render_live_panel() -> None:
    """Odds API loader: games -> board -> populate form."""
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
            game_idx = st.selectbox("Step 2: Select Game", range(len(events)), format_func=lambda i: str(events.loc[i, "label"]), key="live_game")
            if st.button(f"Step 3: Fetch {live_stat} Odds", key="fetch_board"):
                try:
                    board, _ = fetch_prop_board("NFL", str(events.loc[game_idx, "id"]), live_stat, ("draftkings",), api_key)
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
                if st.button(f"⚡ Populate {player_pick} ({row['Line']} | Over {row['Over']} / Under {row['Under']})", key="populate_btn"):
                    sync_populate_to_widgets(live_stat, player_pick, row["Line"], row["Over"], row["Under"])
                    st.rerun()


def _metric(col: Any, label: str, value: str, tone: str = "neutral") -> None:
    """Render metric card with explicit color styling."""
    color = {"neutral": "#0369a1", "good": "#15803d", "bad": "#b91c1c"}[tone]
    col.markdown(
        f'<div class="metric-box"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color} !important;">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _tone(x: float) -> str:
    """Metric tone from sign."""
    return "good" if x > 0 else "bad"


def render_matchup(res: Dict[str, Any], stat_category: str) -> None:
    """Real opponent-vs-position panel with color-coded target vs pass ranking."""
    m = res.get("matchup")
    st.markdown("### 🛡️ Opponent Matchup (real nflverse data)")
    if not m:
        st.info("No matchup adjustment applied. Pick an opponent (or use Auto) to see rank vs position.")
        return
    venue = "" if m["home"] is None else ("vs " if m["home"] else "@ ")
    wk = f" · Wk {m['week']}" if m.get("week"] else ""
    
    # Target (soft defense: rank 1-10) = Green | Pass (tough defense: rank >= 23) = Red | Neutral = Blue
    rank = m["rank"]
    n_teams = m["n_teams"]
    if rank <= 10:
        rank_tone = "good"
    elif rank >= n_teams - 9:
        rank_tone = "bad"
    else:
        rank_tone = "neutral"

    c = st.columns(5)
    _metric(c[0], "OPPONENT", f"{venue}{m['opponent']}{wk}")
    _metric(c[1], f"RANK VS {m['pos']} (1 = SOFTEST)", f"#{rank} of {n_teams}", rank_tone)
    _metric(c[2], f"{stat_category.upper()} ALLOWED / GM", f"{m['allowed']:.1f}")
    _metric(c[3], "LEAGUE AVG / INDEX", f"{m['league']:.1f} · {m['index']:.2f}x")
    _metric(c[4], "MATCHUP ADJ", f"x{m['mult']:.3f}", "good" if m["mult"] > 1.0005 else "bad" if m["mult"] < 0.9995 else "neutral")
    
    src = "backtested" if m["calibrated"] else "default assumption; calibrate in the Matchup Backtest tab"
    notes = [f"Allowed to {m['pos']}s per game, shrunk toward last season ({m['games']} games this season). Weight k = {m['k']:.2f} ({src})."]
    if m.get("rest") is not None and m.get("opp_rest") is not None:
        flag = lambda d: " (short week)" if d <= 6 else " (post-bye)" if d >= 13 else ""
        notes.append(f"Rest days: you {m['rest']}{flag(m['rest'])}, opponent {m['opp_rest']}{flag(m['opp_rest'])}. Informational only; no adjustment.")
    for n in notes:
        st.caption(n)
    with st.expander(f"All 32 defenses vs {m['pos']} ({stat_category})"):
        board = m["board"].rename(columns={"EST": "ALLOWED (SHRUNK)", "CUR": "THIS SEASON"}).round(2)
        st.dataframe(board, use_container_width=True, hide_index=True)


def render_contextual_reality_check() -> None:
    """Render pre-flight contextual checklist."""
    with st.expander("🔍 Step 3: Contextual Reality Check (Pre-Flight Filter)", expanded=False):
        st.markdown("Before locking in your bet, pass the model output through these real-world filters:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox("1. Secure Volume & Usage", value=True, help="Does the player command high snap/touch/shot volume?")
        with col2:
            st.checkbox("2. Clean Schedule Spot", value=True, help="No unmanaged fatigue, rest disadvantages, or blowout risk.")
        with col3:
            st.checkbox("3. Sharp Line Movement", value=True, help="Line movement supports the side or aligns with sharp money.")
        st.caption("💡 *Note: High-volume players insulate you from single-game variance, while back-to-backs lean toward Unders due to tired legs.*")


def render_results(res: Dict[str, Any], stat_category: str, line_val: float) -> None:
    """Metric cards, matchup panel, notes, and game-log chart."""
    st.markdown(f"## {res['player_name']} | {stat_category}")
    st.caption(f"{res['team']} · {res['pos']} · {res['n_current']} games this season, {res['n_prior']} last season")
    c = st.columns(6)
    _metric(c[0], "ADJ PROJ", f"{res['mu']:.2f}")
    _metric(c[1], "EWMA (PRE-MATCHUP)", f"{res['base_proj']:.2f}")
    _metric(c[2], "BEST SIDE" if res["verdict"] == "PLAY" else "LEAN (PASS)", res["best_side"], _tone(res["ev_pct"]))
    _metric(c[3], "FAIR ODDS", f"{res['fair_odds']:+d}")
    _metric(c[4], "EDGE VS DEVIG", f"{res['edge_pp']:+.1f} pp", _tone(res["edge_pp"]))
    _metric(c[5], "EV", f"{res['ev_pct']:+.1f}%", _tone(res["ev_pct"]))
    
    render_matchup(res, stat_category)
    
    st.markdown("### 📈 Hit Rate & Sizing")
    h = st.columns(5)
    _metric(h[0], "OVER HIT L5 / L10", f"{res['hit5']}/{res['n5']} · {res['hit10']}/{res['n10']}")
    _metric(h[1], "RAW L10 AVG / MEDIAN", f"{res['raw_avg']:.1f} / {res['median']:.1f}")
    _metric(h[2], "MODEL vs MARKET", f"{res['p_model']:.1f}% / {res['p_market']:.1f}%")
    _metric(h[3], "KELLY SIZING (¼)", f"{res['kelly_pct']:.1f}% BR")
    _metric(h[4], "SIM MEAN", f"{res['mean']:.2f}")
    for w in res["warnings"]:
        st.caption(f"⚠️ {w}")
    st.markdown("---")
    
    render_contextual_reality_check()
    
    st.markdown("### 📊 Game Log Performance")
    st.caption("Last 10 games (faded bars = last season). Dashed blue = line, dotted orange = model projection.")
    gl = pd.DataFrame(res["gamelog_details"])
    if gl.empty:
        return
    colors = np.where(gl["stat_value"] > line_val, "#22c55e", np.where(gl["stat_value"] < line_val, "#ef4444", "#94a3b8"))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gl["x_label"], y=gl["stat_value"], marker_color=colors, marker_opacity=np.where(gl["season"] == "current", 1.0, 0.5),
        text=[f"{v:.1f}" if v % 1 else f"{int(v)}" for v in gl["stat_value"]],
        textposition="outside", textfont=dict(color="#0f172a", size=12, family="sans-serif"),
    ))
    fig.add_shape(type="line", x0=-0.5, x1=len(gl) - 0.5, y0=line_val, y1=line_val, line=dict(color="#2563eb", width=2, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, x1=len(gl) - 0.5, y0=res["mu"], y1=res["mu"], line=dict(color="#f97316", width=2, dash="dot"))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(color="#0f172a"), height=380,
        margin=dict(l=20, r=20, t=30, b=40), showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(color="#0f172a")),
        yaxis=dict(showgrid=True, gridcolor="#e2e8f0", tickfont=dict(color="#0f172a")),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_matchup_backtest() -> None:
    """Calibrate opponent weights."""
    ss = st.session_state
    st.markdown(
        "Estimates **k** in `stat ÷ player's season average = 1 + k × (opponent index − 1)`, with the opponent index built "
        "only from games played *before* each matchup (no look-ahead), clustered by player. k = 0.35 means a defense "
        "allowing 10% more than average lifts a projection 3.5%. Results are shrunk toward the default before use."
    )
    c1, c2 = st.columns(2)
    n_seasons = c1.slider("Seasons to backtest", 1, 4, 3)
    c2.write("")
    if c2.button("🔬 Run matchup backtest", key="run_mbt"):
        cur = default_season()
        with st.spinner("Loading nflverse seasons and running the backtest…"):
            try:
                first = cur - n_seasons + 1
                weekly = load_weekly(cur, back=n_seasons)
                ss["mbt_table"] = run_backtests(weekly, first)
            except Exception as e:
                ss["mbt_table"] = None
                st.error(f"Backtest failed: {e}")
    table = ss.get("mbt_table")
    if table is not None and not table.empty:
        st.dataframe(table.round(3), use_container_width=True, hide_index=True)
        st.caption("K_HAT is the raw estimate, K_EFF is what the model will use. SIGNIFICANT = |K_HAT| ≥ 2 × SE.")
        a, b = st.columns(2)
        if a.button("✅ Apply to model", key="apply_mbt"):
            ss["matchup_k"] = {r.STAT: float(r.K_EFF) for r in table.itertuples()}
            save_k_file(ss["matchup_k"])
            st.toast("Calibrated matchup weights applied")
        if b.button("↩️ Reset to defaults", key="reset_mbt"):
            ss["matchup_k"] = None
            save_k_file(None)
            st.toast("Back to default weights")
    elif table is not None:
        st.warning("Not enough player-games for any stat. Try more seasons.")
    active = ss.get("matchup_k")
    st.markdown("**Active matchup weights**" + (" (backtested)" if active else " (default assumptions)"))
    st.dataframe(pd.DataFrame({"STAT": list(DEFAULT_K), "k": [round((active or DEFAULT_K).get(s, DEFAULT_K[s]), 3) for s in DEFAULT_K]}),
                 use_container_width=True, hide_index=True)
    if active:
        st.caption("To make these permanent, paste into DEFAULT_K in nfl_matchups.py:")
        st.code("DEFAULT_K = " + json.dumps({k: round(v, 3) for k, v in active.items()}, indent=4), language="python")


def main() -> None:
    """Streamlit entry point."""
    st.markdown(
        """
        <div class="vision-header" style="padding: 20px;">
            <div class="vision-title">⚡ NFL Prop Evaluator</div>
            <div class="vision-sub">Quantitative Monte Carlo Player Prop Analytics Platform</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    ss = st.session_state
    def_player, def_line = DEFAULTS
    if "matchup_k" not in ss:
        ss["matchup_k"] = load_k_file()
    for key, default in (("input_player", def_player), ("input_stat", stat_options()[0]), ("input_line", float(def_line)),
                         ("input_over", -110), ("input_under", -110), ("input_opp", AUTO_OPP), ("input_source", SOURCES[0])):
        ss.setdefault(key, default)

    tab_eval, tab_bt = st.tabs(["⚡ Evaluate", "🔬 Matchup Backtest"])
    with tab_eval:
        render_live_panel()
        with st.form("evaluation_form"):
            c1, c2, c3 = st.columns([2, 1, 1])
            player_name = c1.text_input("Player Name", key="input_player")
            stat_category = c2.selectbox("Stat Category", stat_options(), key="input_stat")
            line_val = c3.number_input("Line", step=0.5, key="input_line")
            c4, c5, c6, c7 = st.columns(4)
            over_odds = c4.number_input("Over Odds", key="input_over")
            under_odds = c5.number_input("Under Odds", key="input_under")
            opp_choice = c6.selectbox("Opponent", [AUTO_OPP, "None"] + NFL_TEAMS, key="input_opp")
            proj_source = c7.selectbox("Projection Model", SOURCES, key="input_source")
            submitted = st.form_submit_button("⚡ Evaluate Prop")

        if submitted or ss.pop("pending_eval", False):
            opponent = "AUTO" if opp_choice == AUTO_OPP else None if opp_choice == "None" else opp_choice
            res = evaluate_prop(player_name=player_name, stat_category=stat_category, line=line_val, over_odds=over_odds,
                                under_odds=under_odds, projection_source=proj_source, season=default_season(), opponent=opponent,
                                matchup_k=ss.get("matchup_k"))
            ss["last_res"] = res
            ss["last_ctx"] = {"stat": stat_category, "line": float(line_val)}
        res = ss.get("last_res")
        if res:
            if "error" in res:
                st.error(res["error"])
            else:
                render_results(res, ss["last_ctx"]["stat"], ss["last_ctx"]["line"])

        with st.expander("ℹ️ How the model works"):
            st.markdown(
                "- **Projection Model**: *Model* = shrunk EWMA × matchup; *My mean* / *My median* = raw mean / median of the last 10 games × matchup.\n"
                "- **Projection** = (EWMA of this season's last 8 games blended with last season's average, ~4 pseudo-games of prior weight) × matchup multiplier.\n"
                "- **Matchup** = what the opponent allows per game to your player's position group (QB/RB/WR/TE) for this exact stat, "
                "shrunk toward last season, ranked 1-32 (1 = allows the most). Multiplier = 1 + k × (index − 1), clipped to ±15%.\n"
                "- **Distribution**: Gamma for yardage, Normal for completions/attempts/carries, Negative Binomial (Poisson if under-dispersed) for TDs, receptions, INTs. 50,000 seeded draws.\n"
                "- **Market**: two-way multiplicative devig. Edge = model % − devigged market %. EV uses the model probability at the offered price (pushes refunded).\n"
                "- **Kelly**: ¼ Kelly capped at 5% of bankroll; 0 when EV ≤ 0."
            )
    with tab_bt:
        render_matchup_backtest()


if __name__ == "__main__":
    main()

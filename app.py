"""Prop Evaluator (Vision UI): NFL/NBA/MLB props with live lines, mean/median projections, Kelly."""
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
    run_backtest,
    stat_options,
)

try:  # Live lines need odds_api_client.py next to this file.
    from odds_api_client import (
        DEFAULT_BOOKS,
        MARKETS,
        OddsApiError,
        estimate_cost,
        fetch_prop_board,
        list_events,
        parse_books,
        player_offers,
        summarize_board,
    )

    HAS_ODDS = True
except Exception:
    HAS_ODDS = False

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prop Evaluator | Vision UI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

GREEN, RED, GRAY = "#00E676", "#FF5252", "#9CA3AF"

# ---------------------------------------------------------------------------
# Vision UI dark glassmorphism CSS
# ---------------------------------------------------------------------------
VISION_UI_CSS = """
<style>
.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(0, 117, 255, 0.12) 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, rgba(111, 66, 193, 0.12) 0%, transparent 40%),
                #060B26 !important;
    color: #FFFFFF !important;
}

/* Sidebar is not used: all controls sit at the top of the page */
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Glass cards: custom blocks, bordered containers, and expanders */
.vision-card,
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"] {
    background: rgba(15, 21, 53, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
}
.vision-card { padding: 20px !important; margin-bottom: 20px !important; }
/* Avoid double borders on containers nested inside other glass cards */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"] [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stExpander"] [data-testid="stExpander"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stMetric"] {
    background: rgba(15, 21, 53, 0.65) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
}
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] {
    color: #A0AEC0 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* Electric blue buttons (regular and form-submit) */
.stButton > button,
[data-testid="stFormSubmitButton"] > button {
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
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(0, 117, 255, 0.6) !important;
}

div[data-baseweb="input"], div[data-baseweb="select"] {
    background: rgba(15, 21, 53, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    color: #FFFFFF !important;
}
div[data-baseweb="input"] input { color: #FFFFFF !important; }

/* Mobile: stack columns */
@media (max-width: 768px) {
    [data-testid="column"], [data-testid="stColumn"] {
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
# Cached engine calls
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def cached_evaluate(**kwargs: Any) -> Dict[str, Any]:
    """Cache the engine call for 5 minutes so reruns do not refetch data."""
    return evaluate_prop_multisport(**kwargs)


@st.cache_data(ttl=300, show_spinner=False)
def cached_backtest(**kwargs: Any) -> Dict[str, Any]:
    """Cache the walk-forward backtest."""
    return run_backtest(**kwargs)


# ---------------------------------------------------------------------------
# Live lines (The Odds API)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Loading games...")
def cached_events(sport: str, api_key: str) -> Tuple[pd.DataFrame, Any]:
    """Cached wrapper for the free games endpoint."""
    return list_events(sport, api_key)


@st.cache_data(ttl=120, show_spinner="Fetching lines...")
def cached_board(
    sport: str, event_id: str, market: str, books: Tuple[str, ...], api_key: str
) -> Tuple[pd.DataFrame, Any]:
    """Cached wrapper (2 min) so repeat clicks do not spend extra credits."""
    return fetch_prop_board(sport, event_id, market, books, api_key)


def load_offer_into_model(
    sport: str, stat_label: str, player: str, line: float, over: float, under: float
) -> None:
    """Button callback: copy a chosen book offer into the controls and run the evaluation."""
    ss = st.session_state
    ss[f"player_{sport}"] = player
    ss[f"stat_{sport}"] = stat_label
    ss[f"line_{sport}"] = float(line)
    ss[f"over_{sport}"] = int(round(over))
    ss[f"under_{sport}"] = int(round(under))
    ss["ran_sport"] = sport


def render_live_panel(sport: str) -> None:
    """Expander: load games (free), pick a game and prop type, fetch the board."""
    ss = st.session_state
    with st.expander("📡 Live Lines (The Odds API)"):
        api_key = st.text_input(
            "API key", type="password", value=os.environ.get("ODDS_API_KEY", ""), key="odds_key"
        ).strip()
        st.caption(
            "Keep your key private: never paste it into chat or screenshots. "
            "Locally you can also run `export ODDS_API_KEY=...` before launching."
        )
        books_text = st.text_input(
            "Bookmaker keys (comma-separated)", value=DEFAULT_BOOKS, key="odds_books"
        )
        live_stat = st.selectbox("Prop type", list(MARKETS[sport].keys()), key=f"live_stat_{sport}")

        if st.button("Load games (free)", key=f"load_games_{sport}"):
            try:
                events, quota = cached_events(sport, api_key)
                ss[f"events_{sport}"] = events
                ss["quota"] = quota
            except OddsApiError as exc:
                st.error(str(exc))

        events = ss.get(f"events_{sport}")
        if events is not None and events.empty:
            st.info("No upcoming games are listed right now.")
        elif events is not None:
            game_idx = st.selectbox(
                "Game",
                range(len(events)),
                format_func=lambda i: str(events.loc[i, "label"]),
                key=f"live_game_{sport}_{len(events)}",
            )
            books = parse_books(books_text)
            cost = estimate_cost(1, books)
            if st.button(f"Fetch {live_stat} lines (~{cost} credit)", key=f"fetch_board_{sport}"):
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

        quota = ss.get("quota")
        if quota is not None and quota.remaining is not None:
            st.caption(f"Credits remaining: {quota.remaining} (last call cost {quota.last})")

        render_live_board(sport)


def render_live_board(sport: str) -> None:
    """Consensus lines, best prices, and a one-click load into the controls."""
    state = st.session_state.get("board")
    if not state or state["sport"] != sport:
        return
    board: pd.DataFrame = state["data"]
    st.markdown(f"**Live board: {state['stat']} | {state['game']}**")
    if board.empty:
        st.info(
            "No lines returned. Props usually post within a day of the game, and not every "
            "book lists every market. Try another game, prop type, or bookmaker list."
        )
        return
    summary = summarize_board(board)
    st.dataframe(
        summary.style.format(
            {"Line": "{:g}", "Best Over": "{:+.0f}", "Best Under": "{:+.0f}", "No-vig Over %": "{:.1f}"},
            na_rep="-",
        ),
        use_container_width=True, hide_index=True,
    )
    player = st.selectbox(
        "Player", summary["Player"].tolist(),
        key=f"live_player_{sport}_{state['game']}_{state['stat']}",
    )
    offers = player_offers(board, player)
    if offers.empty:
        st.info("No single book posted both Over and Under for this player.")
        return
    labels = [
        f"{b} | line {ln:g} | O {o:+.0f} / U {u:+.0f}"
        for b, ln, o, u in zip(offers["Book"], offers["Line"], offers["Over"], offers["Under"])
    ]
    pick = st.selectbox(
        "Price from", range(len(offers)), format_func=lambda i: labels[i],
        key=f"live_offer_{sport}_{player}",
    )
    row = offers.iloc[pick]
    st.button(
        "Load into evaluator and run", key=f"load_model_{sport}",
        on_click=load_offer_into_model,
        args=(sport, state["stat"], player, row["Line"], row["Over"], row["Under"]),
    )
    st.dataframe(
        offers.style.format(
            {"Line": "{:g}", "Over": "{:+.0f}", "Under": "{:+.0f}",
             "No-vig Over %": "{:.1f}", "Hold %": "{:.1f}"}
        ),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "The evaluator prices one book at a time so the devig uses a real two-sided market. "
        "'Best Over/Under' in the board is line shopping across books at the consensus line."
    )


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
def render_controls(sport: str) -> Tuple[Dict[str, Any], bool]:
    """Top control panel. Returns (engine kwargs, whether Run was clicked)."""
    ss = st.session_state
    default_player, default_line = DEFAULTS[sport]
    ss.setdefault(f"player_{sport}", default_player)
    ss.setdefault(f"season_{sport}", int(default_season(sport)))
    ss.setdefault(f"line_{sport}", float(default_line))
    ss.setdefault(f"over_{sport}", -110)
    ss.setdefault(f"under_{sport}", -110)
    ss.setdefault(f"custom_{sport}", float(default_line))

    with st.container(border=True):
        with st.form(f"controls_{sport}"):
            col_player, col_season, col_stat, col_line = st.columns([2, 1, 1.5, 1])
            with col_player:
                player_name = st.text_input("Player", key=f"player_{sport}")
            with col_season:
                season = st.number_input(
                    "Season" if sport != "NBA" else "Season start year (2026 = 2026-27)",
                    1999, 2100, step=1, key=f"season_{sport}",
                )
            with col_stat:
                stat_category = st.selectbox(
                    "Stat Category", stat_options(sport), key=f"stat_{sport}"
                )
            with col_line:
                line = st.number_input("Line", step=0.5, key=f"line_{sport}")

            col_over, col_under, col_source, col_custom = st.columns([1, 1, 1.5, 1.5])
            with col_over:
                over_odds = st.number_input("Over Odds (American)", step=5, key=f"over_{sport}")
            with col_under:
                under_odds = st.number_input("Under Odds (American)", step=5, key=f"under_{sport}")
            with col_source:
                projection_source = st.selectbox("Projection Source", list(SOURCES))
            with col_custom:
                custom_val = st.number_input(
                    "Your Mean / Median (only for 'My mean' / 'My median')",
                    min_value=0.0, step=0.1, key=f"custom_{sport}",
                )

            with st.expander("⚙️ Advanced Model Settings", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    ewma_decay = st.slider("EWMA Decay (Lower = More Recent)", 0.1, 1.0, 0.80)
                    devig_method = st.selectbox("Devig Method", list(DEVIG_METHODS))
                    n_sims = st.select_slider("Simulations", [10_000, 25_000, 50_000, 100_000], value=50_000)
                with col_b:
                    max_recent_games = st.slider("Max Recent Games", 1, 30, 10)
                    model_weight = st.slider("Model Weight (1.0 = trust model fully)", 0.0, 1.0, 0.5, 0.05)
                    seed = st.number_input("Seed", value=42, step=1)
                with col_c:
                    matchup_multiplier = st.slider("Matchup Multiplier (Model source only)", 0.5, 1.5, 1.0)
                    bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=50.0)
                    kelly_fraction = st.slider("Kelly Fraction", 0.05, 1.0, 0.25, 0.05)
                    kelly_cap_pct = st.slider("Max Allocation (% of bankroll)", 0.5, 10.0, 3.0, 0.5)
                    edge_warn_pts = st.slider("Warn if model edge exceeds (pts)", 3.0, 20.0, 8.0, 0.5)

            submitted = st.form_submit_button("⚡ Run Evaluation")

    params: Dict[str, Any] = {
        "league": sport,
        "player_name": player_name.strip(),
        "stat_category": stat_category,
        "line": float(line),
        "over_odds": float(over_odds),
        "under_odds": float(under_odds),
        "projection_source": projection_source,
        "custom_val": float(custom_val),
        "ewma_decay": float(ewma_decay),
        "max_recent_games": int(max_recent_games),
        "matchup_multiplier": float(matchup_multiplier),
        "season": int(season),
        "bankroll": float(bankroll),
        "kelly_fraction": float(kelly_fraction),
        "kelly_cap_pct": float(kelly_cap_pct),
        "devig_method": devig_method,
        "model_weight": float(model_weight),
        "n_sims": int(n_sims),
        "seed": int(seed),
        "edge_warn_pts": float(edge_warn_pts),
    }
    return params, bool(submitted)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _style_chart(fig: go.Figure, x_title: str, y_title: str) -> go.Figure:
    """Apply the glass chart theme."""
    fig.update_layout(
        paper_bgcolor="rgba(15, 21, 53, 0.65)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#A0AEC0"),
        xaxis=dict(title=x_title, gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(title=y_title, gridcolor="rgba(255, 255, 255, 0.05)"),
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    return fig


def _side_colors(x: np.ndarray, line: float) -> list:
    """Green above the line, red below, gray exactly on it (a push)."""
    return np.where(x > line, GREEN, np.where(x < line, RED, GRAY)).tolist()


def distribution_chart(sims: np.ndarray, line: float, family: str, label: str, median: float) -> go.Figure:
    """Simulated outcome distribution with the book line and median marked."""
    if family == "count":
        counts = np.bincount(sims.astype(int))
        x = np.arange(len(counts), dtype=float)
        heights = counts / len(sims)
    else:
        counts, bins = np.histogram(sims, bins=35)
        x = 0.5 * (bins[:-1] + bins[1:])
        heights = counts / len(sims)
    fig = go.Figure(data=[go.Bar(x=x, y=heights, marker_color=_side_colors(x, line), opacity=0.85)])
    fig.add_vline(x=line, line_dash="dash", line_color="#FFFFFF", annotation_text=f"Line {line:g}")
    fig.add_vline(
        x=median, line_dash="dot", line_color="#F59E0B",
        annotation_text="Median", annotation_position="bottom right",
    )
    return _style_chart(fig, label, "Probability")


def recent_chart(values: np.ndarray, labels: list, line: float, label: str) -> go.Figure:
    """Recent game results versus the book line."""
    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color=_side_colors(values, line), opacity=0.85)])
    fig.add_hline(y=line, line_dash="dash", line_color="#FFFFFF")
    return _style_chart(fig, "Game", label)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
def render_results(result: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Render metrics, Kelly card, probability table, charts, and backtest."""
    stat_category = params["stat_category"]
    line = result["line"]

    for message in result.get("warnings", []):
        st.warning(message)

    st.subheader(f"{result['player_name']} | {stat_category} | {params['league']}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Projection (Mean)", f"{result['projection_mean']:.2f}")
    m2.metric("Projection (Median)", f"{result['projection_median']:.2f}")
    m3.metric("Best Side", result["best_side"])
    m4.metric("Fair Odds (Blended)", f"{result['fair_odds']:+d}")
    m5.metric("Edge vs Market", f"{result['edge_vs_market']:+.1f}%")
    m6.metric("Expected Value (EV)", f"{result['ev_pct']:+.1f}%")

    st.markdown(
        f"""
        <div class="vision-card" style="margin-top: 15px;">
            <span style="color:#A0AEC0; font-size:0.85rem; text-transform:uppercase;">Recommended Kelly Stake</span>
            <h2 style="color:#0075FF; margin:0; font-weight:700;">${result['kelly_stake_dollars']:.2f}
            <span style="font-size:1rem; color:#A0AEC0;">({result['kelly_pct']:.2f}% of bankroll)</span></h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    key = result["best_side_key"]
    st.caption(
        f"{key.title()}: model {result[f'{key}_model_pct']:.1f}% | market "
        f"({result['devig_method']} devig) {result[f'{key}_market_pct']:.1f}% | blended "
        f"{result[f'{key}_blend_pct']:.1f}% at model weight {result['model_weight']:.2f}. "
        f"Recent games ({result['n_games']}): weighted median {result['recent_weighted_median']:.2f} | "
        f"simple median {result['recent_simple_median']:.2f} | simple mean {result['recent_simple_mean']:.2f}"
    )

    st.markdown("### Market & Model Probability Breakdown")
    rows = []
    for side, book_odds in (("over", params["over_odds"]), ("under", params["under_odds"])):
        rows.append(
            {
                "Side": side.title(),
                "Win %": f"{result[f'{side}_win_pct']:.1f}%",
                "Push %": f"{result[f'{side}_push_pct']:.1f}%",
                "Model %": f"{result[f'{side}_model_pct']:.1f}%",
                "Market %": f"{result[f'{side}_market_pct']:.1f}%",
                "Blend %": f"{result[f'{side}_blend_pct']:.1f}%",
                "Fair Odds": f"{result[f'{side}_fair_odds']:+d}",
                "Book Odds": f"{book_odds:+.0f}",
                "Edge %": f"{result[f'{side}_edge']:+.1f}%",
                "EV %": f"{result[f'{side}_ev']:+.1f}%",
                "Kelly %": f"{result[f'{side}_kelly_pct']:.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("#### Simulated Distribution")
        st.plotly_chart(
            distribution_chart(
                result["simulations"], line, result["family"], stat_category, result["projection_median"]
            ),
            use_container_width=True,
        )
    with chart_col2:
        st.markdown(f"#### Recent {stat_category}")
        st.plotly_chart(
            recent_chart(
                np.asarray(result["recent_games"], dtype=float), result["game_labels"], line, stat_category
            ),
            use_container_width=True,
        )

    render_backtest(params)

    with st.expander("Model details"):
        st.write(
            f"Games used: {result['n_games']} | Mean: {result['projection_mean']:.2f} | "
            f"Sim median: {result['projection_median']:.2f} | Variance: {result['variance']:.2f} | "
            f"Devigged market Over prob ({result['devig_method']}): {result['market_over_prob']:.1%}"
        )
        st.dataframe(
            pd.DataFrame({"Game": result["game_labels"], stat_category: result["recent_games"]}),
            use_container_width=True, hide_index=True,
        )


def render_backtest(params: Dict[str, Any]) -> None:
    """Walk-forward backtest expander for the current player and stat."""
    with st.expander("Backtest (walk-forward on this player's logs)"):
        st.caption(
            "Each game is predicted using only earlier games, against a proxy line (floor of the "
            "median of the previous 8 games + 0.5), not real market lines. This checks whether the "
            "probabilities are calibrated, not whether the bets would have made money. One player's "
            "logs are a small sample, so treat the results as a rough sanity check."
        )
        n_games = st.slider("Games to test over (most recent)", 10, 60, 30)
        if not st.checkbox("Run backtest"):
            return
        bt = cached_backtest(
            league=params["league"],
            player_name=params["player_name"],
            stat_category=params["stat_category"],
            season=params["season"],
            ewma_decay=params["ewma_decay"],
            max_recent_games=params["max_recent_games"],
            n_games=int(n_games),
        )
        if "error" in bt:
            st.info(bt["error"])
            return
        s = bt["summary"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Games tested", f"{int(s['n'])}")
        c2.metric("Brier (0.250 = coin flip)", f"{s['brier']:.3f}")
        c3.metric("Log loss (0.693 = coin flip)", f"{s['logloss']:.3f}")
        c4.metric("Side hit rate", "n/a" if np.isnan(s["hit"]) else f"{s['hit']:.1%}")
        c5.metric("Avg predicted vs actual Over", f"{s['avg_pred']:.0%} / {s['actual']:.0%}")
        st.dataframe(bt["calibration"], use_container_width=True, hide_index=True)
        st.dataframe(
            bt["table"].style.format(
                {"Proxy Line": "{:.1f}", "P(Over)": "{:.1%}", "Actual": "{:.1f}", "Over": "{:.0f}"}
            ),
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the Vision UI prop evaluator."""
    st.title("⚡ Prop Evaluator")
    st.caption("Monte Carlo prop analytics for NFL, NBA and MLB")

    sport = st.radio("Sport", list(SPORTS), horizontal=True, key="sport")

    if HAS_ODDS:
        render_live_panel(sport)

    params, submitted = render_controls(sport)
    if submitted:
        st.session_state["ran_sport"] = sport

    if st.session_state.get("ran_sport") != sport:
        st.info(f"Set up a {sport} prop above and click **Run Evaluation**.")
        st.stop()

    if not params["player_name"]:
        st.error("Enter a player name.")
        st.stop()

    with st.spinner("Evaluating prop market..."):
        result = cached_evaluate(**params)

    if "error" in result:
        st.error(result["error"])
        st.stop()

    render_results(result, params)


if __name__ == "__main__":
    main()

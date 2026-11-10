"""NFL Prop Evaluator: real opponent-vs-position matchups, devigged edge, calibrated matchup weights."""
import json
import uuid
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from multi_sport_engine import (
    BLOWOUT_SPREAD,
    DEFAULTS,
    RECENT_GAMES,
    SOURCES,
    GameInfo,
    add_six_point_column,
    assess,
    default_season,
    evaluate_prop,
    fetch_forecast,
    fetch_game_info,
    game_matches,
    label_gameday,
    opponent_from_label,
    stat_options,
    style_tags,
)
from nfl_matchups import DEFAULT_K, run_backtests
from nfl_headshots import fetch_roster, player_photo
from odds_api_client import (
    MARKETS,
    OddsApiError,
    fetch_prop_board,
    list_events,
    player_offers,
    summarize_board,
)
from sports_data import NFL_TEAMS, load_weekly
import live_game

st.set_page_config(
    page_title="NFL Prop Evaluator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {
        display: none !important;
    }

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

    [data-testid="stExpander"] summary, 
    [data-testid="stExpander"] summary * {
        color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 16px !important;
    }

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

    label, p, span, div[data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

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
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return os.environ.get(name, "").strip()


def _to_american(x: float) -> int:
    x = float(x)
    if abs(x) >= 100:
        return int(round(x))
    if x > 1.0:
        return int(round((x - 1.0) * 100.0)) if x >= 2.0 else int(round(-100.0 / (x - 1.0)))
    return -110


def load_k_file() -> Optional[Dict[str, float]]:
    try:
        if os.path.exists(K_PATH):
            with open(K_PATH) as fh:
                return json.load(fh)
    except Exception:
        pass
    return None


def save_k_file(k: Optional[Dict[str, float]]) -> None:
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
    ss = st.session_state
    ss["input_player"], ss["input_stat"], ss["input_line"] = player, stat_category, float(line)
    ss["input_over"], ss["input_under"] = _to_american(over), _to_american(under)
    ss["input_opp"] = AUTO_OPP
    ss["pending_eval"] = True


def _adj_cfg() -> Dict[str, Any]:
    ss = st.session_state
    split = lambda key: [x.strip() for x in str(ss[key]).split(",") if x.strip()]
    return {"injuries": ss["adj_inj"], "absorb_scale": float(ss["adj_absorb"]),
            "manual_out": split("adj_out"), "manual_q": split("adj_q"), "manual_in": split("adj_in")}


def scan_board(summary: pd.DataFrame, stat_category: str, game_label: str, progress: Any = None) -> Dict[str, Any]:
    ss = st.session_state
    out: Dict[str, Any] = {}
    try:
        weekly: Optional[pd.DataFrame] = load_weekly(default_season(), 1)
    except Exception:
        weekly = None
    gameday = label_gameday(game_label)
    n = len(summary)

    def run(name: str, r: pd.Series, opponent: str) -> Dict[str, Any]:
        return evaluate_prop(
            player_name=name,
            stat_category=stat_category,
            line=float(r["Line"]),
            over_odds=_to_american(r["Over"]),
            under_odds=_to_american(r["Under"]),
            projection_source=ss.get("input_source", SOURCES[0]),
            season=default_season(),
            opponent=opponent,
            matchup_k=ss.get("matchup_k"),
            adj_cfg=_adj_cfg(),
        )

    for k, (_, r) in enumerate(summary.iterrows()):
        name = str(r["Player"])
        if r[["Line", "Over", "Under"]].isna().any():
            out[name] = "incomplete odds"
            continue
        if progress is not None:
            progress.progress(k / max(n, 1), text=f"Scanning {name} ({k + 1}/{n})")
        try:
            res = run(name, r, "AUTO")
            if "error" in res:
                out[name] = str(res["error"])[:70]
                continue
            if not game_matches(game_label, res.get("team"), (res.get("matchup") or {}).get("opponent")):
                opp = opponent_from_label(game_label, res.get("team"))
                if opp is None:
                    out[name] = f"{res.get('team')} isn't in this game"
                    continue
                res = run(name, r, opp)
                if "error" in res:
                    out[name] = str(res["error"])[:70]
                    continue
            if res.get("matchup") and gameday:
                res["matchup"] = {**res["matchup"], "gameday": gameday}
            out[name] = assess(res, stat_category, float(r["Line"]), weekly, default_season(), cached_game_info, cached_forecast)
        except Exception as e:
            out[name] = f"error: {str(e)[:60]}"
    if progress is not None:
        progress.progress(1.0, text="Scan complete")
    return out


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
            game_idx = st.selectbox("Step 2: Select Game", range(len(events)), format_func=lambda i: str(events.loc[i, "label"]), key="live_game")
            if st.button(f"Step 3: Fetch {live_stat} Odds", key="fetch_board"):
                try:
                    board, _ = fetch_prop_board("NFL", str(events.loc[game_idx, "id"]), live_stat, ("draftkings",), api_key)
                    ss["board"], ss["board_scan"], ss["board_token"] = board, None, uuid.uuid4().hex
                    ss["board_game_label"] = str(events.loc[game_idx, "label"])
                except OddsApiError as exc:
                    st.error(str(exc))
        board = ss.get("board")
        if board is not None and not board.empty:
            summary = summarize_board(board)
            shown = summary
            scan = ss.get("board_scan")
            if scan is not None and scan["token"] == ss.get("board_token"):
                shown = add_six_point_column(summary, scan["six"])
            st.dataframe(style_tags(shown) if "Six-Point" in shown else shown, use_container_width=True)
            st.caption("🔥 SHARP = 6/6 criteria · ⚡ LEAN = 5/6 · other rows show their score. Each prop scored dynamically for its best EV side.")
            if st.button("🔎 Scan board for 5/6 & 6/6 props", key="scan_board_btn"):
                bar = st.progress(0.0, text="Starting scan...")
                ss["board_scan"] = {"token": ss.get("board_token"), "six": scan_board(summary, live_stat, ss.get("board_game_label", ""), bar)}
                st.rerun()
            player_pick = st.selectbox("Step 4: Choose Player to Evaluate", shown["Player"].tolist(), key="pick_player")
            offers = player_offers(board, player_pick)
            if not offers.empty:
                row = offers.iloc[0]
                if st.button(f"⚡ Populate {player_pick} ({row['Line']} | Over {row['Over']} / Under {row['Under']})", key="populate_btn"):
                    sync_populate_to_widgets(live_stat, player_pick, row["Line"], row["Over"], row["Under"])
                    st.rerun()


@st.cache_data(ttl=300, show_spinner=False)
def cached_game_info(team: str, opponent: Optional[str], gameday: Optional[str]) -> Optional[GameInfo]:
    return fetch_game_info(team, opponent, gameday)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_forecast(lat: float, lon: float) -> Dict[str, Any]:
    return fetch_forecast(lat, lon)


@st.cache_data(ttl=21600, show_spinner=False)
def cached_roster(team: str) -> Dict[str, Any]:
    return fetch_roster(team)


def _player_photo(res: Dict[str, Any]) -> Optional[str]:
    if not res.get("team"):
        return live_game.headshot_url(res["player_name"])
    try:
        return player_photo(res["player_name"], res["team"], res.get("pos"), cached_roster)
    except Exception:
        return None


def _metric(col: Any, label: str, value: str, tone: str = "neutral") -> None:
    color = {"neutral": "#0369a1", "good": "#15803d", "bad": "#b91c1c"}[tone]
    col.markdown(
        f'<div class="metric-box"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color} !important;">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _tone(x: float) -> str:
    return "good" if x > 0 else "bad"


def render_final_assessment(res: Dict[str, Any]) -> None:
    six = res.get("six_point")
    if not six:
        err = res.get("six_point_error")
        if err:
            st.caption(f"⚠️ Six-point assessment unavailable ({err}).")
        return
    best = six[six["best"]]
    other = six["Under" if six["best"] == "Over" else "Over"]
    tone = "good" if "SHARP" in best["tier"] else "neutral" if "LEAN" in best["tier"] else "bad"
    st.markdown("### 🏁 Final Assessment: Dual-Directional 6-Point Check")
    a, b, c = st.columns([1, 1, 2])
    _metric(a, f"TIER · {best['side'].upper()}", best["tier"], tone)
    _metric(b, "CRITERIA MET", f"{best['score']}/6", tone)
    c.caption(f"6/6 = SHARP · 5/6 = LEAN · 4 or fewer = PASS/FADE. Scored dynamically for the model's best side ({best['side']}). "
              f"The {other['side']} side scores {other['score']}/6 ({other['tier']}). A check that can't be verified counts as unpassed.")
    if "PASS" not in best["tier"] and res["ev_pct"] <= 0:
        st.warning("⚠️ The six-point score is positive but the model's EV at this price is not. Check the price before betting.")
    for ck in best["checks"]:
        icon = "✅" if ck["passed"] else ("⚪" if not ck["verified"] else "❌")
        st.markdown(f"{icon} **{ck['label']}**: {ck['detail']}")
    st.caption(f"Defensive ranks use what each defense allows this position group (season-long, and its last {RECENT_GAMES} games). "
               f"Spread comes from ESPN scoreboard (blowout = {BLOWOUT_SPREAD:g}+); weather is Open-Meteo forecast at kickoff.")


def render_matchup(res: Dict[str, Any], stat_category: str) -> None:
    m = res.get("matchup")
    st.markdown("### 🛡️ Opponent Matchup (real nflverse data)")
    if not m:
        st.info("No matchup adjustment applied. Pick an opponent (or use Auto) to see rank vs position.")
        return
    venue = "" if m["home"] is None else ("vs " if m["home"] else "@ ")
    wk = f" · Wk {m['week']}" if m.get("week") else ""
    c = st.columns(5)
    _metric(c[0], "OPPONENT", f"{venue}{m['opponent']}{wk}")
    _metric(c[1], f"RANK VS {m['pos']} (1 = SOFTEST)", f"#{m['rank']} of {m['n_teams']}", "good" if m["rank"] <= 10 else "bad" if m["rank"] >= m["n_teams"] - 9 else "neutral")
    _metric(c[2], f"{stat_category.upper()} ALLOWED / GM", f"{m['allowed']:.1f}")
    _metric(c[3], "LEAGUE AVG / INDEX", f"{m['league']:.1f} · {m['index']:.2f}x")
    _metric(c[4], "MATCHUP ADJ", f"x{m['mult']:.3f}", "good" if m["mult"] > 1.0005 else "bad" if m["mult"] < 0.9995 else "neutral")
    src = "backtested" if m["calibrated"] else "default assumption"
    notes = [f"Allowed to {m['pos']}s per game, shrunk toward last season ({m['games']} games). Weight k = {m['k']:.2f} ({src})."]
    if m.get("rest") is not None and m.get("opp_rest") is not None:
        flag = lambda d: " (short week)" if d <= 6 else " (post-bye)" if d >= 13 else ""
        notes.append(f"Rest days: you {m['rest']}{flag(m['rest'])}, opponent {m['opp_rest']}{flag(m['opp_rest'])}.")
    for n in notes:
        st.caption(n)
    with st.expander(f"All 32 defenses vs {m['pos']} ({stat_category})"):
        board = m["board"].rename(columns={"EST": "ALLOWED (SHRUNK)", "CUR": "THIS SEASON"}).round(2)
        st.dataframe(board, use_container_width=True, hide_index=True)


def _scoreboard_body() -> None:
    rows = live_game.scoreboard()
    if not rows:
        st.caption("Live scoreboard unavailable right now.")
        return
    focus = live_game.canon((st.session_state.get("last_res") or {}).get("team", ""))
    for i in range(0, len(rows), 4):
        cols = st.columns(4)
        for col, r in zip(cols, rows[i:i + 4]):
            star = "⭐ " if focus in (r["away"], r["home"]) else ""
            live = r["state"] == "in"
            score = f"{r['away']} {r['away_score']} – {r['home_score']} {r['home']}" if r["state"] != "pre" else f"{r['away']} @ {r['home']}"
            _metric(col, f"{star}{'🔴 LIVE · ' if live else ''}{r['detail']}", score, "good" if live else "neutral")
    st.caption("Auto-refreshes every 20 seconds while any game is live.")


def render_live_scoreboard() -> None:
    any_live = any(r["state"] == "in" for r in live_game.scoreboard())
    with st.expander("📺 Live NFL Scoreboard", expanded=True):
        if hasattr(st, "fragment"):
            st.fragment(run_every=20 if any_live else None)(_scoreboard_body)()
        else:
            _scoreboard_body()


def render_live_script(res: Dict[str, Any], stat_category: str) -> None:
    ctx = st.session_state.get("last_ctx", {})
    st.markdown("### 🏈 Live Game Script")
    spec = live_game.spec_for(stat_category)
    if spec is None:
        st.caption("No live game-script model for this stat.")
        return
    game = live_game.find_game(res["team"])
    if game is None:
        st.caption(f"No {res['team']} game found on current ESPN scoreboard.")
        return
    if game.state == "pre":
        st.caption(f"Pregame ({game.detail}). The live adjustment switches on at kickoff.")
        return
    if st.button("🔄 Refresh live score", key="live_refresh"):
        live_game.refresh()
        st.rerun()
    st.markdown(
        f'<div style="text-align:center;font-size:2.4rem;font-weight:800;color:#0f172a;">'
        f'{game.team} {game.team_score} – {game.opp_score} {game.opp}</div>'
        f'<div style="text-align:center;font-weight:700;color:#0369a1;margin-bottom:12px;">🔴 {game.detail}</div>',
        unsafe_allow_html=True,
    )
    spread = st.number_input(f"Pregame spread for {game.team} (0 = ignore)", value=0.0, step=0.5, key="live_spread")
    auto = live_game.fetch_player_stat(game.event_id, res["player_name"], spec)
    current = st.number_input(f"{stat_category} so far", value=float(auto or 0.0), step=1.0, key=f"live_cur_{res['player_name']}_{stat_category}_{auto}")
    out = live_game.evaluate_live(res["mu"], spec, game, float(spread), float(current), float(ctx["line"]), float(ctx["over"]), float(ctx["under"]))
    word = "Trailing" if out["lean"] < -0.05 else "Leading" if out["lean"] > 0.05 else "Even"
    flow = "pass-heavy" if out["lean"] < -0.05 else "run-heavy" if out["lean"] > 0.05 else "neutral"
    c = st.columns(6)
    _metric(c[0], f"{game.team} vs {game.opp}", f"{game.team_score}-{game.opp_score} · {game.detail}")
    _metric(c[1], "GAME SCRIPT", f"{word} · {flow}", "good" if out["lean"] > 0.05 else "bad" if out["lean"] < -0.05 else "neutral")
    _metric(c[2], "SO FAR", f"{current:.1f}")
    _metric(c[3], "LIVE FINAL PROJ", f"{out['final_proj']:.1f}", _tone(out["final_proj"] - ctx["line"]))
    _metric(c[4], "LIVE VERDICT", f"{out['side']} {ctx['line']}", _tone(out["ev_pct"]))
    _metric(c[5], "LIVE EV / EDGE", f"{out['ev_pct']:+.1f}% / {out['edge_pp']:+.1f}pp", _tone(out["ev_pct"]))


def render_contextual_reality_check() -> None:
    with st.expander("🔍 Step 3: Contextual Reality Check (Pre-Flight Filter)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox("1. Secure Volume & Usage", value=True)
        with col2:
            st.checkbox("2. Clean Schedule Spot", value=True)
        with col3:
            st.checkbox("3. Sharp Line Movement", value=True)


def render_injury_usage(res: Dict[str, Any]) -> None:
    inj = res.get("inj")
    if not inj:
        return
    st.markdown("### 🩺 Injuries & Usage")
    grp = inj.get("group")
    share_lbl = {"rec": "TARGET SHARE", "rush": "CARRY SHARE"}.get(grp)
    c = st.columns(4)
    _metric(c[0], share_lbl or "ROLE", f"{inj['share']:.1f}%" if share_lbl and inj.get("share") is not None else "Passing game")
    _metric(c[1], "USAGE ADJ", f"x{inj['mult']:.3f}", "good" if inj["mult"] > 1.0005 else "bad" if inj["mult"] < 0.9995 else "neutral")
    um = float(res.get("usage_mult", inj["mult"])) or 1.0
    _metric(c[2], "PROJ BEFORE USAGE", f"{res['mu'] / um:.2f}")
    _metric(c[3], "PROJ AFTER USAGE", f"{res['mu']:.2f}", "good" if um > 1.0005 else "bad" if um < 0.9995 else "neutral")
    if inj["table"]:
        st.dataframe(pd.DataFrame(inj["table"]), use_container_width=True, hide_index=True)


def render_tiers(res: Dict[str, Any], stat_category: str) -> None:
    t = res.get("tiers")
    if not t:
        return
    st.markdown("### 🎯 Player vs Defense Tiers")
    fmt = lambda x: "n/a" if x["avg"] is None else f"{x['avg']:.1f} ({x['n']} g)"
    c = st.columns(5)
    for col, lab, key in zip(c, ["ALL GAMES", "VS 5 SOFTEST", "VS 10 SOFTEST", "VS 10 TOUGHEST", "VS 5 TOUGHEST"], ["overall", "soft5", "soft10", "tough10", "tough5"]):
        _metric(col, lab, fmt(t[key]))


def render_results(res: Dict[str, Any], stat_category: str, line_val: float) -> None:
    photo_col, title_col = st.columns([1, 6])
    photo = _player_photo(res)
    if photo:
        photo_col.image(photo, width=120)
    title_col.markdown(f"## {res['player_name']} | {stat_category}")
    title_col.caption(f"{res['team']} · {res['pos']} · {res['n_current']} games this season, {res['n_prior']} last season")
    c = st.columns(6)
    _metric(c[0], "ADJ PROJ", f"{res['mu']:.2f}")
    _metric(c[1], "EWMA (PRE-MATCHUP)", f"{res['base_proj']:.2f}")
    _metric(c[2], "BEST SIDE (BY EV)", res["best_side"], _tone(res["ev_pct"]))
    fair_odds_disp = f"{int(round(res['fair_odds'])):+d}" if isinstance(res.get("fair_odds"), (int, float)) else str(res.get("fair_odds", ""))
    _metric(c[3], "FAIR ODDS", fair_odds_disp)
    _metric(c[4], "EDGE VS DEVIG", f"{res['edge_pp']:+.1f} pp", _tone(res["edge_pp"]))
    _metric(c[5], "EV", f"{res['ev_pct']:+.1f}%", _tone(res["ev_pct"]))
    st.markdown("---")
    render_final_assessment(res)
    st.markdown("---")
    render_matchup(res, stat_category)
    render_injury_usage(res)
    render_tiers(res, stat_category)
    render_live_script(res, stat_category)
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
    gl = pd.DataFrame(res["gamelog_details"])
    if gl.empty:
        return
    colors = np.where(gl["stat_value"] > line_val, "#22c55e", np.where(gl["stat_value"] < line_val, "#ef4444", "#94a3b8"))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=gl["x_label"], y=gl["stat_value"], marker_color=colors,
        text=[f"{v:.1f}" if v % 1 else f"{int(v)}" for v in gl["stat_value"]],
        textposition="outside", textfont=dict(color="#0f172a", size=12),
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
    ss = st.session_state
    c1, c2 = st.columns(2)
    n_seasons = c1.slider("Seasons to backtest", 1, 4, 3)
    if c2.button("🔬 Run matchup backtest", key="run_mbt"):
        cur = default_season()
        with st.spinner("Loading nflverse seasons and running backtest…"):
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
        a, b = st.columns(2)
        if a.button("✅ Apply to model", key="apply_mbt"):
            ss["matchup_k"] = {r.STAT: float(r.K_EFF) for r in table.itertuples()}
            save_k_file(ss["matchup_k"])
            st.toast("Calibrated matchup weights applied")
        if b.button("↩️ Reset to defaults", key="reset_mbt"):
            ss["matchup_k"] = None
            save_k_file(None)
            st.toast("Back to default weights")


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
    ss = st.session_state
    def_player, def_line = DEFAULTS
    if "matchup_k" not in ss:
        ss["matchup_k"] = load_k_file()
    for key, default in (("input_player", def_player), ("input_stat", stat_options()[0]), ("input_line", float(def_line)),
                         ("input_over", -110), ("input_under", -110), ("input_opp", AUTO_OPP), ("input_source", SOURCES[0]),
                         ("adj_inj", True), ("adj_absorb", 1.0), ("adj_out", ""), ("adj_q", ""), ("adj_in", "")):
        ss.setdefault(key, default)

    tab_eval, tab_bt = st.tabs(["⚡ Evaluate", "🔬 Matchup Backtest"])
    with tab_eval:
        render_live_scoreboard()
        render_live_panel()
        with st.expander("🩺 Injuries & usage settings", expanded=False):
            st.checkbox("Adjust for injured / returning teammates", key="adj_inj")
            st.slider("Absorption strength", 0.5, 1.0, key="adj_absorb", step=0.05)
            st.text_input("Teammates OUT", key="adj_out")
            st.text_input("Teammates QUESTIONABLE", key="adj_q")
            st.text_input("Teammates ACTIVE", key="adj_in")
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
                                matchup_k=ss.get("matchup_k"),
                                adj_cfg=_adj_cfg())
            if "error" not in res:
                try:
                    weekly = load_weekly(default_season(), 1)
                except Exception:
                    weekly = None
                try:
                    game_label = ss.get("board_game_label", "")
                    gameday = label_gameday(game_label)
                    if res.get("matchup") and gameday:
                        res["matchup"] = {**res["matchup"], "gameday": gameday}
                    res["six_point"] = assess(res, stat_category, float(line_val), weekly, default_season(), cached_game_info, cached_forecast)
                except Exception as e:
                    res["six_point_error"] = str(e)
            ss["last_res"] = res
            ss["last_ctx"] = {"stat": stat_category, "line": float(line_val), "over": float(over_odds), "under": float(under_odds)}
        res = ss.get("last_res")
        if res:
            if "error" in res:
                st.error(res["error"])
            else:
                render_results(res, ss["last_ctx"]["stat"], ss["last_ctx"]["line"])

    with tab_bt:
        render_matchup_backtest()


if __name__ == "__main__":
    main()

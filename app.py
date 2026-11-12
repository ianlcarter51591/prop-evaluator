"""NFL Prop Evaluator: real opponent-vs-position matchups, devigged edge, calibrated matchup weights."""
import io
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import startup_check
    _STARTUP_PROBLEMS = startup_check.problems()
except ImportError:
    _STARTUP_PROBLEMS = []
if _STARTUP_PROBLEMS:
    st.set_page_config(page_title="NFL Prop Evaluator", page_icon="⚡", layout="wide")
    st.error("The app can't start: some files are missing or out of date.")
    for _p in _STARTUP_PROBLEMS:
        st.code(_p)
    st.caption("Upload the named file(s) to the same folder as app.py on GitHub, then reboot the app.")
    st.stop()

from multi_sport_engine import DEFAULTS, SOURCES, default_season, evaluate_prop, stat_options
from nfl_matchups import DEFAULT_K, run_backtests
import odds_credits
from six_point_core import LEGEND, assessment_summary
from nfl_six_point import (
    BLOWOUT_SPREAD, RECENT_GAMES, add_six_point_column, game_matches, label_gameday, opponent_from_label, style_tags,
)
from nfl_headshots import fetch_roster, player_photo
from odds_api_client import (
    MARKETS,
    OddsApiError,
    fetch_prop_board,
    list_events,
    player_offers,
    summarize_board,
)
from sports_data import NFL_TEAMS, clean_name, load_weekly
import cloud_store
import nfl_bet_log as bl
import nfl_snapshots as snaps_mod
import nfl_vegas
import shared_ui
import live_game

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
    """Coerce a price to American odds (decimal odds such as 1.91 are converted instead of truncated to 1 or 2)."""
    x = float(x)
    if abs(x) >= 100:
        return int(round(x))
    if x > 1.0:
        return int(round((x - 1.0) * 100.0)) if x >= 2.0 else int(round(-100.0 / (x - 1.0)))
    return -110


def load_k_file() -> Optional[Dict[str, float]]:
    """Saved calibrated matchup weights (best effort; ephemeral on Streamlit Cloud)."""
    try:
        if os.path.exists(K_PATH):
            with open(K_PATH) as fh:
                return json.load(fh)
    except Exception:
        pass
    return None


def save_k_file(k: Optional[Dict[str, float]]) -> None:
    """Persist (or clear) the calibrated weights."""
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


def _adj_cfg() -> Dict[str, Any]:
    """Injury settings from the 🩺 expander (shared by Evaluate and the board scan)."""
    ss = st.session_state
    split = lambda key: [x.strip() for x in str(ss[key]).split(",") if x.strip()]
    return {"injuries": ss["adj_inj"], "absorb_scale": float(ss["adj_absorb"]),
            "manual_out": split("adj_out"), "manual_q": split("adj_q"), "manual_in": split("adj_in"),
            "vegas": bool(ss.get("adj_vegas", True)), "vegas_k": ss.get("vegas_k")}


SNAP_NAME, VEGAS_K_NAME = "nfl_snapshots.csv", "nfl_vegas_k.json"


def _read_csv_text(text: Optional[str]) -> pd.DataFrame:
    """CSV text -> all-string frame (empty frame for no text)."""
    return pd.read_csv(io.StringIO(text), dtype=str) if text and text.strip() else pd.DataFrame()


def load_snapshots() -> pd.DataFrame:
    """Snapshots from permanent storage if configured, else the local file."""
    try:
        return snaps_mod.normalize(_read_csv_text(cloud_store.load_text(SNAP_NAME, snaps_mod.SNAP_PATH)))
    except Exception:
        return snaps_mod.load()


def save_snapshots(df: pd.DataFrame, force: bool = True) -> pd.DataFrame:
    """Save snapshots; if permanent storage hands back a saved copy we never loaded, merge it in and save again."""
    recovered = cloud_store.save_text(SNAP_NAME, df.to_csv(index=False), snaps_mod.SNAP_PATH, force)
    if recovered:
        try:
            df = snaps_mod.upsert(snaps_mod.normalize(_read_csv_text(recovered)), df.to_dict("records"))
            cloud_store.save_text(SNAP_NAME, df.to_csv(index=False), snaps_mod.SNAP_PATH, True)
        except Exception:
            pass
    return df


def load_bet_log() -> pd.DataFrame:
    """Bet log from permanent storage if configured, else the local file."""
    try:
        return bl.normalize_log(_read_csv_text(cloud_store.load_text(bl.LOG_NAME, bl.LOG_NAME)))
    except Exception:
        return bl.normalize_log(pd.DataFrame())


def save_bet_log(df: pd.DataFrame) -> pd.DataFrame:
    """Save the bet log (merging in a saved copy that was never loaded, if there is one)."""
    recovered = cloud_store.save_text(bl.LOG_NAME, df.to_csv(index=False), bl.LOG_NAME, True)
    if recovered:
        try:
            df = bl.merge_logs(_read_csv_text(recovered), df)
            cloud_store.save_text(bl.LOG_NAME, df.to_csv(index=False), bl.LOG_NAME, True)
        except Exception:
            pass
    return df


def load_vegas_k() -> Optional[Dict[str, float]]:
    """Backtested Vegas weights, if any were applied."""
    try:
        text = cloud_store.load_text(VEGAS_K_NAME, VEGAS_K_NAME)
        data = json.loads(text) if text and text.strip() else None
        return {str(k): float(v) for k, v in data.items()} if isinstance(data, dict) and data else None
    except Exception:
        return None


def save_vegas_k(k: Optional[Dict[str, float]]) -> None:
    """Persist (or clear) the Vegas weights."""
    cloud_store.save_text(VEGAS_K_NAME, json.dumps(k or {}), VEGAS_K_NAME, True)


def _now_et_text() -> str:
    """Current Eastern time as 'YYYY-MM-DD HH:MM' (UTC fallback if zoneinfo is unavailable)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def record_snapshot(res: Dict[str, Any], stat_category: str, line: float, over: float, under: float, game_label: str,
                    snap_source: str) -> None:
    """Save what the model believed for this prop (projection, probabilities, six-point) so it can be graded later.

    Never raises: a snapshot problem must not break an evaluation. If nfl_snapshots.py isn't uploaded yet it is skipped
    and main() shows a warning.
    """
    ss = st.session_state
    try:
        import nfl_snapshots as sn
    except ImportError:
        ss["snap_missing"] = True
        return
    try:
        if "snapshots" not in ss:
            ss["snapshots"] = load_snapshots()
        row = sn.make_row(res, stat_category, float(line), float(over), float(under), snap_source, _now_et_text(),
                          int(default_season()), (label_gameday(game_label) if game_label else "") or "")
        ss["snapshots"] = save_snapshots(sn.upsert(ss["snapshots"], [row]), force=snap_source != "scan")
    except Exception:
        pass


def run_evaluation(name: str, stat_category: str, line: float, over: float, under: float, source: str,
                   opponent: Optional[str], game_label: str = "", strict: bool = False,
                   snap_source: str = "evaluate") -> Dict[str, Any]:
    """Run one evaluation and save it as a prediction snapshot (see record_snapshot). `source` is the projection model."""
    res = _run_evaluation(name, stat_category, line, over, under, source, opponent, game_label, strict)
    if "error" not in res:
        record_snapshot(res, stat_category, line, over, under, game_label, snap_source)
    return res


def _run_evaluation(name: str, stat_category: str, line: float, over: float, under: float, source: str,
                    opponent: Optional[str], game_label: str = "", strict: bool = False) -> Dict[str, Any]:
    """One evaluation, six-point score included (the engine does both, so Evaluate and the board scan always agree).

    With a game selected in Step 2 the six-point spread/weather lookup is pinned to that game's date. If AUTO picks an
    opponent who isn't in the selected game (e.g. on game day the schedule has already moved on), the player is
    re-evaluated against the other team in that game.
    """
    ss = st.session_state
    gameday = label_gameday(game_label) if game_label else None

    def run(opp: Optional[str]) -> Dict[str, Any]:
        return evaluate_prop(player_name=name, stat_category=stat_category, line=float(line), over_odds=float(over),
                             under_odds=float(under), projection_source=source, season=default_season(), opponent=opp,
                             matchup_k=ss.get("matchup_k"), adj_cfg=_adj_cfg(), gameday=gameday)

    res = run(opponent)
    if game_label and opponent == "AUTO" and "error" not in res:
        if not game_matches(game_label, res.get("team"), (res.get("matchup") or {}).get("opponent")):
            opp = opponent_from_label(game_label, res.get("team"))
            if opp is None:
                if strict:      # board scan: the player must belong to the selected game
                    return {"error": f"{res.get('team')} isn't in this game"}
                return res      # manual Evaluate of someone from another game: keep the normal result
            res = run(opp)
    return res


def scan_board(summary: pd.DataFrame, stat_category: str, game_label: str, progress: Any = None) -> Dict[str, Any]:
    """Evaluate every player on the odds board; map each to his six-point result or to the reason he wasn't scored."""
    ss = st.session_state
    out: Dict[str, Any] = {}
    n = len(summary)
    for k, (_, r) in enumerate(summary.iterrows()):
        name = str(r["Player"])
        if r[["Line", "Over", "Under"]].isna().any():
            out[name] = "incomplete odds"
            continue
        if progress is not None:
            progress.progress(k / max(n, 1), text=f"Scanning {with_pos(name)} ({k + 1}/{n})")
        try:
            res = run_evaluation(name, stat_category, r["Line"], r["Over"], r["Under"],
                                 ss.get("input_source", SOURCES[0]), "AUTO", game_label, strict=True, snap_source="scan")
            if "error" in res:
                out[name] = str(res["error"])[:70]
            elif res.get("six_point"):
                out[name] = res["six_point"]
            else:
                out[name] = f"not scored: {res.get('six_point_error', 'no data')}"[:70]
        except Exception as e:
            out[name] = f"error: {str(e)[:60]}"
    if progress is not None:
        progress.progress(1.0, text="Scan complete")
    cloud_store.flush()                      # the scan saved many snapshots in a row: send the last ones now
    return out


def render_live_panel() -> None:
    """Odds API loader (unchanged flow): games -> board -> populate the form."""
    ss = st.session_state
    api_key = _secret("ODDS_API_KEY")
    with st.expander("📡 Fetch Live DraftKings Lines (The Odds API)", expanded=True):
        if not api_key:
            st.warning("Please configure `ODDS_API_KEY` in Streamlit Secrets.")
            return
        live_stat = st.selectbox("Select Prop Market", list(MARKETS["NFL"].keys()), key="live_stat")
        if st.button("Step 1: Load Upcoming Games", key="load_games"):
            try:
                events, remaining = list_events("NFL", api_key)
                odds_credits.record(remaining)
                ss["events"] = events
            except OddsApiError as exc:
                st.error(odds_credits.friendly_error(exc))
        events = ss.get("events")
        if events is not None and not events.empty:
            game_idx = st.selectbox("Step 2: Select Game", range(len(events)), format_func=lambda i: str(events.loc[i, "label"]), key="live_game")
            refresh = st.checkbox("Refresh odds (costs credits; otherwise a board fetched in the last 10 min is reused for free)",
                                  key="refresh_odds")
            if st.button(f"Step 3: Fetch {live_stat} Odds", key="fetch_board"):
                ev_id = str(events.loc[game_idx, "id"])

                def fresh() -> Any:
                    board_, remaining_ = fetch_prop_board("NFL", ev_id, live_stat, ("draftkings",), api_key)
                    odds_credits.record(remaining_)
                    return board_, remaining_

                try:
                    (board, _), cached_age = odds_credits.fetch_cached(("NFL", ev_id, live_stat), fresh, lambda v: not v[0].empty, refresh)
                    st.caption(odds_credits.cache_note(cached_age))
                    ss["board"], ss["board_scan"], ss["board_token"] = board, None, uuid.uuid4().hex
                    ss["board_game_label"] = str(events.loc[game_idx, "label"])
                    ss["board_stat"] = live_stat
                except OddsApiError as exc:
                    st.error(odds_credits.friendly_error(exc))
        _credit_text, _credit_level = odds_credits.credits_status()
        if _credit_level in ("low", "out"):
            st.warning(_credit_text)
        elif _credit_text:
            st.caption(_credit_text)
        board = ss.get("board")
        if board is not None and not board.empty:
            summary = summarize_board(board)
            shown = summary
            scan = ss.get("board_scan")
            if scan is not None and scan["token"] == ss.get("board_token"):
                shown = add_six_point_column(summary, scan["six"])
            _why = positions_problem()
            if _why:
                st.warning(f"Player positions couldn't be loaded: {_why}. Reload the page to retry.")
            shown_pos = shown.copy()
            shown_pos.insert(1, "Pos", shown_pos["Player"].map(lambda n: position_of(n) or "—"))
            st.dataframe(style_tags(shown_pos) if "Six-Point" in shown_pos else shown_pos, use_container_width=True)
            st.caption(LEGEND)
            if "Six-Point" in shown:
                tags = shown["Six-Point"].astype(str)
                scored = int((~tags.str.startswith("—")).sum())
                st.caption(f"Last scan: {int(tags.str.startswith('🔥').sum())} SHARP · {int(tags.str.startswith('⚡').sum())} LEAN · "
                           f"{scored} of {len(tags)} props scored.")
            if st.button("🔎 Scan board for 5/6 & 6/6 props", key="scan_board_btn"):
                bar = st.progress(0.0, text="Starting scan...")
                ss["board_scan"] = {"token": ss.get("board_token"), "six": scan_board(summary, live_stat, ss.get("board_game_label", ""), bar)}
                st.rerun()
            player_pick = st.selectbox("Step 4: Choose Player to Evaluate", shown["Player"].tolist(), key="pick_player", format_func=with_pos)
            offers = player_offers(board, player_pick)
            if not offers.empty:
                row = offers.iloc[0]
                if st.button(f"⚡ Populate {with_pos(player_pick)} ({row['Line']} | Over {row['Over']} / Under {row['Under']})", key="populate_btn"):
                    sync_populate_to_widgets(live_stat, player_pick, row["Line"], row["Over"], row["Under"])
                    st.rerun()


@st.cache_data(ttl=21600, show_spinner=False)
def cached_roster(team: str) -> Dict[str, Any]:
    """ESPN team roster (cached 6 hours) for team-aware headshots."""
    return fetch_roster(team)


def _player_photo(res: Dict[str, Any]) -> Optional[str]:
    """Headshot of the evaluated player on HIS team (a name-only search shows the wrong 'Josh Allen')."""
    if not res.get("team"):
        return live_game.headshot_url(res["player_name"])
    try:
        return player_photo(res["player_name"], res["team"], res.get("pos"), cached_roster)
    except Exception:
        return None


def _metric(col: Any, label: str, value: str, tone: str = "neutral") -> None:
    """Render one metric card; tone in {'neutral','good','bad'}."""
    color = {"neutral": "#0369a1", "good": "#15803d", "bad": "#b91c1c"}[tone]
    col.markdown(
        f'<div class="metric-box"><div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{color} !important;">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _tone(x: float) -> str:
    """Metric tone from sign."""
    return "good" if x > 0 else "bad"


def render_final_assessment(res: Dict[str, Any]) -> None:
    """Final assessment: six-point criteria for the better-scoring side (6/6 SHARP, 5/6 LEAN). Same result the odds table tags."""
    err = res.get("six_point_error")
    info = assessment_summary(res)
    if info is None:
        if err:
            st.caption(f"⚠️ Six-point assessment unavailable ({err}).")
        return
    best, other = info["best"], info["other"]
    tone = {"SHARP": "good", "LEAN": "neutral", "PASS": "bad"}[best["tier"]]
    st.markdown("### 🏁 Final Assessment: 6-Point Check")
    a, b, c = st.columns([1, 1, 2])
    _metric(a, f"TIER · {best['side'].upper()}", best["tier"], tone)
    _metric(b, f"CRITERIA FOR {best['side'].upper()}", f"{best['score']}/6", tone)
    c.caption(f"**{best['score']}/6 criteria support the {best['side']}** (6/6 = SHARP · 5/6 = LEAN · 4 or fewer = PASS). "
              f"The {other['side']} meets {other['score']}/6 ({other['tier']}). A check that can't be verified counts as failed.")
    if info["side_differs"]:
        st.info(f"The model's highest-EV side is {info['ev_side']}, but {best['side']} meets more of the six criteria.")
    if best["tier"] != "PASS" and info["side_ev_pct"] <= 0:
        st.warning(f"⚠️ {best['side']} meets the criteria but the model's EV at this price is {info['side_ev_pct']:+.1f}%. "
                   "Check the price before betting.")
    for ck in best["checks"]:
        icon = "✅" if ck["passed"] else ("⚪" if not ck["verified"] else "❌")
        st.markdown(f"{icon} **{ck['label']}**: {ck['detail']}")
    st.caption(f"Defensive ranks use what each defense allows this position group (season-long, and its last {RECENT_GAMES} games). "
               f"Spread comes from the ESPN scoreboard (blowout = {BLOWOUT_SPREAD:g}+); weather is the Open-Meteo forecast for kickoff at outdoor stadiums. "
               "Thresholds live at the top of nfl_six_point.py.")


@st.cache_data(ttl=3600, show_spinner=False)
def _name_positions(season: int) -> Dict[str, str]:
    """{normalized player name -> position group (QB/RB/WR/TE/K)} from the nflverse weekly data the app already loads (latest row wins)."""
    w = load_weekly(season, back=1).sort_values(["SEASON", "WEEK"])
    last = w.drop_duplicates("CNAME", keep="last")
    return dict(zip(last["CNAME"], last["POS"]))


def position_of(name: Any) -> Optional[str]:
    """Position group of a player (QB, RB, WR, TE, K), or None when nflverse doesn't list him."""
    try:
        return _name_positions(int(default_season())).get(clean_name(str(name)))
    except Exception:
        return None


def positions_problem() -> str:
    """Why positions can't be shown right now ('' when they load fine). Failures are never cached, so a reload retries."""
    try:
        _name_positions(int(default_season()))
        return ""
    except Exception as e:
        return str(e)


def with_pos(name: Any) -> str:
    """'Patrick Mahomes (QB)' for display; the plain name when the position is unknown."""
    p = position_of(name)
    return f"{name} ({p})" if p else str(name)


def render_matchup(res: Dict[str, Any], stat_category: str) -> None:
    """Real opponent-vs-position panel (replaces the old hash-generated placeholders)."""
    m = res.get("matchup")
    st.markdown("### 🛡️ Opponent vs this stat by position (real nflverse data)")
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
    src = "backtested" if m["calibrated"] else "default assumption; calibrate in the Matchup Backtest tab"
    notes = [f"Allowed to {m['pos']}s per game, shrunk toward last season ({m['games']} games this season). Weight k = {m['k']:.2f} ({src})."]
    if m.get("rest") is not None and m.get("opp_rest") is not None:
        flag = lambda d: " (short week)" if d <= 6 else " (post-bye)" if d >= 13 else ""
        notes.append(f"Rest days: you {m['rest']}{flag(m['rest'])}, opponent {m['opp_rest']}{flag(m['opp_rest'])}. Informational only; no adjustment.")
    env = res.get("vegas")
    if env:
        src = "backtested weight" if env.get("calibrated") else "default weight; calibrate in the Matchup Backtest tab"
        notes.append(f"💰 {env['note']} (k = {env['k']:.2f}, {src}).")
    for n in notes:
        st.caption(n)
    if m.get("by_pos"):
        st.caption(f"{res['player_name']} is a **{m['pos']}**, so the rank above is against {m['pos']}s. The same defense against every position for {stat_category}:")
        st.dataframe(pd.DataFrame(m["by_pos"]).rename(columns={"pos": "POSITION", "rank": "RANK (1 = SOFTEST)", "n_teams": "OF", "allowed": "ALLOWED / GM",
                                                               "index": "INDEX"}).round(2), use_container_width=True, hide_index=True)
    with st.expander(f"All 32 defenses vs {m['pos']} ({stat_category})"):
        board = m["board"].rename(columns={"EST": "ALLOWED (SHRUNK)", "CUR": "THIS SEASON"}).round(2)
        st.dataframe(board, use_container_width=True, hide_index=True)


def _scoreboard_body() -> None:
    """Grid of score cards for every NFL game; green = live; star = the evaluated player's game."""
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
    """Scoreboard expander; self-refreshing fragment when a game is live (needs Streamlit >= 1.37)."""
    any_live = any(r["state"] == "in" for r in live_game.scoreboard())
    with st.expander("📺 Live NFL Scoreboard", expanded=True):
        if hasattr(st, "fragment"):
            st.fragment(run_every=20 if any_live else None)(_scoreboard_body)()
        else:
            _scoreboard_body()


def render_live_script(res: Dict[str, Any], stat_category: str) -> None:
    """Live score + clock + box score -> game-script-adjusted verdict (pregame: shows status only)."""
    ctx = st.session_state.get("last_ctx", {})
    st.markdown("### 🏈 Live Game Script")
    spec = live_game.spec_for(stat_category)
    if spec is None:
        st.caption("No live game-script model for this stat (combos, INTs and longest-play props are skipped).")
        return
    game = live_game.find_game(res["team"])
    if game is None:
        st.caption(f"No {res['team']} game found on the current ESPN NFL scoreboard.")
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
    spread = st.number_input(
        f"Pregame spread for {game.team} (negative = favorite, 0 = ignore)", value=0.0, step=0.5, key="live_spread",
        help="Removes the script the market already expected, so only the deviation from it moves the projection.",
    )
    auto = live_game.fetch_player_stat(game.event_id, res["player_name"], spec)
    if auto is None:
        st.caption("No box-score entry found yet; assuming 0 so far. Edit below if that is wrong.")
    current = st.number_input(
        f"{stat_category} so far (auto-filled from the box score; edit to override)", value=float(auto or 0.0),
        step=1.0, key=f"live_cur_{res['player_name']}_{stat_category}_{auto}",
    )
    out = live_game.evaluate_live(res["mu"], spec, game, float(spread), float(current), float(ctx["line"]),
                                  float(ctx["over"]), float(ctx["under"]))
    word = "Trailing" if out["lean"] < -0.05 else "Leading" if out["lean"] > 0.05 else "Even"
    flow = "pass-heavy" if out["lean"] < -0.05 else "run-heavy" if out["lean"] > 0.05 else "neutral"
    c = st.columns(6)
    _metric(c[0], f"{game.team} vs {game.opp}", f"{game.team_score}-{game.opp_score} · {game.detail}")
    _metric(c[1], "GAME SCRIPT", f"{word} · {flow}", "good" if out["lean"] > 0.05 else "bad" if out["lean"] < -0.05 else "neutral")
    _metric(c[2], "SO FAR", f"{current:.1f}")
    _metric(c[3], "LIVE FINAL PROJ", f"{out['final_proj']:.1f}", _tone(out["final_proj"] - ctx["line"]))
    _metric(c[4], "LIVE VERDICT" if out["verdict"] == "PLAY" else "LIVE LEAN (PASS)", f"{out['side']} {ctx['line']}", _tone(out["ev_pct"]))
    _metric(c[5], "LIVE EV / EDGE", f"{out['ev_pct']:+.1f}% / {out['edge_pp']:+.1f}pp", _tone(out["ev_pct"]))
    d = st.columns(4)
    _metric(d[0], "P(OVER) / P(UNDER)", f"{out['p_over'] * 100:.1f}% / {out['p_under'] * 100:.1f}%")
    _metric(d[1], "REMAINING PROJ", f"{out['mu_rem']:.1f}")
    _metric(d[2], "PACE MULT (REST OF GAME)", f"x{out['mult']:.3f}", _tone(out["mult"] - 1))
    _metric(d[3], "LIVE KELLY (¼)", f"{out['kelly_pct']:.1f}% BR")
    st.caption(
        f"Game-script sensitivities are default assumptions (beta = {spec.beta:+.2f} on remaining volume). "
        "Compare against a fresh live line: a stale pregame price will overstate the edge."
    )


def render_contextual_reality_check() -> None:
    """Render a pre-flight contextual checklist to manually verify usage, schedule, and line movement."""
    with st.expander("🔍 Step 3: Contextual Reality Check (Pre-Flight Filter)", expanded=False):
        st.markdown(
            "Before locking in your bet, pass the model output through these real-world filters:"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox("1. Secure Volume & Usage", value=True, help="Does the player command high snap/touch/shot volume?")
        with col2:
            st.checkbox("2. Clean Schedule Spot", value=True, help="No unmanaged fatigue, rest disadvantages, or blowout risk.")
        with col3:
            st.checkbox("3. Sharp Line Movement", value=True, help="Line movement supports the side or aligns with sharp money.")
            
        st.caption("💡 *Note: High-volume players insulate you from single-game variance, while back-to-backs lean toward Unders due to tired legs.*")


def render_injury_usage(res: Dict[str, Any]) -> None:
    """Injury / opportunity-redistribution panel (teammates out -> target or carry share moves)."""
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
    if inj.get("listed"):
        st.caption("🩺 Injury report for " + str(res.get("team", "this team")) + ": " + " · ".join(f"{x['name']} ({x['status']})" for x in inj["listed"]))
    elif inj.get("feed_ok", True):
        st.caption("🩺 The injury feed lists no relevant teammates for this team.")
    if inj["table"]:
        st.dataframe(pd.DataFrame(inj["table"]), use_container_width=True, hide_index=True)
        st.caption("Weight = P(misses) minus the share of his recent games already missed (no double counting; a returning teammate pulls usage back down). "
                   "Absorption rates are assumptions; scale them in the injury settings.")
    else:
        st.caption("No injured or returning teammates move this projection. Add names in the injury settings if the feed is stale.")
    st.caption("The projection, simulation, edge, EV and verdict above already include this usage adjustment.")


def render_tiers(res: Dict[str, Any], stat_category: str) -> None:
    """Player's average vs the softest / toughest defenses for this stat (context only)."""
    t = res.get("tiers")
    if not t:
        return
    st.markdown("### 🎯 Player vs Defense Tiers")
    fmt = lambda x: "n/a" if x["avg"] is None else f"{x['avg']:.1f} ({x['n']} g)"
    c = st.columns(5)
    for col, lab, key in zip(c, ["ALL GAMES", "VS 5 SOFTEST", "VS 10 SOFTEST", "VS 10 TOUGHEST", "VS 5 TOUGHEST"], ["overall", "soft5", "soft10", "tough10", "tough5"]):
        _metric(col, lab, fmt(t[key]))
    st.caption(f"Each defense is ranked per season by what it allowed {res.get('pos', 'this position')}s in {stat_category} (1 = softest). Small samples: context, not an adjustment.")


def render_results(res: Dict[str, Any], stat_category: str, line_val: float) -> None:
    """Metric cards, matchup panel, notes, and the game-log chart."""
    photo_col, title_col = st.columns([1, 6])
    photo = _player_photo(res)
    if photo:
        photo_col.image(photo, width=120)
    title_col.markdown(f"## {res['player_name']} ({res['pos']}) | {stat_category}")
    title_col.caption(f"{res['team']} · {res['pos']} · {res['n_current']} games this season, {res['n_prior']} last season")
    c = st.columns(6)
    _metric(c[0], "ADJ PROJ", f"{res['mu']:.2f}")
    _metric(c[1], "EWMA (PRE-MATCHUP)", f"{res['base_proj']:.2f}")
    _metric(c[2], "BEST SIDE (BY EV)", res["best_side"], _tone(res["ev_pct"]))
    _metric(c[3], "FAIR ODDS", f"{res['fair_odds']:+d}")
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
    render_log_form(res, stat_category, line_val)
    st.markdown("---")
    
    # Contextual Reality Check Card
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


def render_snapshots() -> None:
    """Every evaluation and board scan, saved automatically; grade them against real results and see why picks missed."""
    ss = st.session_state
    try:
        import nfl_snapshots as sn
    except ImportError:
        st.warning("nfl_snapshots.py is not uploaded to the app's GitHub repo yet.")
        return
    if "snapshots" not in ss:
        ss["snapshots"] = load_snapshots()
    snaps = ss["snapshots"]
    st.caption("Saved automatically each time you press Evaluate or scan a board (latest snapshot per player, stat, line and game).")
    shared_ui.render_storage_status(st)
    a, b = st.columns(2)
    if a.button("🏁 Grade finished games", key="grade_snaps"):
        from sports_data import fetch_nfl_player_logs
        graded, n, notes = sn.grade(snaps, fetch_nfl_player_logs)
        ss["snapshots"] = save_snapshots(graded)
        ss["snap_notes"] = notes[:5]
        st.toast(f"Graded {n} snapshot(s)")
        st.rerun()
    b.download_button("⬇️ Download snapshots (CSV)", data=snaps.to_csv(index=False).encode("utf-8"),
                      file_name="nfl_snapshots.csv", mime="text/csv", key="dl_snaps")
    for note in ss.pop("snap_notes", []):
        st.warning(note)
    up = st.file_uploader("Import saved snapshots (CSV)", type="csv", key="snap_upload")
    if up is not None:
        uid = getattr(up, "file_id", f"{up.name}-{up.size}")
        if ss.get("snap_upload_done") != uid:
            try:
                ss["snapshots"] = save_snapshots(sn.upsert(snaps, sn.normalize(pd.read_csv(up, dtype=str)).to_dict("records")))
                ss["snap_upload_done"] = uid
                st.rerun()
            except Exception as e:
                st.error(f"Could not read that file: {e}")
    snaps = ss["snapshots"]
    if snaps.empty:
        st.info("No snapshots yet. Evaluate a prop or scan a board and they will appear here.")
        return
    tiers = sn.by_tier(snaps)
    if not tiers.empty:
        st.markdown("**Graded results by tier shown**")
        st.dataframe(tiers.round(1), use_container_width=True, hide_index=True)
    shared_ui.render_validation(st, snaps)
    st.markdown("### 📋 Every saved prediction")
    show = st.radio("Show", ["All", "Pending", "Graded", "Misses"], horizontal=True, key="snap_filter")
    view = snaps
    if show == "Pending":
        view = snaps[snaps["STATUS"] == "Pending"]
    elif show == "Graded":
        view = snaps[snaps["STATUS"] == "Graded"]
    elif show == "Misses":
        view = snaps[snaps["SHOWN_RESULT"] == "Miss"]
    cols = ["SEASON", "WEEK", "PLAYER", "STAT", "LINE", "SHOWN_SIDE", "TIER_SHOWN", "SIX_OVER", "SIX_UNDER", "MU", "ACTUAL", "Z",
            "VOL_EXP", "VOL_ACT", "SHOWN_RESULT", "WHY", "P_OVER", "P_UNDER", "FAILED_OVER", "FAILED_UNDER", "OPPONENT", "SOURCE", "TS_ET"]
    st.dataframe(view.sort_values(["SEASON", "WEEK", "PLAYER"], ascending=[False, False, True])[cols].round(2),
                 use_container_width=True, hide_index=True)
    st.caption("WHY is a heuristic: volume (attempts or receptions) 25% or more off his normal is a usage miss; a result more than 1 sd "
               "(his last-10 spread) against the pick is an outlier; otherwise normal variance. Z = (actual - projection) / sd. "
               "A player with no stats for a week that later weeks logged is voided (did not play).")


def render_log_form(res: Dict[str, Any], stat_category: str, line_val: float) -> None:
    """'Log this bet' under an evaluation: saves the side you took with the model and market numbers at that moment."""
    ss = st.session_state
    if "sides" not in res:
        return
    st.markdown("### 📝 Log this bet")
    c1, c2, c3 = st.columns(3)
    default_side = 0 if str(res.get("best_side", "")).startswith("Over") else 1
    side = c1.selectbox("Side", ["Over", "Under"], index=default_side, key=f"log_side_{res.get('player_name')}_{stat_category}_{line_val}")
    stake = c2.number_input("Stake (units)", min_value=0.01, value=1.0, step=0.25, key="log_stake")
    c3.write("")
    if c3.button("Log bet", key="log_btn"):
        row = bl.make_bet_row(res, stat_category, float(line_val), side if side in ("Over", "Under") else "Over", float(stake or 1.0), int(default_season()))
        ss["bet_log"], added = bl.add_bet(ss.get("bet_log", bl.normalize_log(pd.DataFrame())), row)
        if added:
            ss["bet_log"] = save_bet_log(ss["bet_log"])
            ss["editor_ver"] = ss.get("editor_ver", 0) + 1
            st.toast(f"Logged {row['SIDE']} {float(line_val):g} — {res['player_name']}")
        else:
            st.warning("That exact bet (player, stat, side, line, game) is already in your log.")


def _fmt(x: Optional[float], spec: str = ".1f", suffix: str = "") -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:{spec}}{suffix}"


def render_bet_log() -> None:
    """Bets you placed: closing lines, auto-grading by week, units, ROI, closing-line value and calibration."""
    ss = st.session_state
    if "bet_log" not in ss:
        ss["bet_log"] = load_bet_log()
    ss.setdefault("editor_ver", 0)
    shared_ui.render_storage_status(st)
    a, b, c = st.columns(3)
    if a.button("🏁 Auto-grade finished games", key="grade_btn"):
        from sports_data import fetch_nfl_player_logs
        graded, n, notes = bl.auto_grade(ss["bet_log"], fetch_nfl_player_logs)
        ss["bet_log"] = save_bet_log(graded)
        ss["editor_ver"] += 1
        ss["grade_notes"] = notes[:5]
        st.toast(f"Graded {n} bet(s)")
        st.rerun()
    board = ss.get("board")
    if b.button("📸 Use current board as closing lines", key="close_btn"):
        if board is None or board.empty or not ss.get("board_stat"):
            st.warning("Fetch a prop board first (Evaluate tab, Step 3), ideally right before kickoff.")
        else:
            updated, n = bl.apply_board_as_closing(ss["bet_log"], summarize_board(board), ss["board_stat"])
            ss["bet_log"] = save_bet_log(updated)
            ss["editor_ver"] += 1
            st.toast(f"Closing lines applied to {n} pending bet(s) ({ss['board_stat']} market)")
            st.rerun()
    c.download_button("⬇️ Download log (CSV)", data=ss["bet_log"].to_csv(index=False).encode("utf-8"),
                      file_name=bl.LOG_NAME, mime="text/csv", key="dl_log")
    for note in ss.pop("grade_notes", []):
        st.warning(note)
    up = st.file_uploader("Import a saved bet log (CSV)", type="csv", key="log_upload")
    if up is not None:
        uid = getattr(up, "file_id", f"{up.name}-{up.size}")
        if ss.get("_up_id") != uid:
            try:
                ss["bet_log"] = save_bet_log(bl.merge_logs(ss["bet_log"], pd.read_csv(up, dtype=str)))
                ss["_up_id"] = uid
                ss["editor_ver"] += 1
                st.rerun()
            except Exception as e:
                st.error(f"Could not import that file: {e}")
    log: pd.DataFrame = ss["bet_log"]
    if log.empty:
        st.info("No bets logged yet. Evaluate a prop, then use “Log this bet” under the results.")
        return
    cols = [c_ for c_ in bl.LOG_COLS if c_ not in ("ID", "LAST_WEEK_SEEN")] + ["LAST_WEEK_SEEN", "ID"]
    key = f"bet_editor_{ss['editor_ver']}"
    fmt = lambda label: st.column_config.NumberColumn(label, format="%.1f")
    edited = st.data_editor(
        log[cols], key=key, num_rows="dynamic", hide_index=True, use_container_width=True,
        disabled=[c_ for c_ in cols if c_ not in bl.EDITABLE_COLS],
        column_config={
            "ID": None, "TS": None, "LAST_WEEK_SEEN": None,
            "RESULT": st.column_config.SelectboxColumn("RESULT", options=bl.RESULT_OPTIONS, required=True),
            "P_MODEL": fmt("P_MODEL %"), "P_MARKET": fmt("P_MKT %"), "EDGE_PP": fmt("EDGE pp"), "EV_PCT": fmt("EV %"),
            "LINE_MOVE": fmt("LINE MOVE"), "CLV_PP": fmt("CLV pp"), "PROFIT": st.column_config.NumberColumn("PROFIT (u)", format="%.2f"),
        },
    )
    state = ss.get(key) or {}
    if isinstance(edited, pd.DataFrame) and (state.get("edited_rows") or state.get("added_rows") or state.get("deleted_rows")):
        ss["bet_log"] = save_bet_log(bl.normalize_log(edited))
        ss["editor_ver"] += 1
        st.rerun()
    stats, by_stat, calib = bl.summarize_log(ss["bet_log"])
    st.markdown("### 📈 Performance")
    m = st.columns(6)
    _metric(m[0], "BETS / GRADED", f"{int(stats['bets'])} / {int(stats['graded'])}")
    _metric(m[1], "WIN %", _fmt(stats["win_pct"], ".1f", "%"))
    _metric(m[2], "UNITS", _fmt(stats["units"], "+.2f"), _tone(stats["units"]) if np.isfinite(stats["units"]) else "neutral")
    _metric(m[3], "ROI", _fmt(stats["roi"], "+.1f", "%"), _tone(stats["roi"]) if np.isfinite(stats["roi"]) else "neutral")
    _metric(m[4], "AVG CLV", _fmt(stats["avg_clv"], "+.2f", " pp"), _tone(stats["avg_clv"]) if np.isfinite(stats["avg_clv"]) else "neutral")
    _metric(m[5], "+CLV RATE", _fmt(stats["pos_clv"], ".0f", "%"))
    st.caption(
        f"CLV is measured on {int(stats['n_clv'])} bet(s) with closing odds at the same line. "
        f"Brier (lower is better): model {_fmt(stats['brier_model'], '.4f')} vs market {_fmt(stats['brier_market'], '.4f')}. "
        "Treat everything as noise until you have 50+ graded bets; CLV converges much faster than win/loss. "
        "To record closing lines, fetch the board right before kickoff and press the closing-lines button, or type them into the table."
    )
    st.markdown("**By stat category**")
    st.dataframe(by_stat, use_container_width=True, hide_index=True)
    st.markdown("**Calibration (graded bets)**")
    st.dataframe(calib, use_container_width=True, hide_index=True)


def render_vegas_backtest() -> None:
    """Calibrate how much of the market's implied team points should reach each stat (closing lines, no look-ahead)."""
    ss = st.session_state
    st.markdown("---")
    st.subheader("Vegas implied points")
    st.markdown(
        "Estimates **k** in `stat ÷ player's season average = 1 + k × (implied team points ÷ team's usual points − 1)`, using each "
        "game's closing total and spread, a points baseline built only from earlier games, and the opponent index in the same "
        "regression, so k is what Vegas adds **on top of** the matchup factor. Results are shrunk toward the default before use."
    )
    c1, c2 = st.columns(2)
    n_seasons = c1.slider("Seasons to backtest", 1, 4, 3, key="vbt_seasons")
    c2.write("")
    if c2.button("🔬 Run Vegas backtest", key="run_vbt"):
        cur = default_season()
        with st.spinner("Loading nflverse seasons and closing lines…"):
            try:
                n = int(n_seasons or 3)
                ss["vbt_table"] = nfl_vegas.run_backtests(load_weekly(cur, back=n), nfl_vegas.load_lines(cur, n), cur - n + 1)
            except Exception as e:
                ss["vbt_table"] = None
                st.error(f"Backtest failed: {e}")
    table = ss.get("vbt_table")
    if table is not None and not table.empty:
        st.dataframe(table.round(3), use_container_width=True, hide_index=True)
        st.caption("K_V is the raw estimate, K_EFF is what the model will use. SIGNIFICANT = |K_V| ≥ 2 × SE_V. "
                   "An estimated points baseline dilutes K_V somewhat, so it tends to read low rather than high.")
        a, b = st.columns(2)
        if a.button("✅ Apply to model", key="apply_vbt"):
            ss["vegas_k"] = {r.STAT: float(r.K_EFF) for r in table.itertuples()}
            save_vegas_k(ss["vegas_k"])
            st.toast("Calibrated Vegas weights applied")
        if b.button("↩️ Reset to defaults", key="reset_vbt"):
            ss["vegas_k"] = None
            save_vegas_k(None)
            st.toast("Back to default Vegas weights")
    elif table is not None:
        st.warning("Not enough player-games for any stat. Try more seasons.")
    active = ss.get("vegas_k")
    st.markdown("**Active Vegas weights**" + (" (backtested)" if active else " (default assumptions)"))
    st.dataframe(pd.DataFrame({"STAT": list(nfl_vegas.DEFAULT_K),
                               "k": [round((active or nfl_vegas.DEFAULT_K).get(s_, nfl_vegas.DEFAULT_K[s_]), 3) for s_ in nfl_vegas.DEFAULT_K]}),
                 use_container_width=True, hide_index=True)


def render_matchup_backtest() -> None:
    """Calibrate how much weight the opponent index deserves, using only information available before each game."""
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
        st.caption("K_HAT is the raw estimate, K_EFF is what the model will use. SIGNIFICANT = |K_HAT| ≥ 2 × SE. "
                   "Stats where it's 'no' are still dominated by noise; K_EFF stays close to the default.")
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
    cloud_store.configure(_secret("GITHUB_GIST_TOKEN"))
    if "matchup_k" not in ss:
        ss["matchup_k"] = load_k_file()
    if "vegas_k" not in ss:
        ss["vegas_k"] = load_vegas_k()
    if "bet_log" not in ss:
        ss["bet_log"] = load_bet_log()
    ss.setdefault("editor_ver", 0)
    for key, default in (("input_player", def_player), ("input_stat", stat_options()[0]), ("input_line", float(def_line)),
                         ("input_over", -110), ("input_under", -110), ("input_opp", AUTO_OPP), ("input_source", SOURCES[0]),
                         ("adj_inj", True), ("adj_absorb", 1.0), ("adj_out", ""), ("adj_q", ""), ("adj_in", ""),
                         ("adj_vegas", True)):
        ss.setdefault(key, default)

    if ss.get("snap_missing"):
        st.warning("nfl_snapshots.py is not uploaded to the app's GitHub repo, so predictions are NOT being saved. Upload it (it is in the zip).")
    tab_eval, tab_log, tab_snap, tab_parlay, tab_bt = st.tabs(["⚡ Evaluate", "📒 Bet Log", "📸 Snapshots", "🎟️ Parlay", "🔬 Matchup Backtest"])
    with tab_eval:
        render_live_scoreboard()
        render_live_panel()
        with st.expander("🩺 Injuries & usage settings", expanded=False):
            st.checkbox("Adjust for injured / returning teammates (target and carry share redistribution)", key="adj_inj")
            st.slider("Absorption strength (scales how much of an absent player's share teammates pick up)", 0.5, 1.0, key="adj_absorb", step=0.05)
            st.text_input("Teammates OUT (comma-separated; adds to the ESPN feed)", key="adj_out")
            st.text_input("Teammates QUESTIONABLE (counted as 50%)", key="adj_q")
            st.text_input("Teammates ACTIVE (overrides a stale injury feed)", key="adj_in")
            st.caption("Applies when you press Evaluate. Check the team's report close to kickoff: the feed can lag.")
            st.checkbox("Use Vegas implied team points (total and spread) in the projection", key="adj_vegas")
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
            res = run_evaluation(player_name, stat_category, line_val, over_odds, under_odds, proj_source, opponent,
                                 ss.get("board_game_label", "") if opponent == "AUTO" else "")
            ss["last_res"] = res
            ss["last_ctx"] = {"stat": stat_category, "line": float(line_val), "over": float(over_odds), "under": float(under_odds)}
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
                "- **Final assessment (6-point check)**: baby line, AI edge, implied chance, defensive rank (season + last 4 games), injury flags, game script (spread under 10, no bad weather). "
                "6/6 = SHARP, 5/6 = LEAN; checks that can't be verified count as failed. Thresholds are in nfl_six_point.py.\n"
                "- **Kelly**: ¼ Kelly capped at 5% of bankroll; 0 when EV ≤ 0.\n"
                "- **Vegas implied points**: implied team points = total ÷ 2 ± spread ÷ 2, compared with the team's usual points (earlier games this season, shrunk to last season). "
                "Multiplier = 1 + k × (ratio − 1), capped ±8%, applied on top of matchup and usage. k is a default assumption until you run the Vegas backtest.\n"
                "- **Injuries & usage**: injured teammates' target share (receiving) or carry share (rushing) is redistributed to the others in proportion to their own share "
                "(same-position teammates weighted 1.5x), net of what the player's recent games already reflect, so a returning teammate pulls usage back down. "
                "Passing props lose a little efficiency when top targets are out, and receivers lose a little when the QB is out. The projection is shifted and the verdict re-priced; absorption rates are assumptions."
            )
    with tab_log:
        render_bet_log()
    with tab_snap:
        render_snapshots()
    with tab_parlay:
        if "snapshots" not in ss:
            ss["snapshots"] = load_snapshots()
        shared_ui.render_parlay(st, ss["snapshots"])
    with tab_bt:
        render_matchup_backtest()
        render_vegas_backtest()


if __name__ == "__main__":
    main()

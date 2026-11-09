"""Integration test: real multi_sport_engine + nfl_matchups + nfl_adjust + nfl_six_point on synthetic league data.
Stubs only the modules not shipped here (sports_data, odds_api_client, live_game, plotly, streamlit).
Run: python -m unittest -v test_nfl_integration"""
from __future__ import annotations

import re
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

TEAMS = ["ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
         "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"]
STATS = ["PASSING YARDS", "PASS COMPLETIONS", "PASS ATTEMPTS", "PASSING TDS", "RUSHING YARDS", "RUSHING ATTEMPTS",
         "RECEIVING YARDS", "RECEPTIONS", "INTERCEPTIONS", "TARGETS"]
SEASON = 2026


def clean_name(n) -> str:
    return re.sub(r"[^a-z0-9]", "", str(n).lower())


def make_weekly(den_factor: float) -> pd.DataFrame:
    """32 teams x (2025 wk1-17, 2026 wk1-9). Opponent pairing by XOR so every game is symmetric."""
    rng = np.random.default_rng(7)
    roster = [("QB", 1), ("RB", 1), ("WR", 3), ("TE", 1)]
    base = {"QB": {"PASSING YARDS": 255, "PASS COMPLETIONS": 23, "PASS ATTEMPTS": 35, "PASSING TDS": 1.6, "RUSHING YARDS": 12,
                   "RUSHING ATTEMPTS": 3, "INTERCEPTIONS": 0.7},
            "RB": {"RUSHING YARDS": 62, "RUSHING ATTEMPTS": 14, "RECEIVING YARDS": 20, "RECEPTIONS": 2.5, "TARGETS": 3.2},
            "WR": {"RECEIVING YARDS": 58, "RECEPTIONS": 4.8, "TARGETS": 7.0},
            "TE": {"RECEIVING YARDS": 40, "RECEPTIONS": 3.8, "TARGETS": 5.0}}
    rows = []
    for season, weeks in ((SEASON - 1, 17), (SEASON, 9)):
        for wk in range(1, weeks + 1):
            k = (wk % 31) + 1
            for i, team in enumerate(TEAMS):
                opp = TEAMS[i ^ k]
                for pos, n in roster:
                    for j in range(n):
                        pid = f"{team}-{pos}{j}"
                        row = {"SEASON": season, "WEEK": wk, "TEAM": team, "OPP": opp, "PID": pid,
                               "NAME": f"{team} {pos}{j}", "CNAME": clean_name(f"{team} {pos}{j}"), "POS": pos,
                               **{s: 0.0 for s in STATS}}
                        for s, m in base[pos].items():
                            f = den_factor if (opp == "DEN" and s in ("PASSING YARDS", "PASS COMPLETIONS", "PASS ATTEMPTS")) else 1.0
                            row[s] = max(0.0, rng.normal(m * f * (1.0 - 0.12 * j), m * 0.18))
                        rows.append(row)
    return pd.DataFrame(rows)


class FakeSt(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.calls, self.session_state = [], {}

    def cache_data(self, *a, **k):
        return a[0] if a and callable(a[0]) else (lambda f: f)

    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [MagicMock() for _ in range(n)]

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, a, k))


def install(weekly: pd.DataFrame):
    """Install stub modules and import the app fresh."""
    qb = weekly[weekly["PID"] == "KC-QB0"].sort_values(["SEASON", "WEEK"])
    logs = qb.assign(SEASON_TAG=np.where(qb["SEASON"] == SEASON, "current", "prior"),
                     DATE=[f"{s}-{w:02d}" for s, w in zip(qb["SEASON"], qb["WEEK"])], OPPONENT=qb["OPP"])
    sd = types.ModuleType("sports_data")
    sd.NFL_TEAMS = TEAMS
    sd.clean_name = clean_name
    sd.clean_series = lambda s: s.map(clean_name)
    sd.current_nfl_season = lambda: SEASON
    sd.fetch_nfl_player_logs = lambda name, season: (logs, None)
    sd.load_schedule = lambda season: None
    sd.load_weekly = lambda season, back=1: weekly
    sd.next_game = lambda team, sched: {"opponent": "DEN", "week": 10, "home": False, "gameday": "2026-11-08", "rest": 7, "opp_rest": 7}
    sys.modules["sports_data"] = sd
    sys.modules["streamlit"] = FakeSt()
    for name in ("plotly", "plotly.graph_objects", "odds_api_client", "live_game"):
        sys.modules[name] = MagicMock()
    for name in ("app", "multi_sport_engine", "nfl_adjust", "nfl_six_point", "nfl_matchups"):
        sys.modules.pop(name, None)
    import app
    import nfl_adjust
    return app, nfl_adjust


def feed(rows):
    return pd.DataFrame(rows, columns=["NAME", "STATUS", "P_OUT"]), True


class TestNflIntegration(unittest.TestCase):
    def evaluate(self, den_factor=1.35, injuries=None, adj=None, line=235.5, over=-120, under=100, feed_ok=True):
        weekly = make_weekly(den_factor)
        app, adj_mod = install(weekly)
        data = feed(injuries or []) if feed_ok else (pd.DataFrame(columns=["NAME", "STATUS", "P_OUT"]), False)
        p = patch.object(adj_mod, "load_injuries", return_value=data)
        p.start()
        self.addCleanup(p.stop)
        import multi_sport_engine as eng
        res = eng.evaluate_prop("KC QB0", "Passing Yards", line, over, under, season=SEASON, opponent="AUTO",
                                adj_cfg=adj or {"injuries": True, "absorb_scale": 1.0, "manual_out": [], "manual_q": [], "manual_in": []})
        self.assertNotIn("error", res, res.get("error"))
        return app, weekly, res

    def assess(self, app, weekly, res, spread=4.5, home="DET", gust=10.0):
        import nfl_six_point as sp
        game = sp.GameInfo(home, "KC", "2026-11-08T18:00Z", spread, False, None)
        fc = lambda a, b: {"hourly": {"time": pd.date_range("2026-11-08T12:00", periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist(),
                                       "wind_speed_10m": [5.0] * 24, "wind_gusts_10m": [gust] * 24, "precipitation": [0.0] * 24,
                                       "snowfall": [0.0] * 24, "temperature_2m": [55.0] * 24}}
        return sp.assess(res, "Passing Yards", 235.5, weekly, SEASON, lambda *a: game, fc)

    def test_injury_layer_exposes_the_new_flags(self):
        app, weekly, res = self.evaluate()
        inj = res["inj"]
        for key in ("checked", "self_status", "qb_flag", "feed_ok", "listed"):
            self.assertIn(key, inj)
        self.assertTrue(inj["checked"])
        self.assertEqual((inj["self_status"], inj["qb_flag"]), ("", ""))

    def test_soft_defense_over_everything_clear_is_sharp(self):
        app, weekly, res = self.evaluate(den_factor=1.35)
        self.assertEqual(res["best_side"].split()[0], "Over")
        six = self.assess(app, weekly, res)
        by = {c["key"]: c for c in six["Over"]["checks"]}
        self.assertTrue(by["defensive_rank"]["passed"], by["defensive_rank"]["detail"])
        self.assertTrue(by["injury"]["passed"] and by["game_script"]["passed"] and by["baby_line"]["passed"])
        self.assertEqual(six["best"], "Over")
        self.assertIn(six["Over"]["tier"], {"SHARP", "LEAN"})     # edge / implied depend on the simulated sample
        self.assertEqual(len(six["Over"]["checks"]), 6)

    def test_player_himself_flagged_fails_injury_check(self):
        app, weekly, res = self.evaluate(injuries=[("KC QB0", "Questionable", 0.4)])
        self.assertIn("Questionable", res["inj"]["self_status"])
        six = self.assess(app, weekly, res)
        c = [c for c in six["Over"]["checks"] if c["key"] == "injury"][0]
        self.assertFalse(c["passed"])
        self.assertIn("the player", c["detail"])

    def test_listed_skill_teammate_fails_injury_check(self):
        app, weekly, res = self.evaluate(injuries=[("KC WR0", "Out", 1.0)])
        six = self.assess(app, weekly, res)
        c = [c for c in six["Over"]["checks"] if c["key"] == "injury"][0]
        self.assertFalse(c["passed"])
        self.assertIn("KC WR0", c["detail"])

    def test_manual_override_marks_the_player(self):
        adj = {"injuries": True, "absorb_scale": 1.0, "manual_out": [], "manual_q": ["KC QB0"], "manual_in": []}
        app, weekly, res = self.evaluate(adj=adj)
        self.assertIn("manual", res["inj"]["self_status"].lower())

    def test_injury_adjustment_off_or_feed_down_is_unverified(self):
        app, weekly, res = self.evaluate(adj={"injuries": False})
        six = self.assess(app, weekly, res)
        c = [c for c in six["Over"]["checks"] if c["key"] == "injury"][0]
        self.assertEqual((c["passed"], c["verified"]), (False, False))
        app, weekly, res = self.evaluate(feed_ok=False)
        six = self.assess(app, weekly, res)
        c = [c for c in six["Over"]["checks"] if c["key"] == "injury"][0]
        self.assertEqual((c["passed"], c["verified"]), (False, False))

    def test_tough_defense_fails_over_rank(self):
        app, weekly, res = self.evaluate(den_factor=0.65)
        six = self.assess(app, weekly, res)
        d = [c for c in six["Over"]["checks"] if c["key"] == "defensive_rank"][0]
        self.assertFalse(d["passed"])
        u = [c for c in six["Under"]["checks"] if c["key"] == "defensive_rank"][0]
        self.assertTrue(u["passed"], u["detail"])

    def test_blowout_and_wind_fail_game_script(self):
        app, weekly, res = self.evaluate()
        six = self.assess(app, weekly, res, spread=13.5)
        self.assertFalse([c for c in six["Over"]["checks"] if c["key"] == "game_script"][0]["passed"])
        six = self.assess(app, weekly, res, home="GB", gust=30.0)
        g = [c for c in six["Over"]["checks"] if c["key"] == "game_script"][0]
        self.assertFalse(g["passed"])
        self.assertIn("wind", g["detail"])

    def test_existing_engine_fields_untouched(self):
        app, weekly, res = self.evaluate()
        for k in ("verdict", "best_side", "ev_pct", "edge_pp", "p_market", "matchup", "inj", "tiers", "mu", "kelly_pct"):
            self.assertIn(k, res)

    def test_app_wrappers_and_render(self):
        app, weekly, res = self.evaluate()
        import nfl_six_point as sp
        game = sp.GameInfo("DET", "KC", "2026-11-08T18:00Z", 4.5, False, None)
        with patch.object(app, "fetch_game_info", return_value=game) as fg:
            self.assertEqual(app.cached_game_info("KC", "DEN", "2026-11-08"), game)
            fg.assert_called_once_with("KC", "DEN", "2026-11-08")
        res["six_point"] = sp.assess(res, "Passing Yards", 235.5, weekly, SEASON, lambda *a: game, lambda a, b: {})
        app.st.calls.clear()
        app.render_final_assessment(res)
        md = [c[1][0] for c in app.st.calls if c[0] == "markdown"]
        self.assertEqual(len([m for m in md if m[:1] in "✅❌⚪"]), 6)
        self.assertTrue(any("Final Assessment" in m for m in md))
        res2 = {**res, "six_point": None, "six_point_error": "boom"}
        app.st.calls.clear()
        app.render_final_assessment(res2)
        self.assertTrue(any("boom" in str(c) for c in app.st.calls))


class TestNflBoardScan(unittest.TestCase):
    """scan_board runs the real engine per player and tags only players in the selected game."""
    LABEL = "Kansas City Chiefs @ Denver Broncos (2026-11-08 18:00 UTC)"

    def setUp(self):
        weekly = make_weekly(1.35)
        self.weekly = weekly
        self.app, adj_mod = install(weekly)
        import multi_sport_engine as eng
        import nfl_six_point as sp
        self.sp = sp

        def logs_for(name, season):
            pid = name.replace(" ", "-")
            rows = weekly[weekly["PID"] == pid].sort_values(["SEASON", "WEEK"])
            if rows.empty:
                return None, f"No game logs found for '{name}'."
            return rows.assign(SEASON_TAG=np.where(rows["SEASON"] == SEASON, "current", "prior"),
                               DATE=[f"{s}-{w:02d}" for s, w in zip(rows["SEASON"], rows["WEEK"])], OPPONENT=rows["OPP"]), None

        nxt = {"KC": "DEN", "DEN": "KC", "MIA": "BUF"}
        game = sp.GameInfo("DET", "KC", "2026-11-08T18:00Z", 4.5, False, None)
        for target, obj in ((eng, {"fetch_nfl_player_logs": logs_for,
                                   "next_game": lambda team, s: {"opponent": nxt.get(team, "KC"), "week": 10, "home": False,
                                                                 "gameday": "2026-11-08", "rest": 7, "opp_rest": 7}}),
                            (adj_mod, {"load_injuries": lambda: feed([])}),
                            (self.app, {"fetch_game_info": lambda *a: game, "fetch_forecast": lambda a, b: {}})):
            for name, fn in obj.items():
                p = patch.object(target, name, fn)
                p.start()
                self.addCleanup(p.stop)
        self.app.st.session_state.update({"adj_inj": True, "adj_absorb": 1.0, "adj_out": "", "adj_q": "", "adj_in": ""})

    def summary(self):
        return pd.DataFrame({"Player": ["KC QB0", "DEN QB0", "MIA QB0", "Nobody Here", "No Odds"],
                             "Line": [235.5] * 5, "Over": [-120] * 4 + [float("nan")], "Under": [100] * 5})

    def test_scores_players_in_the_selected_game_only(self):
        six = self.app.scan_board(self.summary(), "Passing Yards", self.LABEL)
        self.assertEqual(set(six), {"KC QB0", "DEN QB0", "MIA QB0", "Nobody Here", "No Odds"})
        for scored in ("KC QB0", "DEN QB0"):
            self.assertIsNotNone(six[scored], scored)
            self.assertEqual(len(six[scored][six[scored]["best"]]["checks"]), 6)
        for skipped in ("MIA QB0", "Nobody Here", "No Odds"):       # other game / no logs / incomplete odds
            self.assertIsNone(six[skipped], skipped)

    def test_kc_qb_against_a_soft_denver_defense_scores_well(self):
        six = self.app.scan_board(self.summary(), "Passing Yards", self.LABEL)
        best = six["KC QB0"][six["KC QB0"]["best"]]
        by = {c["key"]: c for c in best["checks"]}
        self.assertTrue(by["defensive_rank"]["passed"], by["defensive_rank"]["detail"])
        self.assertGreaterEqual(best["score"], 5)
        self.assertIn(best["tier"], {"SHARP", "LEAN"})

    def test_uses_the_form_settings(self):
        seen = []
        real = self.app.evaluate_prop
        def spy(**kw):
            seen.append(kw)
            return real(**kw)
        self.app.st.session_state.update({"input_source": "My mean", "adj_q": "KC WR0, KC WR1"})
        with patch.object(self.app, "evaluate_prop", side_effect=spy):
            self.app.scan_board(self.summary().head(1), "Passing Yards", self.LABEL)
        self.assertEqual(seen[0]["projection_source"], "My mean")
        self.assertEqual(seen[0]["opponent"], "AUTO")
        self.assertEqual(seen[0]["adj_cfg"]["manual_q"], ["KC WR0", "KC WR1"])

    def test_errors_and_progress(self):
        with patch.object(self.app, "evaluate_prop", side_effect=RuntimeError("boom")):
            six = self.app.scan_board(self.summary(), "Passing Yards", self.LABEL)
        self.assertTrue(all(v is None for v in six.values()))
        bar = MagicMock()
        self.app.scan_board(self.summary(), "Passing Yards", self.LABEL, bar)
        self.assertEqual(bar.progress.call_args.args[0], 1.0)
        self.assertGreaterEqual(bar.progress.call_count, 3)

    def test_tagged_table(self):
        six = self.app.scan_board(self.summary(), "Passing Yards", self.LABEL)
        table = self.sp.add_six_point_column(self.summary(), six)
        self.assertEqual(table[self.sp.TAG_COL].iloc[-1], "—")
        self.assertTrue(any(t.startswith(("🔥", "⚡")) for t in table[self.sp.TAG_COL].iloc[:2]))


if __name__ == "__main__":
    unittest.main()

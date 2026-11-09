"""Unit tests for nfl_six_point.py. Run: python -m unittest -v test_nfl_six_point"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import nfl_six_point as sp

CLEAN_INJ = {"checked": True, "feed_ok": True, "self_status": "", "listed": [], "qb_flag": ""}
GOOD_WX = sp.Weather(True, False, "indoor stadium, no weather risk")


def score(side="Over", stat="PASSING YARDS", line=244.5, proj=270.0, p_market=0.55, season=(3, 32), recent=(5, 32),
          inj=None, spread=6.5, weather=GOOD_WX):
    return sp.score_prop(side, stat, line, proj, p_market, season[0], season[1], recent[0], recent[1], "QB",
                         CLEAN_INJ if inj is None else inj, spread, weather)


class TestTiersAndChecks(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual([sp.tier_for(n) for n in range(7)], ["PASS"] * 5 + ["LEAN", "SHARP"])

    def test_perfect_over_is_sharp(self):
        r = score()
        self.assertEqual((r.score, r.tier), (6, "SHARP"))

    def test_one_miss_lean_two_miss_pass(self):
        self.assertEqual(score(spread=12.5).tier, "LEAN")
        self.assertEqual(score(spread=12.5, p_market=0.49).tier, "PASS")

    def test_baby_line(self):
        self.assertFalse(sp.check_line("PASSING YARDS", 150.5).passed)
        self.assertTrue(sp.check_line("PASSING YARDS", 175.5).passed)
        self.assertFalse(sp.check_line("RECEPTIONS", 2.5).passed)
        self.assertTrue(sp.check_line("INTERCEPTIONS", 0.5).passed)

    def test_edge(self):
        self.assertTrue(sp.check_edge("Over", "PASSING YARDS", 244.5, 270).passed)
        self.assertFalse(sp.check_edge("Over", "PASSING YARDS", 244.5, 255).passed)    # 10.5 < 14.7 (6%)
        self.assertFalse(sp.check_edge("Over", "PASSING YARDS", 244.5, 230).passed)    # wrong way
        self.assertTrue(sp.check_edge("Under", "PASSING YARDS", 244.5, 225).passed)
        self.assertFalse(sp.check_edge("Over", "RECEPTIONS", 4.5, 4.9).passed)         # < 0.75 abs
        self.assertTrue(sp.check_edge("Over", "RECEPTIONS", 4.5, 5.4).passed)

    def test_implied(self):
        self.assertTrue(sp.check_implied("Over", 0.51).passed)
        self.assertFalse(sp.check_implied("Over", 0.5).passed)

    def test_defense_with_32_teams(self):
        self.assertTrue(sp.check_defense("Over", 16, 32, 1, 32).passed)
        self.assertFalse(sp.check_defense("Over", 17, 32, 1, 32).passed)
        self.assertFalse(sp.check_defense("Over", 1, 32, 17, 32).passed)
        self.assertTrue(sp.check_defense("Under", 17, 32, 32, 32).passed)
        self.assertFalse(sp.check_defense("Under", 16, 32, 32, 32).passed)

    def test_defense_unverified(self):
        for args in ((None, 32, 5, 32), (5, 32, None, None), (None, None, None, None)):
            c = sp.check_defense("Over", *args)
            self.assertEqual((c.passed, c.verified), (False, False))

    def test_injury(self):
        self.assertTrue(sp.check_injury(CLEAN_INJ).passed)
        c = sp.check_injury({**CLEAN_INJ, "self_status": "Questionable"})
        self.assertFalse(c.passed)
        self.assertIn("the player", c.detail)
        c = sp.check_injury({**CLEAN_INJ, "listed": [{"name": "WR1", "status": "OUT", "p_out": 1.0}]})
        self.assertIn("WR1 (OUT)", c.detail)
        c = sp.check_injury({**CLEAN_INJ, "qb_flag": "QB1 OUT"})
        self.assertFalse(c.passed)
        # listed but p_out 0 (manual ACTIVE override) is not a flag
        self.assertTrue(sp.check_injury({**CLEAN_INJ, "listed": [{"name": "X", "status": "ACTIVE (manual)", "p_out": 0.0}]}).passed)

    def test_injury_unverified(self):
        for inj in (None, {}, {"checked": False, "feed_ok": True}, {"checked": True, "feed_ok": False}):
            c = sp.check_injury(inj)
            self.assertEqual((c.passed, c.verified), (False, False))

    def test_game_script(self):
        self.assertTrue(sp.check_game_script(9.5, GOOD_WX).passed)
        self.assertFalse(sp.check_game_script(10.0, GOOD_WX).passed)
        bad = sp.check_game_script(3.0, sp.Weather(True, True, "bad weather: wind 22 mph"))
        self.assertEqual((bad.passed, bad.verified), (False, True))
        for args in ((None, GOOD_WX), (3.0, None), (3.0, sp.Weather(False, False, "forecast unavailable"))):
            c = sp.check_game_script(*args)
            self.assertEqual((c.passed, c.verified), (False, False))

    def test_unverifiable_blocks_sharp(self):
        self.assertEqual(score(weather=None).tier, "LEAN")
        self.assertEqual(score(weather=None, season=(None, None)).tier, "PASS")


class TestRecentRank(unittest.TestCase):
    def weekly(self):
        rows = []
        for season in (2025, 2026):
            for wk in range(1, 9):
                for i in range(6):
                    team, opp = f"T{i}", f"D{i}"
                    # D0 allows a lot only in its last 4 games; D1 allows a lot all year; others normal
                    v = 250.0
                    if opp == "D0" and season == 2026 and wk >= 5:
                        v = 380.0
                    if opp == "D1":
                        v = 330.0
                    rows.append({"SEASON": season, "WEEK": wk, "TEAM": team, "OPP": opp, "POS": "QB", "PASSING YARDS": v})
        return pd.DataFrame(rows)

    def test_recent_rank_uses_last_n_games_only(self):
        r, n = sp.recent_defense_rank(self.weekly(), "PASSING YARDS", "QB", 2026, "D0")
        self.assertEqual((r, n), (1, 6))                         # soft recently despite normal season
        r1, _ = sp.recent_defense_rank(self.weekly(), "PASSING YARDS", "QB", 2026, "D1")
        self.assertEqual(r1, 2)
        r2, _ = sp.recent_defense_rank(self.weekly(), "PASSING YARDS", "QB", 2026, "D5")
        self.assertGreaterEqual(r2, 3)

    def test_unknown_or_thin_opponent(self):
        self.assertEqual(sp.recent_defense_rank(self.weekly(), "PASSING YARDS", "QB", 2026, "ZZZ"), (None, None))
        thin = self.weekly()
        thin = thin[~((thin["OPP"] == "D2") & (thin["SEASON"] == 2026) & (thin["WEEK"] > 1))]
        self.assertEqual(sp.recent_defense_rank(thin, "PASSING YARDS", "QB", 2026, "D2"), (None, None))

    def test_abbreviation_canonicalised(self):
        w = self.weekly().replace({"OPP": {"D0": "LA"}})
        self.assertEqual(sp.recent_defense_rank(w, "PASSING YARDS", "QB", 2026, "LAR")[0], 1)


def scoreboard(*events):
    return {"events": list(events)}


def event(home, away, odds=None, neutral=False, indoor=None, date="2026-11-08T18:00Z"):
    comp = {"date": date, "neutralSite": neutral, "venue": {"indoor": indoor} if indoor is not None else {},
            "competitors": [{"homeAway": "home", "team": {"abbreviation": home}},
                            {"homeAway": "away", "team": {"abbreviation": away}}]}
    if odds is not None:
        comp["odds"] = odds
    return {"date": date, "competitions": [comp]}


class TestScoreboard(unittest.TestCase):
    def test_numeric_spread_and_text_spread(self):
        g = sp.parse_scoreboard_game(scoreboard(event("DEN", "KC", [{"spread": -3.5}])), "KC", "DEN")
        self.assertEqual((g.home, g.away, g.spread, g.neutral), ("DEN", "KC", 3.5, False))
        g = sp.parse_scoreboard_game(scoreboard(event("DEN", "KC", [{"details": "KC -7"}])), "KC")
        self.assertEqual(g.spread, 7.0)
        g = sp.parse_scoreboard_game(scoreboard(event("DEN", "KC", [{"details": "EVEN"}])), "KC")
        self.assertEqual(g.spread, 0.0)

    def test_no_odds(self):
        self.assertIsNone(sp.parse_scoreboard_game(scoreboard(event("DEN", "KC")), "KC").spread)
        self.assertIsNone(sp.parse_scoreboard_game(scoreboard(event("DEN", "KC", [])), "KC").spread)

    def test_picks_the_right_game_and_aliases(self):
        sb = scoreboard(event("NYG", "PHI", [{"spread": -1.5}]), event("WSH", "LAR", [{"spread": -9.5}]))
        g = sp.parse_scoreboard_game(sb, "LA", "WAS")
        self.assertEqual((g.home, g.away, g.spread), ("WAS", "LA", 9.5))
        self.assertIsNone(sp.parse_scoreboard_game(sb, "KC"))
        self.assertIsNone(sp.parse_scoreboard_game(sb, "PHI", "WAS"))     # right team, wrong opponent

    def test_neutral_and_indoor_flags(self):
        g = sp.parse_scoreboard_game(scoreboard(event("JAX", "BUF", [], neutral=True, indoor=False)), "BUF")
        self.assertTrue(g.neutral)
        self.assertFalse(g.indoor)


def forecast(wind=5.0, gust=10.0, precip=0.0, snow=0.0, temp=55.0, start="2026-11-08T12:00"):
    times = pd.date_range(start, periods=24, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    n = len(times)
    return {"hourly": {"time": times, "wind_speed_10m": [wind] * n, "wind_gusts_10m": [gust] * n,
                       "precipitation": [precip] * n, "snowfall": [snow] * n, "temperature_2m": [temp] * n}}


class TestWeather(unittest.TestCase):
    def test_window_is_three_hours_from_kickoff(self):
        p = forecast()
        p["hourly"]["wind_speed_10m"][6] = 30.0       # 18:00 (kickoff hour)
        p["hourly"]["wind_speed_10m"][9] = 40.0       # 21:00 (after the window)
        w = sp.summarize_forecast(p, "2026-11-08T18:00Z")
        self.assertEqual(w["wind"], 30.0)

    def test_kickoff_not_on_the_hour_floors(self):
        p = forecast()
        p["hourly"]["precipitation"][8] = 4.0         # 20:00
        w = sp.summarize_forecast(p, "2026-11-08T20:20Z")
        self.assertEqual(w["precip"], 4.0)

    def test_missing_hours(self):
        self.assertIsNone(sp.summarize_forecast({"hourly": {}}, "2026-11-08T18:00Z"))
        self.assertIsNone(sp.summarize_forecast(forecast(), "2026-12-25T18:00Z"))

    def test_judge(self):
        ok = sp.judge_weather({"wind": 8, "gust": 14, "precip": 0.0, "snow": 0.0, "temp": 48})
        self.assertEqual((ok.verified, ok.bad), (True, False))
        for w, txt in (({"wind": 16, "gust": 20, "precip": 0, "snow": 0, "temp": 50}, "wind"),
                       ({"wind": 5, "gust": 26, "precip": 0, "snow": 0, "temp": 50}, "wind"),
                       ({"wind": 5, "gust": 8, "precip": 3.0, "snow": 0, "temp": 50}, "rain"),
                       ({"wind": 5, "gust": 8, "precip": 0, "snow": 0.6, "temp": 30}, "snow"),
                       ({"wind": 5, "gust": 8, "precip": 0, "snow": 0, "temp": 15}, "°F")):
            r = sp.judge_weather(w)
            self.assertTrue(r.bad)
            self.assertIn(txt, r.detail)

    def info(self, home, **kw):
        base = dict(home=home, away="KC", kickoff_iso="2026-11-08T18:00Z", spread=3.5, neutral=False, indoor=None)
        return sp.GameInfo(**{**base, **kw})

    def test_game_weather_branches(self):
        calls = []
        def fc(lat, lon):
            calls.append((lat, lon))
            return forecast()
        self.assertTrue(sp.game_weather(self.info("DET"), fc).verified)           # dome
        self.assertEqual(calls, [])                                               # no forecast fetched for domes
        self.assertIn("retractable", sp.game_weather(self.info("DAL"), fc).detail)
        self.assertIn("canopy", sp.game_weather(self.info("LAC"), fc).detail)
        out = sp.game_weather(self.info("DEN"), fc)
        self.assertEqual((out.verified, out.bad), (True, False))
        self.assertEqual(calls, [(39.7439, -105.0201)])
        self.assertFalse(sp.game_weather(self.info("DEN", neutral=True), fc).verified)
        self.assertFalse(sp.game_weather(self.info("XXX"), fc).verified)
        self.assertFalse(sp.game_weather(None, fc).verified)

    def test_game_weather_failure_and_bad(self):
        def boom(lat, lon):
            raise RuntimeError("timeout")
        r = sp.game_weather(self.info("GB"), boom)
        self.assertEqual((r.verified, r.bad), (False, False))
        self.assertIn("timeout", r.detail)
        r = sp.game_weather(self.info("GB"), lambda a, b: forecast(wind=22.0))
        self.assertEqual((r.verified, r.bad), (True, True))

    def test_every_team_has_a_stadium(self):
        from nfl_matchups import DEFAULT_K  # noqa: F401 (import sanity)
        teams = {"ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
                 "LA", "LAC", "LV", "MIA", "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"}
        self.assertEqual(set(sp.STADIUMS), teams)
        for lat, lon, roof in sp.STADIUMS.values():
            self.assertTrue(24 < lat < 49 and -125 < lon < -66)
            self.assertIn(roof, {"outdoor", "dome", "retractable", "canopy"})


class TestAssess(unittest.TestCase):
    def res(self, best="Over 244.5", p_market=55.0, inj=None):
        return {"best_side": best, "p_market": p_market, "mu": 270.0, "team": "KC",
                "matchup": {"opponent": "DEN", "pos": "QB", "rank": 3, "n_teams": 32, "gameday": "2026-11-08"},
                "inj": CLEAN_INJ if inj is None else inj}

    def weekly(self):
        rows = [{"SEASON": 2026, "WEEK": w, "TEAM": f"T{i}", "OPP": opp, "POS": "QB", "PASSING YARDS": 400.0 if opp == "DEN" else 250.0}
                for w in range(1, 9) for i, opp in enumerate(["DEN"] + [f"D{j}" for j in range(31)])]
        return pd.DataFrame(rows)

    def test_both_sides_scored_and_market_prob_flipped(self):
        game = sp.GameInfo("DET", "KC", "2026-11-08T18:00Z", 4.5, False, True)
        out = sp.assess(self.res(), "Passing Yards", 244.5, self.weekly(), 2026, lambda *a: game, lambda a, b: forecast())
        self.assertEqual(out["best"], "Over")
        self.assertEqual((out["Over"]["score"], out["Over"]["tier"]), (6, "SHARP"))
        # Under: edge wrong way, market 45%, defense soft -> fails several
        self.assertEqual(out["Under"]["tier"], "PASS")
        implied_under = [c for c in out["Under"]["checks"] if c["key"] == "implied_chance"][0]
        self.assertIn("45.0%", implied_under["detail"])

    def test_game_lookup_failure_marks_game_script_unverified(self):
        def boom(*a):
            raise RuntimeError("espn down")
        out = sp.assess(self.res(), "Passing Yards", 244.5, self.weekly(), 2026, boom, lambda a, b: forecast())
        g = [c for c in out["Over"]["checks"] if c["key"] == "game_script"][0]
        self.assertEqual((g["passed"], g["verified"]), (False, False))
        self.assertEqual(out["Over"]["tier"], "LEAN")

    def test_no_matchup_or_weekly_makes_defense_unverified(self):
        r = self.res()
        r["matchup"] = None
        out = sp.assess(r, "Passing Yards", 244.5, None, 2026, lambda *a: None, lambda a, b: forecast())
        d = [c for c in out["Over"]["checks"] if c["key"] == "defensive_rank"][0]
        self.assertEqual((d["passed"], d["verified"]), (False, False))


class TestTagging(unittest.TestCase):
    def six(self, score_, side="Over"):
        return {"best": side, side: {"side": side, "score": score_, "tier": sp.tier_for(score_), "checks": []}}

    def board(self):
        return pd.DataFrame({"Player": ["A", "B", "C", "D", "E"], "Line": [1.5, 2.5, 3.5, 4.5, 5.5],
                             "Over": [-110] * 5, "Under": [-110] * 5})

    def test_game_matches_label(self):
        label = "Kansas City Chiefs @ Denver Broncos (2026-11-09 18:00 UTC)"
        self.assertTrue(sp.game_matches(label, "KC", "DEN"))
        self.assertTrue(sp.game_matches(label, "DEN", "KC"))
        self.assertFalse(sp.game_matches(label, "KC", "LV"))          # right team, wrong opponent
        self.assertFalse(sp.game_matches(label, "MIA", "BUF"))
        self.assertFalse(sp.game_matches(label, None, "DEN"))
        self.assertFalse(sp.game_matches(label, "XXX", "DEN"))
        self.assertTrue(sp.game_matches("Los Angeles Rams @ Washington Commanders", "LAR", "WSH"))   # ESPN codes canonicalised

    def test_every_team_has_a_nickname(self):
        self.assertEqual(set(sp.NICKNAMES), set(sp.STADIUMS))

    def test_tag_text(self):
        self.assertEqual(sp.tag_text(self.six(6)), "🔥 SHARP 6/6 · Over")
        self.assertEqual(sp.tag_text(self.six(5, "Under")), "⚡ LEAN 5/6 · Under")
        self.assertEqual(sp.tag_text(self.six(4)), "4/6")
        self.assertEqual(sp.tag_text(None), "—")

    def test_column_and_sorting(self):
        six = {"A": self.six(4), "B": self.six(6), "C": None, "D": self.six(5, "Under"), "E": self.six(4)}
        out = sp.add_six_point_column(self.board(), six)
        self.assertEqual(out["Player"].tolist(), ["B", "D", "A", "E", "C"])
        self.assertEqual(out[sp.TAG_COL].tolist(), ["🔥 SHARP 6/6 · Over", "⚡ LEAN 5/6 · Under", "4/6", "4/6", "—"])
        self.assertEqual(self.board().columns.tolist(), ["Player", "Line", "Over", "Under"])
        self.assertTrue((sp.add_six_point_column(self.board(), {})[sp.TAG_COL] == "—").all())

    def test_row_styling(self):
        out = sp.add_six_point_column(self.board(), {"A": self.six(6), "B": self.six(5), "C": self.six(3)})
        html = sp.style_tags(out).to_html()
        self.assertIn("#166534", html)
        self.assertIn("#a16207", html)


if __name__ == "__main__":
    unittest.main()

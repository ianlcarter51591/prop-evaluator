"""Recommended model settings at startup and the restore button (NFL)."""
import unittest

from test_nfl_integration import install, make_weekly


class TestSettings(unittest.TestCase):
    def test_restore_and_list_what_is_off(self):
        app, _ = install(make_weekly(1.0))
        ss = app.st.session_state
        app.restore_settings()
        self.assertEqual(app.settings_off(ss), [])
        ss.update(adj_role_auto=False, adj_absorb=0.5)
        self.assertEqual(app.settings_off(ss), ["Absorption strength: 0.5 (recommended 1)", "Detect role changes: off (recommended on)"])
        app.restore_settings()
        self.assertEqual(app.settings_off(ss), [])


if __name__ == "__main__":
    unittest.main()

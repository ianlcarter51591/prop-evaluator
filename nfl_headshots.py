"""Team-aware player headshots (a plain name search returns the wrong person for common names like Josh Allen)."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

from nfl_six_point import canon

ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team}/roster"
HEADSHOT_URL = "https://a.espncdn.com/i/headshots/nfl/players/full/{pid}.png"
ESPN_CODES: Dict[str, str] = {"WAS": "wsh", "LA": "lar"}      # nflverse code -> ESPN team slug (others: lower-case)
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def espn_team_code(team: Any) -> str:
    """nflverse team code -> ESPN roster slug."""
    c = canon(team)
    return ESPN_CODES.get(c, c.lower())


def norm_name(name: Any) -> str:
    """Lower-case alphanumeric name without Jr/Sr/II suffixes."""
    toks = re.sub(r"[^a-z0-9 ]+", "", str(name or "").lower()).split()
    return "".join(t for t in toks if t not in SUFFIXES)


def _dicts(obj: Any) -> Iterator[Mapping[str, Any]]:
    """Every dict nested anywhere in a JSON structure."""
    stack = [obj]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            yield o
            stack.extend(o.values())
        elif isinstance(o, list):
            stack.extend(o)


def find_headshot(roster: Any, name: str, pos: Optional[str] = None) -> Optional[str]:
    """Headshot URL of `name` in an ESPN team-roster payload (position breaks ties), else None."""
    want = norm_name(name)
    hits = [d for d in _dicts(roster)
            if norm_name(d.get("fullName") or d.get("displayName")) == want and (d.get("id") or d.get("headshot"))]
    if not hits:
        return None
    hits.sort(key=lambda d: (str((d.get("position") or {}).get("abbreviation", "")).upper() != str(pos or "").upper()))
    best = hits[0]
    href = (best.get("headshot") or {}).get("href")
    return str(href) if href else HEADSHOT_URL.format(pid=best["id"])


def fetch_roster(team: Any) -> Dict[str, Any]:
    """ESPN roster for one team (network call)."""
    import requests
    resp = requests.get(ROSTER_URL.format(team=espn_team_code(team)), timeout=8)
    resp.raise_for_status()
    return resp.json()


def player_photo(name: str, team: Any, pos: Optional[str] = None,
                 get_roster: Callable[[Any], Mapping[str, Any]] = fetch_roster) -> Optional[str]:
    """Headshot for `name` on `team` (None if the roster has no such player)."""
    return find_headshot(get_roster(team), name, pos)

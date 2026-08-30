#!/usr/bin/env python3
"""
mlb_hit_probs.py — daily batter prop-probability scanner.

Pulls the day's MLB slate, matches every posted lineup against that day's
starting pitcher, and estimates each batter's chance of:

    1+ hit      2+ hits      1+ run scored      1+ RBI

No API key. No third-party packages (stdlib only).

Usage:
    python3 mlb_hit_probs.py
    python3 mlb_hit_probs.py --date 2026-08-29 --min-prob 0.70
    python3 mlb_hit_probs.py --format html --out report.html
    python3 mlb_hit_probs.py --selftest       # offline math check, no network
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date as _date, datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

API = "https://statsapi.mlb.com/api/v1"
UA = "mlb-hit-probs/2.0 (personal use)"

# Timezone used for displayed times. Change to "America/Chicago", etc.
DISPLAY_TZ = "America/New_York"

# ---------------------------------------------------------------------------
# Model constants — tune these.
# ---------------------------------------------------------------------------

LEAGUE_FALLBACK = {
    "h_pa": 0.2215,    # hits per plate appearance
    "ob_pa": 0.3120,   # times on base per PA (H + BB + HBP)
    "hr_pa": 0.0320,   # home runs per PA
    "rbi_pa": 0.1150,  # RBI per PA
    "conv": 0.2900,    # share of non-HR baserunners who come around to score
}

# Regression constants. Higher = trust the small sample less.
K_PRIOR_PA = 150.0       # league prior added to a batter's own season line
K_BATTER_SPLIT = 200.0   # prior added to a batter's platoon split
K_PITCHER_SPLIT = 250.0  # prior added to a pitcher's platoon split
K_BATTER_FORM = 320.0    # prior added to recent-form rate

# Expected plate appearances by lineup slot (1-9), full 9-inning game.
PA_BY_SLOT = [4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95, 3.85]
HOME_PA_FACTOR = 0.975

BULLPEN_ADJ = 0.98
DEFAULT_STARTER_BF = 22.0

# Runs and RBI cluster (a 3-RBI game is one event, not three). A plain Poisson
# overstates the chance of at least one, so divide the rate by these.
RUN_DISPERSION = 1.30
RBI_DISPERSION = 1.35

# --- Combined hits + runs + RBI ("H+R+RBI") -------------------------------
# One plate appearance can produce several at once: a home run is 1 hit,
# 1 run and at least 1 RBI, so it is worth 3+ on its own.
HR_RBI_DIST = ((1, 0.62), (2, 0.26), (3, 0.12))  # RBI driven in by a homer
HIT_RBI_P = 0.30       # chance a non-homer hit drives in a run
OUT_RBI_P = 0.04       # sac fly / productive out
WALK_SCORE_ADJ = 0.90  # walks come around to score slightly less than hits
HRR_CAP = 6            # track totals up to this, everything above lumped in

# Rough hit park factors. EDIT THESE to taste — deliberately mild.
PARK_FACTORS = {
    "Coors Field": 1.08, "Fenway Park": 1.04, "Great American Ball Park": 1.02,
    "Chase Field": 1.02, "Kauffman Stadium": 1.02, "Yankee Stadium": 1.01,
    "Wrigley Field": 1.01, "Globe Life Field": 1.01, "Citizens Bank Park": 1.01,
    "PNC Park": 1.01, "Oriole Park at Camden Yards": 1.00, "Rogers Centre": 1.00,
    "Target Field": 1.00, "Truist Park": 1.00, "Nationals Park": 1.00,
    "Rate Field": 1.00, "Guaranteed Rate Field": 1.00, "Sutter Health Park": 1.00,
    "Busch Stadium": 0.99, "Progressive Field": 0.99, "Comerica Park": 0.99,
    "American Family Field": 0.99, "Angel Stadium": 0.99, "Daikin Park": 0.99,
    "Minute Maid Park": 0.99, "Citi Field": 0.98, "loanDepot park": 0.98,
    "George M. Steinbrenner Field": 0.98, "Tropicana Field": 0.98,
    "Dodger Stadium": 0.97, "Oracle Park": 0.96, "Petco Park": 0.96,
    "T-Mobile Park": 0.95,
}

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def _to_display_tz(dt: datetime) -> tuple[datetime, str]:
    if ZoneInfo is not None:
        try:
            dt = dt.astimezone(ZoneInfo(DISPLAY_TZ))
            return dt, (dt.strftime("%Z") or "ET")
        except Exception:
            pass
    return dt.astimezone(timezone(timedelta(hours=-4))), "ET"


def now_local_str() -> str:
    """Current time in DISPLAY_TZ, e.g. '2026-08-29 9:42am EDT'."""
    dt, label = _to_display_tz(datetime.now(timezone.utc))
    hour = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    return f"{dt:%Y-%m-%d} {hour}:{dt.minute:02d}{ampm} {label}"


def local_time(iso_utc: str) -> str:
    """'2026-08-24T23:05:00Z' -> '7:05p' in DISPLAY_TZ."""
    if not iso_utc:
        return "-"
    try:
        dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ")
        dt, _ = _to_display_tz(dt.replace(tzinfo=timezone.utc))
        hour = dt.hour % 12 or 12
        return f"{hour}:{dt.minute:02d}{'a' if dt.hour < 12 else 'p'}"
    except Exception:
        return "-"


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

_cache: dict[str, dict] = {}


def api_get(path: str, params: dict | None = None, retries: int = 3) -> dict:
    qs = urllib.parse.urlencode(params or {}, doseq=True)
    url = f"{API}/{path.lstrip('/')}" + (f"?{qs}" if qs else "")
    if url in _cache:
        return _cache[url]
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            _cache[url] = data
            return data
        except Exception as exc:
            last = exc
            time.sleep(0.8 * (attempt + 1))
    print(f"  ! request failed: {url}\n    {last}", file=sys.stderr)
    return {}


# ---------------------------------------------------------------------------
# Stat helpers
# ---------------------------------------------------------------------------


def _f(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def batter_pa(stat: dict) -> float:
    pa = _f(stat, "plateAppearances")
    if pa:
        return pa
    return (_f(stat, "atBats") + _f(stat, "baseOnBalls") + _f(stat, "hitByPitch")
            + _f(stat, "sacFlies") + _f(stat, "sacBunts"))


def pitcher_bf(stat: dict) -> float:
    bf = _f(stat, "battersFaced")
    if bf:
        return bf
    return (_f(stat, "atBats") + _f(stat, "baseOnBalls") + _f(stat, "hitByPitch")
            + _f(stat, "sacFlies") + _f(stat, "sacBunts"))


def on_base(stat: dict) -> float:
    return _f(stat, "hits") + _f(stat, "baseOnBalls") + _f(stat, "hitByPitch")


def shrink(events: float, opportunities: float, prior_rate: float, k: float) -> float:
    if opportunities <= 0:
        return prior_rate
    return (events + prior_rate * k) / (opportunities + k)


def log5(batter_rate: float, pitcher_rate: float, league_rate: float) -> float:
    """Odds-ratio matchup estimate (Bill James' log5)."""
    lo, hi = 1e-4, 0.9999
    b = min(max(batter_rate, lo), hi)
    p = min(max(pitcher_rate, lo), hi)
    l = min(max(league_rate, lo), hi)
    num = (b * p) / l
    den = num + ((1 - b) * (1 - p)) / (1 - l)
    return num / den if den > 0 else b


def pa_vs_starter(slot: int, starter_bf: float) -> float:
    """Expected PA that lineup slot `slot` (1-9) gets against the starter."""
    def turns(n: int) -> int:
        return 0 if n < slot else (n - slot) // 9 + 1
    lo = int(math.floor(starter_bf))
    frac = starter_bf - lo
    return turns(lo) * (1 - frac) + turns(lo + 1) * frac


def hit_distribution(p_sp: float, p_bp: float, pa_sp: float,
                     total_pa: float) -> tuple[float, float]:
    """P(at least 1 hit), P(at least 2 hits).

    A batter gets a whole number of plate appearances, not 4.35 of them.
    Using expected PA as a fractional exponent overstates the chance of a hit,
    because (1-p)^n is convex in n. So model the PA count as a mixture of the
    two whole numbers it falls between and work out the exact hit distribution
    for each.
    """
    n_lo = int(math.floor(total_pa))
    w_hi = total_pa - n_lo
    k_sp = int(round(pa_sp))

    p0 = p1 = 0.0
    for n, w in ((n_lo, 1 - w_hi), (n_lo + 1, w_hi)):
        if w <= 0:
            continue
        if n <= 0:
            p0 += w
            continue
        slots = [p_sp] * min(k_sp, n) + [p_bp] * max(0, n - k_sp)
        d0, d1 = 1.0, 0.0
        for p in slots:
            d1 = d1 * (1 - p) + d0 * p
            d0 = d0 * (1 - p)
        p0 += w * d0
        p1 += w * d1
    return 1 - p0, max(0.0, 1 - p0 - p1)


def hrr_increment(p_h: float, p_hr: float, p_ob: float,
                  conv: float) -> dict[int, float]:
    """Distribution of hits+runs+RBI produced by one plate appearance."""
    p_hr = min(p_hr, p_h)
    p_hit = max(0.0, p_h - p_hr)
    p_walk = max(0.0, p_ob - p_h)
    p_out = max(0.0, 1.0 - p_ob)

    d: dict[int, float] = {}

    def add(k, v):
        d[k] = d.get(k, 0.0) + v

    for rbi, w in HR_RBI_DIST:          # homer: 1 hit + 1 run + RBI
        add(2 + rbi, p_hr * w)
    for rbi, wr in ((1, HIT_RBI_P), (0, 1 - HIT_RBI_P)):
        for run, wn in ((1, conv), (0, 1 - conv)):
            add(1 + rbi + run, p_hit * wr * wn)
    cw = conv * WALK_SCORE_ADJ           # walk/HBP: only scores
    add(1, p_walk * cw)
    add(0, p_walk * (1 - cw))
    add(1, p_out * OUT_RBI_P)            # out that drives in a run
    add(0, p_out * (1 - OUT_RBI_P))
    return d


def hrr_distribution(inc_sp: dict, inc_bp: dict, pa_sp: float,
                     total_pa: float) -> list[float]:
    """P(total H+R+RBI == k) for k = 0..HRR_CAP, over a whole game."""
    n_lo = int(math.floor(total_pa))
    w_hi = total_pa - n_lo
    k_sp = int(round(pa_sp))
    out = [0.0] * (HRR_CAP + 1)

    for n, w in ((n_lo, 1 - w_hi), (n_lo + 1, w_hi)):
        if w <= 0:
            continue
        state = [0.0] * (HRR_CAP + 1)
        state[0] = 1.0
        for i in range(max(0, n)):
            inc = inc_sp if i < k_sp else inc_bp
            nxt = [0.0] * (HRR_CAP + 1)
            for s, ps in enumerate(state):
                if ps <= 0.0:
                    continue
                for step, pi in inc.items():
                    nxt[min(HRR_CAP, s + step)] += ps * pi
            state = nxt
        for s in range(HRR_CAP + 1):
            out[s] += w * state[s]
    return out


def at_least(dist: list[float], k: int) -> float:
    return sum(dist[k:]) if k <= HRR_CAP else 0.0


# ---------------------------------------------------------------------------
# Data pulls
# ---------------------------------------------------------------------------


def get_schedule(day: str) -> list[dict]:
    data = api_get("schedule", {
        "sportId": 1, "date": day,
        "hydrate": "probablePitcher,lineups,team,person,venue,linescore",
    })
    games = []
    for d in data.get("dates", []):
        games.extend(d.get("games", []))
    return games


def league_rates(season: int) -> dict:
    data = api_get("teams/stats", {
        "stats": "season", "group": "hitting", "season": season, "sportId": 1,
    })
    tot = {k: 0.0 for k in ("h", "pa", "ob", "hr", "rbi", "r")}
    for block in data.get("stats", []):
        for sp in block.get("splits", []):
            st = sp.get("stat", {})
            tot["h"] += _f(st, "hits")
            tot["pa"] += batter_pa(st)
            tot["ob"] += on_base(st)
            tot["hr"] += _f(st, "homeRuns")
            tot["rbi"] += _f(st, "rbi")
            tot["r"] += _f(st, "runs")
    if tot["pa"] < 5000:
        return dict(LEAGUE_FALLBACK)
    non_hr_ob = tot["ob"] - tot["hr"]
    return {
        "h_pa": tot["h"] / tot["pa"],
        "ob_pa": tot["ob"] / tot["pa"],
        "hr_pa": tot["hr"] / tot["pa"],
        "rbi_pa": tot["rbi"] / tot["pa"],
        "conv": ((tot["r"] - tot["hr"]) / non_hr_ob) if non_hr_ob > 0
                else LEAGUE_FALLBACK["conv"],
    }


def person(pid: int) -> dict:
    people = api_get(f"people/{pid}").get("people", [])
    return people[0] if people else {}


def _season_stat(pid: int, season: int, group: str) -> dict:
    d = api_get(f"people/{pid}/stats",
                {"stats": "season", "group": group, "season": season,
                 "gameType": "R"})
    for block in d.get("stats", []):
        for sp in block.get("splits", []):
            return sp.get("stat", {})
    return {}


def _platoon(pid: int, season: int, group: str) -> dict:
    d = api_get(f"people/{pid}/stats", {
        "stats": "statSplits", "group": group, "season": season,
        "gameType": "R", "sitCodes": "vl,vr",
    })
    out = {}
    for block in d.get("stats", []):
        for sp in block.get("splits", []):
            code = (sp.get("split") or {}).get("code")
            if code:
                out[code] = sp.get("stat", {})
    return out


def hitting_recent(pid: int, season: int, day: str, lookback: int = 21) -> dict:
    end = datetime.strptime(day, "%Y-%m-%d").date()
    start = end - timedelta(days=lookback)
    d = api_get(f"people/{pid}/stats", {
        "stats": "byDateRange", "group": "hitting", "season": season,
        "gameType": "R", "startDate": start.isoformat(), "endDate": end.isoformat(),
    })
    for block in d.get("stats", []):
        for sp in block.get("splits", []):
            return sp.get("stat", {})
    return {}


def team_pitching(team_id: int, season: int) -> dict:
    d = api_get(f"teams/{team_id}/stats",
                {"stats": "season", "group": "pitching", "season": season,
                 "gameType": "R"})
    for block in d.get("stats", []):
        for sp in block.get("splits", []):
            return sp.get("stat", {})
    return {}


def last_lineup(team_id: int, day: str, season: int) -> list[int]:
    end = datetime.strptime(day, "%Y-%m-%d").date() - timedelta(days=1)
    start = end - timedelta(days=7)
    d = api_get("schedule", {
        "sportId": 1, "teamId": team_id,
        "startDate": start.isoformat(), "endDate": end.isoformat(),
    })
    games = []
    for dd in d.get("dates", []):
        games.extend(dd.get("games", []))
    games = [g for g in games
             if g.get("status", {}).get("abstractGameState") == "Final"]
    games.sort(key=lambda g: g.get("gameDate", ""), reverse=True)
    for g in games:
        box = api_get(f"game/{g['gamePk']}/boxscore")
        for side in ("home", "away"):
            t = box.get("teams", {}).get(side, {})
            if t.get("team", {}).get("id") == team_id:
                order = [int(str(x).lstrip("ID")) for x in t.get("battingOrder", [])]
                if len(order) >= 9:
                    return order[:9]
    return []


# ---------------------------------------------------------------------------
# Rate profiles
# ---------------------------------------------------------------------------


def batter_profile(pid: int, season: int, day: str, pitcher_hand: str,
                   lg: dict, use_form: bool) -> dict:
    """Per-PA rates for a batter facing a given pitcher handedness."""
    st = _season_stat(pid, season, "hitting")
    pa = batter_pa(st)

    base_h = shrink(_f(st, "hits"), pa, lg["h_pa"], K_PRIOR_PA)
    base_ob = shrink(on_base(st), pa, lg["ob_pa"], K_PRIOR_PA)
    base_hr = shrink(_f(st, "homeRuns"), pa, lg["hr_pa"], K_PRIOR_PA)
    rbi = shrink(_f(st, "rbi"), pa, lg["rbi_pa"], K_PRIOR_PA)

    non_hr_ob = on_base(st) - _f(st, "homeRuns")
    conv = shrink(_f(st, "runs") - _f(st, "homeRuns"), non_hr_ob, lg["conv"], 60.0)
    conv = min(max(conv, 0.15), 0.50)

    code = "vl" if pitcher_hand == "L" else "vr"
    sp = _platoon(pid, season, "hitting").get(code, {})
    sp_pa = batter_pa(sp)
    h = shrink(_f(sp, "hits"), sp_pa, base_h, K_BATTER_SPLIT)
    ob = shrink(on_base(sp), sp_pa, base_ob, K_BATTER_SPLIT)
    hr = shrink(_f(sp, "homeRuns"), sp_pa, base_hr, K_BATTER_SPLIT)

    form_pa = 0.0
    if use_form:
        rec = hitting_recent(pid, season, day)
        form_pa = batter_pa(rec)
        h = shrink(_f(rec, "hits"), form_pa, h, K_BATTER_FORM)
        ob = shrink(on_base(rec), form_pa, ob, K_BATTER_FORM)

    return {"h": h, "ob": ob, "hr": hr, "rbi": rbi, "conv": conv,
            "season_h": base_h, "season_pa": pa, "split_pa": sp_pa,
            "form_pa": form_pa}


def pitcher_profile(pid: int, season: int, bat_side: str, lg: dict) -> dict:
    st = _season_stat(pid, season, "pitching")
    bf = pitcher_bf(st)

    base_h = shrink(_f(st, "hits"), bf, lg["h_pa"], K_PRIOR_PA)
    base_ob = shrink(on_base(st), bf, lg["ob_pa"], K_PRIOR_PA)
    base_hr = shrink(_f(st, "homeRuns"), bf, lg["hr_pa"], K_PRIOR_PA)

    code = "vl" if bat_side == "L" else "vr"
    sp = _platoon(pid, season, "pitching").get(code, {})
    sp_bf = pitcher_bf(sp)
    h = shrink(_f(sp, "hits"), sp_bf, base_h, K_PITCHER_SPLIT)
    ob = shrink(on_base(sp), sp_bf, base_ob, K_PITCHER_SPLIT)
    hr = shrink(_f(sp, "homeRuns"), sp_bf, base_hr, K_PITCHER_SPLIT)

    gs = _f(st, "gamesStarted")
    exp_bf = (bf / gs) if gs >= 3 and bf else DEFAULT_STARTER_BF
    exp_bf = min(max(exp_bf, 12.0), 30.0)
    return {"h": h, "ob": ob, "hr": hr, "exp_bf": exp_bf,
            "era": st.get("era", "-"), "whip": st.get("whip", "-")}


def bullpen_profile(team_id: int, season: int, lg: dict) -> dict:
    st = team_pitching(team_id, season)
    bf = pitcher_bf(st)
    if bf <= 0:
        return {"h": lg["h_pa"], "ob": lg["ob_pa"], "hr": lg["hr_pa"]}
    return {
        "h": (_f(st, "hits") / bf) * BULLPEN_ADJ,
        "ob": (on_base(st) / bf) * BULLPEN_ADJ,
        "hr": (_f(st, "homeRuns") / bf) * BULLPEN_ADJ,
    }


# ---------------------------------------------------------------------------
# Game processing
# ---------------------------------------------------------------------------


def extract_lineup(game: dict, side: str) -> list[int]:
    key = "homePlayers" if side == "home" else "awayPlayers"
    players = (game.get("lineups") or {}).get(key) or []
    return [p["id"] for p in players if p.get("id")][:9]


def process_game(game: dict, season: int, day: str, lg: dict,
                 use_form: bool, use_park: bool, projected: bool) -> list[dict]:
    teams = game.get("teams", {})
    venue = (game.get("venue") or {}).get("name", "")
    pf = PARK_FACTORS.get(venue, 1.0) if use_park else 1.0
    gtime = game.get("gameDate", "")
    rows: list[dict] = []

    for side, opp_side in (("away", "home"), ("home", "away")):
        t = teams.get(side, {})
        team = t.get("team", {})
        opp_team = teams.get(opp_side, {}).get("team", {})
        sp = teams.get(opp_side, {}).get("probablePitcher") or {}
        if not sp.get("id"):
            continue

        lineup = extract_lineup(game, side)
        source = "POSTED"
        if len(lineup) < 9:
            if not projected:
                continue
            lineup = last_lineup(team["id"], day, season)
            source = "PROJ"
            if len(lineup) < 9:
                continue

        sp_hand = ((sp.get("pitchHand") or {}).get("code") or "R").upper()
        if sp_hand not in ("L", "R"):
            sp_hand = "R"
        bp = bullpen_profile(opp_team.get("id"), season, lg)

        with ThreadPoolExecutor(max_workers=6) as pool:
            infos = list(pool.map(person, lineup))

        for slot, (pid, info) in enumerate(zip(lineup, infos), start=1):
            bat_side = ((info.get("batSide") or {}).get("code") or "R").upper()
            eff_side = bat_side
            if bat_side == "S":
                eff_side = "L" if sp_hand == "R" else "R"

            b = batter_profile(pid, season, day, sp_hand, lg, use_form)
            p = pitcher_profile(sp["id"], season, eff_side, lg)

            def mu(key, lg_key, cap):
                v = log5(b[key], p[key], lg[lg_key]) * pf
                return min(max(v, 0.001), cap)

            def mu_bp(key, lg_key, cap):
                v = log5(b[key], bp[key], lg[lg_key]) * pf
                return min(max(v, 0.001), cap)

            h_sp, h_bp = mu("h", "h_pa", 0.75), mu_bp("h", "h_pa", 0.75)
            ob_sp, ob_bp = mu("ob", "ob_pa", 0.90), mu_bp("ob", "ob_pa", 0.90)
            hr_sp, hr_bp = mu("hr", "hr_pa", 0.25), mu_bp("hr", "hr_pa", 0.25)

            total_pa = PA_BY_SLOT[slot - 1] * (HOME_PA_FACTOR if side == "home" else 1.0)
            pa_sp = min(pa_vs_starter(slot, p["exp_bf"]), total_pa)
            pa_bp = max(0.0, total_pa - pa_sp)

            p_1h, p_2h = hit_distribution(h_sp, h_bp, pa_sp, total_pa)
            xh = pa_sp * h_sp + pa_bp * h_bp

            exp_ob = pa_sp * ob_sp + pa_bp * ob_bp
            exp_hr = pa_sp * hr_sp + pa_bp * hr_bp
            exp_runs = exp_hr + max(0.0, exp_ob - exp_hr) * b["conv"]
            p_run = 1 - math.exp(-exp_runs / RUN_DISPERSION)

            # RBI: season rate, nudged by how good this particular matchup is.
            mult = min(max(h_sp / b["season_h"], 0.70), 1.40) if b["season_h"] else 1.0
            exp_rbi = total_pa * b["rbi"] * mult
            p_rbi = 1 - math.exp(-exp_rbi / RBI_DISPERSION)

            hrr = hrr_distribution(
                hrr_increment(h_sp, hr_sp, ob_sp, b["conv"]),
                hrr_increment(h_bp, hr_bp, ob_bp, b["conv"]),
                pa_sp, total_pa)
            exp_hrr = sum(k * p for k, p in enumerate(hrr))

            rows.append({
                "date": day,
                "game": f"{teams['away']['team'].get('abbreviation', '?')}@"
                        f"{teams['home']['team'].get('abbreviation', '?')}",
                "game_time_utc": gtime,
                "venue": venue,
                "team": team.get("abbreviation", team.get("name", "?")),
                "slot": slot,
                "batter": info.get("fullName", str(pid)),
                "batter_id": pid,
                "bats": bat_side,
                "opp_sp": sp.get("fullName", "?"),
                "sp_hand": sp_hand,
                "sp_era": p["era"],
                "sp_whip": p["whip"],
                "lineup_source": source,
                "pa": round(total_pa, 2),
                "pa_vs_sp": round(pa_sp, 2),
                "pa_vs_bp": round(pa_bp, 2),
                "season_pa": int(b["season_pa"]),
                "split_pa": int(b["split_pa"]),
                "hit_prob": round(p_1h, 4),
                "prob_2h": round(p_2h, 4),
                "prob_run": round(p_run, 4),
                "prob_rbi": round(p_rbi, 4),
                "prob_hrr1": round(at_least(hrr, 1), 4),
                "prob_hrr2": round(at_least(hrr, 2), 4),
                "prob_hrr3": round(at_least(hrr, 3), 4),
                "exp_hits": round(xh, 3),
                "exp_runs": round(exp_runs, 3),
                "exp_rbi": round(exp_rbi, 3),
                "exp_hrr": round(exp_hrr, 3),
            })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

MARKETS = [("hit_prob", "1+ H"),
           ("prob_hrr2", "2+ H+R+RBI"), ("prob_hrr3", "3+ H+R+RBI"),
           ("prob_2h", "2+ H"), ("prob_run", "Run"), ("prob_rbi", "RBI")]

BOARD_KEYS = {"hit": "hit_prob", "hrr2": "prob_hrr2", "hrr3": "prob_hrr3",
              "h2": "prob_2h", "run": "prob_run", "rbi": "prob_rbi"}


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "No rows. (Lineups may not be posted yet — try --projected.)"
    hdr = ["GAME", "TIME", "TM", "#", "BATTER", "B", "OPP SP", "T", "PA",
           "1+H", "HRR2", "HRR3", "2+H", "RUN", "RBI", "SRC"]
    body = [[
        r["game"], local_time(r.get("game_time_utc", "")), r["team"],
        str(r["slot"]), r["batter"][:20], r["bats"], r["opp_sp"][:18],
        r["sp_hand"], f"{r['pa']:.1f}",
        f"{r['hit_prob'] * 100:.1f}",
        f"{r['prob_hrr2'] * 100:.1f}", f"{r['prob_hrr3'] * 100:.1f}",
        f"{r['prob_2h'] * 100:.1f}", f"{r['prob_run'] * 100:.1f}",
        f"{r['prob_rbi'] * 100:.1f}", r["lineup_source"],
    ] for r in rows]
    widths = [max(len(hdr[i]), max(len(b[i]) for b in body)) for i in range(len(hdr))]
    out = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(hdr)),
           "  ".join("-" * w for w in widths)]
    for b in body:
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(b)))
    return "\n".join(out)


def render_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _prob_cell(v: float, good: float) -> str:
    hue = 120 if v >= good else (45 if v >= good * 0.85 else 0)
    return (f"<td style='color:hsl({hue},70%,35%);font-weight:600'>"
            f"{v * 100:.1f}%</td>")


def _row_html(r: dict) -> str:
    return (f"<tr><td>{r['game']}</td>"
            f"<td class='t'>{local_time(r.get('game_time_utc', ''))}</td>"
            f"<td>{r['team']}</td><td>{r['slot']}</td>"
            f"<td class='n'>{r['batter']}</td><td>{r['bats']}</td>"
            f"<td>{r['opp_sp']} ({r['sp_hand']})</td>"
            f"<td>{r['pa']:.1f}</td>"
            + _prob_cell(r["hit_prob"], 0.70)
            + _prob_cell(r["prob_hrr2"], 0.55)
            + _prob_cell(r["prob_hrr3"], 0.30)
            + _prob_cell(r["prob_2h"], 0.28)
            + _prob_cell(r["prob_run"], 0.35)
            + _prob_cell(r["prob_rbi"], 0.35)
            + f"<td>{r['lineup_source']}</td></tr>")


def _table(rows: list[dict]) -> str:
    head = ("<table><tr><th>Game</th><th>Time</th><th>Tm</th><th>#</th>"
            "<th>Batter</th><th>B</th><th>Opposing SP</th><th>PA</th>"
            "<th>1+ H</th><th>2+ HRR</th><th>3+ HRR</th>"
            "<th>2+ H</th><th>Run</th><th>RBI</th><th>Lineup</th></tr>")
    return head + "\n".join(_row_html(r) for r in rows) + "</table>"


def render_html(rows: list[dict], day: str, top_n: int = 20,
                board_key: str = "hit_prob", board_label: str = "1+ hit") -> str:
    main = sorted([r for r in rows if r.get("_main", True)],
                  key=lambda r: -r.get(board_key, 0))

    extras = []
    for key, label in MARKETS:
        if key == board_key:
            continue
        best = sorted(rows, key=lambda r: -r[key])[:top_n]
        if best:
            extras.append(f"<h2>Top {len(best)} &mdash; {label}</h2>{_table(best)}")

    return f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Batter props — {day}</title>
<style>
 body{{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      margin:20px;color:#1a1a1a;background:#fafaf8}}
 h1{{font-size:19px;margin:0 0 2px}}
 h2{{font-size:13px;margin:26px 0 8px;text-transform:uppercase;
     letter-spacing:.06em;color:#555}}
 p.sub{{color:#666;margin:0 0 16px;font-size:12px}}
 a{{color:#1d4ed8}}
 table{{border-collapse:collapse;width:100%;background:#fff;
        box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:6px;overflow:hidden}}
 th{{background:#1f2937;color:#fff;text-align:left;padding:8px 10px;
     font-size:11px;letter-spacing:.06em;text-transform:uppercase}}
 td{{padding:7px 10px;border-top:1px solid #eee}}
 tr:hover td{{background:#f5f7fa}} td.n{{font-weight:600}}
 td.t{{color:#555;white-space:nowrap}}
</style>
<h1>Batter props — {day}</h1>
<p class="sub">{len(main)} batters on the main board &middot; updated {now_local_str()}
 &middot; <a href="results.html">track record &rarr;</a>
 &middot; model estimate only, not betting advice</p>
<h2>{board_label} board</h2>
{_table(main)}
{"".join(extras)}
"""


# ---------------------------------------------------------------------------
# Self test (offline)
# ---------------------------------------------------------------------------


def selftest() -> int:
    ok = True

    def check(label, got, want, tol=1e-6):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {label}: {got:.5f} (want ~{want})")

    def assert_true(label, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {label}{detail}")

    print("log5:")
    check("average vs average = league", log5(0.22, 0.22, 0.22), 0.22)
    check("good hitter vs avg arm", log5(0.26, 0.22, 0.22), 0.26, 1e-9)
    assert_true("good hitter vs bad arm > .260", log5(0.26, 0.26, 0.22) > 0.26)

    print("shrinkage:")
    check("zero sample -> prior", shrink(0, 0, 0.22, 200), 0.22)
    check("large sample dominates", shrink(300, 1000, 0.22, 200), 344 / 1200, 1e-9)

    print("PA vs starter:")
    check("slot 1, 27 BF", pa_vs_starter(1, 27), 3.0)
    check("slot 3, 20 BF", pa_vs_starter(3, 20), 2.0)
    check("slots sum to BF", sum(pa_vs_starter(s, 20) for s in range(1, 10)),
          20.0, 1e-9)
    check("slot 5, 22.5 BF interpolates", pa_vs_starter(5, 22.5), 2.5)

    print("hit distribution:")
    p1, p2 = hit_distribution(0.25, 0.25, 4, 4)
    check("4 PA at .250, 1+ hit", p1, 1 - 0.75 ** 4, 1e-9)
    check("4 PA at .250, 2+ hits", p2,
          1 - 0.75 ** 4 - 4 * 0.25 * 0.75 ** 3, 1e-9)
    p1b, p2b = hit_distribution(0.25, 0.25, 3, 4.5)
    naive = 1 - 0.75 ** 4.5
    assert_true("fractional PA below naive exponent",
                p1b < naive, f": {p1b:.4f} < {naive:.4f}")
    assert_true("2+ below 1+", p2b < p1b, f": {p2b:.4f} < {p1b:.4f}")
    _, p2z = hit_distribution(0.25, 0.25, 1, 1)
    check("1 PA can't produce 2 hits", p2z, 0.0, 1e-12)
    p1m, _ = hit_distribution(0.30, 0.20, 3, 4.5)
    assert_true("mixed rates land between", 0.55 < p1m < 0.80, f": {p1m:.4f}")

    print("run / rbi shape:")
    exp_runs = 0.14 + (1.37 - 0.14) * 0.29
    pr = 1 - math.exp(-exp_runs / RUN_DISPERSION)
    assert_true("league-average run prob in 25-40%", 0.25 < pr < 0.40,
                f": {pr * 100:.1f}%")
    prbi = 1 - math.exp(-(4.4 * 0.115) / RBI_DISPERSION)
    assert_true("league-average RBI prob in 22-38%", 0.22 < prbi < 0.38,
                f": {prbi * 100:.1f}%")

    print("hits + runs + RBI:")
    inc = hrr_increment(0.2215, 0.032, 0.312, 0.29)
    check("increment is a distribution", sum(inc.values()), 1.0, 1e-9)
    mean_inc = sum(k * v for k, v in inc.items())
    assert_true("per-PA mean 0.40-0.55", 0.40 < mean_inc < 0.55, f": {mean_inc:.3f}")
    d = hrr_distribution(inc, inc, 3, 4.4)
    check("game distribution sums to 1", sum(d), 1.0, 1e-9)
    mean_g = sum(k * v for k, v in enumerate(d))
    assert_true("league-average game total 1.7-2.4", 1.7 < mean_g < 2.4,
                f": {mean_g:.2f}")
    p1, p2, p3 = at_least(d, 1), at_least(d, 2), at_least(d, 3)
    assert_true("monotonic 1+ > 2+ > 3+", p1 > p2 > p3,
                f": {p1 * 100:.1f} > {p2 * 100:.1f} > {p3 * 100:.1f}")
    assert_true("league-average 2+ in 45-62%", 0.45 < p2 < 0.62,
                f": {p2 * 100:.1f}%")
    good = hrr_distribution(hrr_increment(0.28, 0.05, 0.37, 0.32),
                            hrr_increment(0.28, 0.05, 0.37, 0.32), 3, 4.6)
    assert_true("better hitter has higher 2+",
                at_least(good, 2) > p2,
                f": {at_least(good, 2) * 100:.1f}% > {p2 * 100:.1f}%")
    single = hrr_distribution(inc, inc, 1, 1)
    assert_true("one PA rarely reaches 3+", at_least(single, 3) < 0.06,
                f": {at_least(single, 3) * 100:.2f}%")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Daily MLB batter prop probabilities.")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--team", default=None, help="filter to a team name/abbrev")
    ap.add_argument("--sort", choices=["prob", "game"], default="prob")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--board-market", choices=list(BOARD_KEYS), default="hit",
                    help="which market drives the main board and --min-prob")
    ap.add_argument("--min-prob", type=float, default=0.0,
                    help="probability floor for the main board, e.g. 0.70")
    ap.add_argument("--top-per-market", type=int, default=20,
                    help="also keep the top N for 2+H, run and RBI (0 to skip)")
    ap.add_argument("--format", choices=["table", "csv", "json", "html"],
                    default="table")
    ap.add_argument("--out", default=None)
    ap.add_argument("--projected", action="store_true")
    ap.add_argument("--no-form", action="store_true")
    ap.add_argument("--no-park", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    day = args.date or now_local_str()[:10]
    season = args.season or int(day[:4])

    games = get_schedule(day)
    if args.team:
        q = args.team.lower()
        games = [g for g in games if any(
            q in str(g["teams"][s]["team"].get(k, "")).lower()
            for s in ("home", "away") for k in ("name", "abbreviation", "teamName"))]
    if not games:
        print(f"No games found for {day}.", file=sys.stderr)
        return 1

    print(f"Scanning {len(games)} game(s) on {day}...", file=sys.stderr)
    lg = league_rates(season)
    print(f"League baseline: {lg['h_pa']:.4f} H/PA, {lg['ob_pa']:.4f} OB/PA",
          file=sys.stderr)

    rows: list[dict] = []
    for g in games:
        away = g["teams"]["away"]["team"].get("abbreviation", "?")
        home = g["teams"]["home"]["team"].get("abbreviation", "?")
        if g.get("status", {}).get("abstractGameState") == "Final":
            print(f"  {away}@{home}: final, skipping", file=sys.stderr)
            continue
        try:
            got = process_game(g, season, day, lg, not args.no_form,
                               not args.no_park, args.projected)
            rows.extend(got)
            print(f"  {away}@{home}: {len(got)} batters", file=sys.stderr)
        except Exception as exc:
            print(f"  {away}@{home}: error — {exc}", file=sys.stderr)

    board_key = BOARD_KEYS[args.board_market]
    for r in rows:
        r["_main"] = r.get(board_key, 0) >= args.min_prob

    if args.min_prob > 0 or args.top_per_market:
        keep = {i for i, r in enumerate(rows) if r["_main"]}
        if args.top_per_market:
            for key, _ in MARKETS:
                if key == board_key:
                    continue
                order = sorted(range(len(rows)), key=lambda i: -rows[i][key])
                keep.update(order[:args.top_per_market])
        rows = [rows[i] for i in sorted(keep)]

    if args.sort == "prob":
        rows.sort(key=lambda r: r.get(board_key, 0), reverse=True)
    else:
        rows.sort(key=lambda r: (r["game"], r["team"], r["slot"]))
    if args.top:
        rows = rows[:args.top]

    if args.format == "table":
        text = render_table([r for r in rows if r.get("_main", True)])
    elif args.format == "csv":
        text = render_csv(rows)
    elif args.format == "json":
        text = json.dumps(rows, indent=2)
    else:
        label = dict(MARKETS).get(board_key, "1+ H")
        text = render_html(rows, day, board_key=board_key, board_label=label)

    if args.out:
        with open(os.path.expanduser(args.out), "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

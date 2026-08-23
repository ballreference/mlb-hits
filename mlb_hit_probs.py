#!/usr/bin/env python3
"""
mlb_hit_probs.py — daily batter hit-probability scanner.

Pulls today's MLB slate from the free MLB Stats API, matches every posted
lineup against that day's probable/announced starting pitcher, and estimates
each batter's probability of recording at least one hit.

No API key. No third-party packages (stdlib only).

Usage:
    python3 mlb_hit_probs.py
    python3 mlb_hit_probs.py --date 2026-08-23 --sort prob --top 40
    python3 mlb_hit_probs.py --team "Tigers" --format table
    python3 mlb_hit_probs.py --format csv --out today.csv
    python3 mlb_hit_probs.py --format html --out report.html
    python3 mlb_hit_probs.py --projected      # use last-game lineups if not posted
    python3 mlb_hit_probs.py --selftest       # offline math check, no network

Cron (runs at 10am and again at 3pm when lineups are posted):
    0 10,15 * * * cd ~/mlb && /usr/bin/python3 mlb_hit_probs.py \
        --projected --format html --out ~/mlb/today.html
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
from datetime import date as _date, datetime, timedelta

API = "https://statsapi.mlb.com/api/v1"
UA = "mlb-hit-probs/1.0 (personal use)"

# ---------------------------------------------------------------------------
# Model constants — tune these.
# ---------------------------------------------------------------------------

# Fallback league-average hits per plate appearance if the live pull fails.
LEAGUE_H_PER_PA_FALLBACK = 0.2215

# Regression (shrinkage) constants. A split with few PA gets pulled toward the
# player's overall rate. Higher number = trust the split less.
K_BATTER_SPLIT = 200.0   # PA of league/overall prior added to a batter's platoon split
K_PITCHER_SPLIT = 250.0  # BF of prior added to a pitcher's platoon split
K_BATTER_FORM = 320.0    # PA of prior added to recent-form rate (heavy regression)
K_PRIOR_PA = 150.0       # PA of league prior added to a batter's own season line

# Expected plate appearances by lineup slot (1-9), full 9-inning game.
PA_BY_SLOT = [4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95, 3.85]
HOME_PA_FACTOR = 0.975   # home team sometimes doesn't bat in the 9th

# Bullpens allow slightly fewer hits per batter than a team's overall staff.
BULLPEN_ADJ = 0.98
DEFAULT_STARTER_BF = 22.0  # if a starter has no track record

# Rough hit park factors. EDIT THESE to taste — they are deliberately mild.
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
# HTTP layer
# ---------------------------------------------------------------------------

_cache: dict[str, dict] = {}


def api_get(path: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET a Stats API endpoint, with in-memory caching and retries."""
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
        except Exception as exc:  # network hiccup, 5xx, timeout
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


def shrink(events: float, opportunities: float, prior_rate: float, k: float) -> float:
    """Regress an observed rate toward a prior using k prior opportunities."""
    if opportunities <= 0:
        return prior_rate
    return (events + prior_rate * k) / (opportunities + k)


def log5(batter_rate: float, pitcher_rate: float, league_rate: float) -> float:
    """Odds-ratio matchup estimate (Bill James' log5)."""
    b, p, l = batter_rate, pitcher_rate, league_rate
    l = min(max(l, 1e-4), 0.9999)
    b = min(max(b, 1e-4), 0.9999)
    p = min(max(p, 1e-4), 0.9999)
    num = (b * p) / l
    den = num + ((1 - b) * (1 - p)) / (1 - l)
    return num / den if den > 0 else b


def pa_vs_starter(slot: int, starter_bf: float) -> float:
    """Expected PA that lineup slot `slot` (1-9) gets against the starter."""
    def turns(n: int) -> int:
        if n < slot:
            return 0
        return (n - slot) // 9 + 1
    lo = int(math.floor(starter_bf))
    frac = starter_bf - lo
    return turns(lo) * (1 - frac) + turns(lo + 1) * frac


# ---------------------------------------------------------------------------
# Data pulls
# ---------------------------------------------------------------------------

def get_schedule(day: str) -> list[dict]:
    data = api_get("schedule", {
        "sportId": 1,
        "date": day,
        "hydrate": "probablePitcher,lineups,team,person,venue,linescore",
    })
    games = []
    for d in data.get("dates", []):
        games.extend(d.get("games", []))
    return games


def league_h_per_pa(season: int) -> float:
    data = api_get("teams/stats", {
        "stats": "season", "group": "hitting", "season": season, "sportId": 1,
    })
    h = pa = 0.0
    for block in data.get("stats", []):
        for sp in block.get("splits", []):
            st = sp.get("stat", {})
            h += _f(st, "hits")
            pa += batter_pa(st)
    if pa > 5000:
        return h / pa
    return LEAGUE_H_PER_PA_FALLBACK


def person(pid: int) -> dict:
    d = api_get(f"people/{pid}")
    people = d.get("people", [])
    return people[0] if people else {}


def hitting_season(pid: int, season: int) -> dict:
    d = api_get(f"people/{pid}/stats",
                {"stats": "season", "group": "hitting", "season": season, "gameType": "R"})
    for block in d.get("stats", []):
        for sp in block.get("splits", []):
            return sp.get("stat", {})
    return {}


def hitting_splits(pid: int, season: int) -> dict:
    """Returns {'vl': stat, 'vr': stat} — vs left / vs right handed pitching."""
    d = api_get(f"people/{pid}/stats", {
        "stats": "statSplits", "group": "hitting", "season": season,
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


def pitching_season(pid: int, season: int) -> dict:
    d = api_get(f"people/{pid}/stats",
                {"stats": "season", "group": "pitching", "season": season, "gameType": "R"})
    for block in d.get("stats", []):
        for sp in block.get("splits", []):
            return sp.get("stat", {})
    return {}


def pitching_splits(pid: int, season: int) -> dict:
    d = api_get(f"people/{pid}/stats", {
        "stats": "statSplits", "group": "pitching", "season": season,
        "gameType": "R", "sitCodes": "vl,vr",
    })
    out = {}
    for block in d.get("stats", []):
        for sp in block.get("splits", []):
            code = (sp.get("split") or {}).get("code")
            if code:
                out[code] = sp.get("stat", {})
    return out


def team_pitching(team_id: int, season: int) -> dict:
    d = api_get(f"teams/{team_id}/stats",
                {"stats": "season", "group": "pitching", "season": season, "gameType": "R"})
    for block in d.get("stats", []):
        for sp in block.get("splits", []):
            return sp.get("stat", {})
    return {}


def last_lineup(team_id: int, day: str, season: int) -> list[int]:
    """Fallback: batting order from this team's most recent completed game."""
    end = datetime.strptime(day, "%Y-%m-%d").date() - timedelta(days=1)
    start = end - timedelta(days=7)
    d = api_get("schedule", {
        "sportId": 1, "teamId": team_id,
        "startDate": start.isoformat(), "endDate": end.isoformat(),
    })
    games = []
    for dd in d.get("dates", []):
        games.extend(dd.get("games", []))
    games = [g for g in games if (g.get("status", {}).get("abstractGameState") == "Final")]
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
# Core rating
# ---------------------------------------------------------------------------

def batter_rate(pid: int, season: int, day: str, pitcher_hand: str,
                lg: float, use_form: bool) -> tuple[float, dict]:
    """Hit-per-PA rate for a batter vs a given pitcher handedness."""
    season_stat = hitting_season(pid, season)
    s_h, s_pa = _f(season_stat, "hits"), batter_pa(season_stat)
    base = shrink(s_h, s_pa, lg, K_PRIOR_PA)

    code = "vl" if pitcher_hand == "L" else "vr"
    sp = hitting_splits(pid, season).get(code, {})
    sp_h, sp_pa = _f(sp, "hits"), batter_pa(sp)
    rate = shrink(sp_h, sp_pa, base, K_BATTER_SPLIT)

    form_pa = 0.0
    if use_form:
        rec = hitting_recent(pid, season, day)
        r_h, form_pa = _f(rec, "hits"), batter_pa(rec)
        rate = shrink(r_h, form_pa, rate, K_BATTER_FORM)

    meta = {"season_pa": s_pa, "season_avg": _f(season_stat, "avg", 0) or
            (s_h / _f(season_stat, "atBats", 1) if _f(season_stat, "atBats") else 0),
            "split_pa": sp_pa, "form_pa": form_pa}
    return rate, meta


def pitcher_rate(pid: int, season: int, bat_side: str, lg: float) -> tuple[float, float, dict]:
    """Returns (hits-allowed-per-BF vs this batter side, expected BF per start, meta)."""
    st = pitching_season(pid, season)
    h, bf = _f(st, "hits"), pitcher_bf(st)
    base = shrink(h, bf, lg, K_PRIOR_PA)

    code = "vl" if bat_side == "L" else "vr"
    sp = pitching_splits(pid, season).get(code, {})
    sp_h, sp_bf = _f(sp, "hits"), pitcher_bf(sp)
    rate = shrink(sp_h, sp_bf, base, K_PITCHER_SPLIT)

    gs = _f(st, "gamesStarted")
    exp_bf = (bf / gs) if gs >= 3 and bf else DEFAULT_STARTER_BF
    exp_bf = min(max(exp_bf, 12.0), 30.0)
    return rate, exp_bf, {"bf": bf, "gs": gs, "era": st.get("era", "-"),
                          "whip": st.get("whip", "-")}


def bullpen_rate(team_id: int, season: int, lg: float) -> float:
    st = team_pitching(team_id, season)
    h, bf = _f(st, "hits"), pitcher_bf(st)
    if bf <= 0:
        return lg
    return (h / bf) * BULLPEN_ADJ


# ---------------------------------------------------------------------------
# Game processing
# ---------------------------------------------------------------------------

def extract_lineup(game: dict, side: str) -> list[int]:
    key = "homePlayers" if side == "home" else "awayPlayers"
    players = (game.get("lineups") or {}).get(key) or []
    return [p["id"] for p in players if p.get("id")][:9]


def process_game(game: dict, season: int, day: str, lg: float,
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
        bp = bullpen_rate(opp_team.get("id"), season, lg)

        # Look up all nine batters in parallel.
        with ThreadPoolExecutor(max_workers=6) as pool:
            infos = list(pool.map(person, lineup))

        for slot, (pid, info) in enumerate(zip(lineup, infos), start=1):
            bat_side = ((info.get("batSide") or {}).get("code") or "R").upper()
            eff_side = bat_side
            if bat_side == "S":  # switch hitter takes the platoon advantage
                eff_side = "L" if sp_hand == "R" else "R"

            b_rate, b_meta = batter_rate(pid, season, day, sp_hand, lg, use_form)
            p_rate, exp_bf, p_meta = pitcher_rate(sp["id"], season, eff_side, lg)

            p_sp = log5(b_rate, p_rate, lg) * pf
            p_bp = log5(b_rate, bp, lg) * pf
            p_sp = min(max(p_sp, 0.01), 0.75)
            p_bp = min(max(p_bp, 0.01), 0.75)

            total_pa = PA_BY_SLOT[slot - 1] * (HOME_PA_FACTOR if side == "home" else 1.0)
            pa_sp = min(pa_vs_starter(slot, exp_bf), total_pa)
            pa_bp = max(0.0, total_pa - pa_sp)

            p_none = ((1 - p_sp) ** pa_sp) * ((1 - p_bp) ** pa_bp)
            hit_prob = 1 - p_none
            xh = pa_sp * p_sp + pa_bp * p_bp

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
                "sp_era": p_meta["era"],
                "sp_whip": p_meta["whip"],
                "lineup_source": source,
                "pa_vs_sp": round(pa_sp, 2),
                "pa_vs_bp": round(pa_bp, 2),
                "p_hit_per_pa_sp": round(p_sp, 4),
                "p_hit_per_pa_bp": round(p_bp, 4),
                "season_pa": int(b_meta["season_pa"]),
                "split_pa": int(b_meta["split_pa"]),
                "hit_prob": round(hit_prob, 4),
                "exp_hits": round(xh, 3),
            })
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def render_table(rows: list[dict]) -> str:
    if not rows:
        return "No rows. (Lineups may not be posted yet — try --projected.)"
    hdr = ["GAME", "TM", "#", "BATTER", "B", "OPP SP", "T", "PA", "HIT%", "xH", "SRC"]
    body = [[
        r["game"], r["team"], str(r["slot"]), r["batter"][:20], r["bats"],
        r["opp_sp"][:18], r["sp_hand"],
        f"{r['pa_vs_sp'] + r['pa_vs_bp']:.1f}",
        f"{r['hit_prob'] * 100:.1f}", f"{r['exp_hits']:.2f}", r["lineup_source"],
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


def render_html(rows: list[dict], day: str) -> str:
    def cell(r):
        pct = r["hit_prob"] * 100
        hue = 120 if pct >= 65 else (45 if pct >= 55 else 0)
        return (f"<tr><td>{r['game']}</td><td>{r['team']}</td><td>{r['slot']}</td>"
                f"<td class='n'>{r['batter']}</td><td>{r['bats']}</td>"
                f"<td>{r['opp_sp']} ({r['sp_hand']})</td>"
                f"<td>{r['pa_vs_sp'] + r['pa_vs_bp']:.1f}</td>"
                f"<td style='color:hsl({hue},70%,35%);font-weight:600'>{pct:.1f}%</td>"
                f"<td>{r['exp_hits']:.2f}</td><td>{r['lineup_source']}</td></tr>")
    body = "\n".join(cell(r) for r in rows)
    return f"""<!doctype html><meta charset="utf-8">
<title>Hit Probabilities — {day}</title>
<style>
 body{{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      margin:24px;color:#1a1a1a;background:#fafaf8}}
 h1{{font-size:19px;margin:0 0 4px}} p.sub{{color:#666;margin:0 0 18px;font-size:13px}}
 table{{border-collapse:collapse;width:100%;background:#fff;
        box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:6px;overflow:hidden}}
 th{{background:#1f2937;color:#fff;text-align:left;padding:8px 10px;
     font-size:11px;letter-spacing:.06em;text-transform:uppercase}}
 td{{padding:7px 10px;border-top:1px solid #eee}}
 tr:hover td{{background:#f5f7fa}} td.n{{font-weight:600}}
</style>
<h1>Batter hit probabilities — {day}</h1>
<p class="sub">{len(rows)} batters &middot; generated {datetime.now():%Y-%m-%d %H:%M}
 &middot; model estimate only, not betting advice</p>
<table><tr><th>Game</th><th>Tm</th><th>#</th><th>Batter</th><th>B</th>
<th>Opposing SP</th><th>PA</th><th>Hit %</th><th>xH</th><th>Lineup</th></tr>
{body}</table>"""


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

    print("log5:")
    check("average vs average = league", log5(0.22, 0.22, 0.22), 0.22)
    check("good hitter vs avg arm", log5(0.26, 0.22, 0.22), 0.26, 1e-9)
    g = log5(0.26, 0.26, 0.22)
    print(f"  [{'ok' if g > 0.26 else 'FAIL'}] good hitter vs bad arm > .260: {g:.5f}")
    ok &= g > 0.26
    b = log5(0.18, 0.18, 0.22)
    print(f"  [{'ok' if b < 0.18 else 'FAIL'}] bad hitter vs good arm < .180: {b:.5f}")
    ok &= b < 0.18

    print("shrinkage:")
    check("zero sample -> prior", shrink(0, 0, 0.22, 200), 0.22)
    check("large sample dominates prior", shrink(300, 1000, 0.22, 200), 344 / 1200, 1e-9)
    check("small sample stays near prior", shrink(10, 20, 0.22, 200), 54 / 220, 1e-9)

    print("PA vs starter:")
    check("slot 1, 27 BF", pa_vs_starter(1, 27), 3.0)
    check("slot 9, 27 BF", pa_vs_starter(9, 27), 3.0)
    check("slot 3, 20 BF", pa_vs_starter(3, 20), 2.0)
    total = sum(pa_vs_starter(s, 20) for s in range(1, 10))
    check("slots sum to BF", total, 20.0, 1e-9)
    check("slot 1, 22.5 BF", pa_vs_starter(1, 22.5), 3.0)
    check("slot 5, 22.5 BF interpolates", pa_vs_starter(5, 22.5), 2.5)
    check("slot 8 vs short start (14 BF)", pa_vs_starter(8, 14), 1.0)

    print("hit probability:")
    p = 1 - (1 - 0.25) ** 4
    check("4 PA at .250 -> 68.4%", p, 0.6836, 1e-3)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Daily MLB batter hit probabilities.")
    ap.add_argument("--date", default=_date.today().isoformat(), help="YYYY-MM-DD")
    ap.add_argument("--season", type=int, default=None, help="stat season (default: from date)")
    ap.add_argument("--team", default=None, help="filter to a team name/abbrev substring")
    ap.add_argument("--sort", choices=["prob", "game"], default="prob")
    ap.add_argument("--top", type=int, default=0, help="only show top N rows")
    ap.add_argument("--min-prob", type=float, default=0.0, help="e.g. 0.6 for 60%%+")
    ap.add_argument("--format", choices=["table", "csv", "json", "html"], default="table")
    ap.add_argument("--out", default=None, help="write output to a file")
    ap.add_argument("--projected", action="store_true",
                    help="fall back to last game's batting order if lineup unposted")
    ap.add_argument("--no-form", action="store_true", help="skip last-21-day form adjustment")
    ap.add_argument("--no-park", action="store_true", help="skip park factors")
    ap.add_argument("--selftest", action="store_true", help="offline math check")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    day = args.date
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
    lg = league_h_per_pa(season)
    print(f"League baseline: {lg:.4f} hits/PA", file=sys.stderr)

    rows: list[dict] = []
    for g in games:
        away = g["teams"]["away"]["team"].get("abbreviation", "?")
        home = g["teams"]["home"]["team"].get("abbreviation", "?")
        state = g.get("status", {}).get("abstractGameState")
        if state == "Final":
            print(f"  {away}@{home}: final, skipping", file=sys.stderr)
            continue
        try:
            got = process_game(g, season, day, lg, not args.no_form,
                               not args.no_park, args.projected)
            rows.extend(got)
            print(f"  {away}@{home}: {len(got)} batters", file=sys.stderr)
        except Exception as exc:
            print(f"  {away}@{home}: error — {exc}", file=sys.stderr)

    rows = [r for r in rows if r["hit_prob"] >= args.min_prob]
    if args.sort == "prob":
        rows.sort(key=lambda r: r["hit_prob"], reverse=True)
    else:
        rows.sort(key=lambda r: (r["game"], r["team"], r["slot"]))
    if args.top:
        rows = rows[:args.top]

    if args.format == "table":
        text = render_table(rows)
    elif args.format == "csv":
        text = render_csv(rows)
    elif args.format == "json":
        text = json.dumps(rows, indent=2)
    else:
        text = render_html(rows, day)

    if args.out:
        with open(os.path.expanduser(args.out), "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"Wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

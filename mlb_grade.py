#!/usr/bin/env python3
"""
mlb_grade.py — score past hit-probability predictions against real box scores.

Reads the daily prediction snapshots written by mlb_hit_probs.py, looks up what
each batter actually did, and keeps a running record broken out by probability
bucket. Also renders today's board so the slate only has to be pulled once.

Layout it expects / creates:
    data/predictions/YYYY-MM-DD.json   snapshot of that day's board
    data/graded/YYYY-MM-DD.json        what actually happened (written once)
    docs/index.html                    today's board
    docs/results.html                  the running record

Usage:
    python3 mlb_grade.py --today data/predictions/2026-08-23.json --date 2026-08-23
    python3 mlb_grade.py                # grade only, don't rebuild today's board
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_hit_probs import api_get, render_html  # noqa: E402

PRED_DIR = "data/predictions"
GRADE_DIR = "data/graded"
DOCS_DIR = "docs"

# Only track predictions at or above this probability — keeps files small.
MIN_TRACK = 0.60
# Buckets shown on the results page.
BUCKET_LO = 70
BUCKET_HI = 90


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

def actual_results(day: str) -> tuple[dict[int, dict], bool]:
    """Return ({player_id: {hits, pa}}, all_games_final) for a date."""
    sched = api_get("schedule", {"sportId": 1, "date": day})
    games = []
    for d in sched.get("dates", []):
        games.extend(d.get("games", []))
    if not games:
        return {}, False

    all_final = all(
        g.get("status", {}).get("abstractGameState") == "Final" for g in games
    )

    totals: dict[int, dict] = defaultdict(lambda: {"hits": 0, "pa": 0})
    for g in games:
        if g.get("status", {}).get("abstractGameState") != "Final":
            continue
        box = api_get(f"game/{g['gamePk']}/boxscore")
        for side in ("home", "away"):
            players = box.get("teams", {}).get(side, {}).get("players", {}) or {}
            for entry in players.values():
                pid = (entry.get("person") or {}).get("id")
                bat = (entry.get("stats") or {}).get("batting") or {}
                if not pid or not bat:
                    continue
                pa = bat.get("plateAppearances") or 0
                if not pa:
                    continue
                totals[pid]["hits"] += bat.get("hits") or 0
                totals[pid]["pa"] += pa
    return dict(totals), all_final


def grade_day(day: str, rows: list[dict]) -> list[dict] | None:
    """Score one day. Returns None if games aren't all final yet."""
    actual, final = actual_results(day)
    if not final or not actual:
        return None

    graded = []
    for r in rows:
        if r["hit_prob"] < MIN_TRACK:
            continue
        pid = r["batter_id"]
        got = actual.get(pid)
        if not got:
            continue  # scratched, never batted — not a fair test
        graded.append({
            "date": day,
            "batter": r["batter"],
            "batter_id": pid,
            "team": r["team"],
            "game": r["game"],
            "opp_sp": r["opp_sp"],
            "prob": r["hit_prob"],
            "exp_hits": r["exp_hits"],
            "source": r["lineup_source"],
            "pa": got["pa"],
            "hits": got["hits"],
            "got_hit": 1 if got["hits"] > 0 else 0,
        })
    return graded


def grade_pending() -> int:
    """Grade every prediction file that doesn't have a graded counterpart."""
    os.makedirs(GRADE_DIR, exist_ok=True)
    today = datetime.utcnow().date().isoformat()
    done = 0

    for path in sorted(glob.glob(f"{PRED_DIR}/*.json")):
        day = os.path.basename(path)[:-5]
        if day >= today:
            continue  # today's games aren't over
        if os.path.exists(f"{GRADE_DIR}/{day}.json"):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                rows = json.load(fh)
        except Exception as exc:
            print(f"  ! could not read {path}: {exc}", file=sys.stderr)
            continue
        graded = grade_day(day, rows)
        if graded is None:
            print(f"  {day}: games not final yet, will retry", file=sys.stderr)
            continue
        with open(f"{GRADE_DIR}/{day}.json", "w", encoding="utf-8") as fh:
            json.dump(graded, fh, indent=1)
        hits = sum(g["got_hit"] for g in graded if g["prob"] >= 0.70)
        n = sum(1 for g in graded if g["prob"] >= 0.70)
        print(f"  {day}: graded {len(graded)} rows — 70%+ went {hits}/{n}",
              file=sys.stderr)
        done += 1
    return done


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def load_graded() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(f"{GRADE_DIR}/*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                rows.extend(json.load(fh))
        except Exception:
            pass
    return rows


def tally(rows: list[dict]) -> dict:
    n = len(rows)
    hits = sum(r["got_hit"] for r in rows)
    exp = sum(r["prob"] for r in rows)
    return {
        "n": n,
        "hits": hits,
        "rate": (hits / n) if n else 0.0,
        "expected": (exp / n) if n else 0.0,
        "edge": ((hits - exp) / n) if n else 0.0,
        "total_hits": sum(r["hits"] for r in rows),
    }


def build_summary(rows: list[dict]) -> dict:
    posted = [r for r in rows if r["source"] == "POSTED"]

    by_bucket = {}
    for pct in range(BUCKET_LO, BUCKET_HI + 1):
        lo = pct / 100.0
        hi = (pct + 1) / 100.0
        sel = [r for r in posted if lo <= r["prob"] < hi]
        if pct == BUCKET_HI:
            sel = [r for r in posted if r["prob"] >= lo]
        if sel:
            by_bucket[pct] = tally(sel)

    cumulative = {}
    for pct in range(BUCKET_LO, 86):
        sel = [r for r in posted if r["prob"] >= pct / 100.0]
        if sel:
            cumulative[pct] = tally(sel)

    by_day = {}
    for r in posted:
        if r["prob"] >= 0.70:
            by_day.setdefault(r["date"], []).append(r)
    daily = {d: tally(v) for d, v in sorted(by_day.items(), reverse=True)}

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "overall_70_posted": tally([r for r in posted if r["prob"] >= 0.70]),
        "overall_70_all": tally([r for r in rows if r["prob"] >= 0.70]),
        "days_tracked": len({r["date"] for r in rows}),
        "by_bucket": by_bucket,
        "cumulative": cumulative,
        "daily": daily,
    }


# ---------------------------------------------------------------------------
# Results page
# ---------------------------------------------------------------------------

def render_results(s: dict) -> str:
    o = s["overall_70_posted"]

    def edge_span(v):
        color = "#15803d" if v >= 0 else "#b91c1c"
        return f"<span style='color:{color}'>{v * 100:+.1f}</span>"

    def rowset(d, label_fmt):
        out = []
        for k, t in sorted(d.items()):
            out.append(
                f"<tr><td class='n'>{label_fmt.format(k)}</td>"
                f"<td>{t['n']}</td><td>{t['hits']}</td>"
                f"<td class='n'>{t['rate'] * 100:.1f}%</td>"
                f"<td>{t['expected'] * 100:.1f}%</td>"
                f"<td>{edge_span(t['edge'])}</td></tr>")
        return "\n".join(out) or "<tr><td colspan='6'>Not enough data yet.</td></tr>"

    daily_rows = []
    for day, t in list(s["daily"].items())[:30]:
        daily_rows.append(
            f"<tr><td class='n'>{day}</td><td>{t['hits']}/{t['n']}</td>"
            f"<td class='n'>{t['rate'] * 100:.1f}%</td>"
            f"<td>{t['expected'] * 100:.1f}%</td>"
            f"<td>{edge_span(t['edge'])}</td></tr>")
    daily_html = "\n".join(daily_rows) or "<tr><td colspan='5'>No graded days yet.</td></tr>"

    return f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Track record</title>
<style>
 body{{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      margin:20px;color:#1a1a1a;background:#fafaf8}}
 h1{{font-size:19px;margin:0 0 2px}} h2{{font-size:14px;margin:26px 0 8px;
      text-transform:uppercase;letter-spacing:.06em;color:#555}}
 p.sub{{color:#666;margin:0 0 16px;font-size:12px}}
 a{{color:#1d4ed8}}
 .cards{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px}}
 .card{{background:#fff;border-radius:8px;padding:12px 14px;flex:1;min-width:120px;
        box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .card .big{{font-size:24px;font-weight:700;line-height:1.1}}
 .card .lbl{{font-size:11px;color:#666;text-transform:uppercase;letter-spacing:.05em}}
 table{{border-collapse:collapse;width:100%;background:#fff;
        box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:6px;overflow:hidden}}
 th{{background:#1f2937;color:#fff;text-align:left;padding:7px 9px;
     font-size:10px;letter-spacing:.05em;text-transform:uppercase}}
 td{{padding:6px 9px;border-top:1px solid #eee}} td.n{{font-weight:600}}
</style>
<h1>Track record</h1>
<p class="sub">{s['days_tracked']} day(s) graded &middot; updated {s['generated'][:16].replace('T', ' ')}
 &middot; <a href="index.html">today's board &rarr;</a></p>

<div class="cards">
  <div class="card"><div class="big">{o['hits']}/{o['n']}</div>
    <div class="lbl">70%+ record</div></div>
  <div class="card"><div class="big">{o['rate'] * 100:.1f}%</div>
    <div class="lbl">actual hit rate</div></div>
  <div class="card"><div class="big">{o['expected'] * 100:.1f}%</div>
    <div class="lbl">model predicted</div></div>
  <div class="card"><div class="big">{edge_span(o['edge'])}</div>
    <div class="lbl">over / under</div></div>
</div>
<p class="sub">Posted lineups only. Including projected lineups:
 {s['overall_70_all']['hits']}/{s['overall_70_all']['n']}
 ({s['overall_70_all']['rate'] * 100:.1f}%).</p>

<h2>By probability bucket</h2>
<table><tr><th>Bucket</th><th>N</th><th>Hits</th><th>Actual</th>
<th>Predicted</th><th>Diff</th></tr>
{rowset(s['by_bucket'], '{}%')}</table>

<h2>Cumulative &mdash; every pick at or above</h2>
<table><tr><th>Threshold</th><th>N</th><th>Hits</th><th>Actual</th>
<th>Predicted</th><th>Diff</th></tr>
{rowset(s['cumulative'], '&ge;{}%')}</table>

<h2>Day by day (70%+)</h2>
<table><tr><th>Date</th><th>Record</th><th>Actual</th><th>Predicted</th><th>Diff</th></tr>
{daily_html}</table>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", help="path to today's prediction JSON")
    ap.add_argument("--date", help="date of that JSON, YYYY-MM-DD")
    args = ap.parse_args()

    os.makedirs(DOCS_DIR, exist_ok=True)

    if args.today and os.path.exists(args.today):
        with open(args.today, encoding="utf-8") as fh:
            rows = json.load(fh)
        with open(f"{DOCS_DIR}/index.html", "w", encoding="utf-8") as fh:
            fh.write(render_html(rows, args.date or ""))
        print(f"Board: {len(rows)} batters", file=sys.stderr)

    print("Grading past days...", file=sys.stderr)
    grade_pending()

    graded = load_graded()
    summary = build_summary(graded)
    with open("data/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(f"{DOCS_DIR}/results.html", "w", encoding="utf-8") as fh:
        fh.write(render_results(summary))

    o = summary["overall_70_posted"]
    print(f"Record at 70%+: {o['hits']}/{o['n']} ({o['rate'] * 100:.1f}%)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
mlb_grade.py — score past prop predictions against real box scores.

Grades four markets: 1+ hit, 2+ hits, 1+ run scored, 1+ RBI. Keeps a running
record per market and per probability bucket, and renders today's board so the
slate only has to be pulled once.

    data/predictions/YYYY-MM-DD.json   snapshot of that day's board
    data/graded/YYYY-MM-DD.json        what actually happened
    docs/index.html                    today's board
    docs/results.html                  the running record

Usage:
    python3 mlb_grade.py --today data/predictions/2026-08-29.json --date 2026-08-29
    python3 mlb_grade.py --backfill 3
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import date as _date, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlb_hit_probs import api_get, now_local_str, render_html  # noqa: E402

PRED_DIR = "data/predictions"
GRADE_DIR = "data/graded"
DOCS_DIR = "docs"

# market key -> (probability field, result field, label, headline threshold)
MARKETS = [
    ("hit", "hit_prob", "got_hit", "1+ hit", 0.70),
    ("hrr2", "prob_hrr2", "got_hrr2", "2+ hits+runs+RBI", 0.50),
    ("hrr3", "prob_hrr3", "got_hrr3", "3+ hits+runs+RBI", 0.30),
    ("h2", "prob_2h", "got_2h", "2+ hits", 0.30),
    ("run", "prob_run", "got_run", "1+ run", 0.35),
    ("rbi", "prob_rbi", "got_rbi", "1+ RBI", 0.35),
]

# Markets that also get a day-by-day breakdown, kept fully separate.
DAY_VIEWS = [
    ("hit", "hit_prob", "got_hit", "1+ hit", 0.70),
    ("hrr2", "prob_hrr2", "got_hrr2", "2+ H+R+RBI", 0.50),
]

# 1%-wide buckets for hits; 2%-wide for the noisier markets.
BUCKETS = {"hit": (52, 88, 2), "h2": (8, 44, 2),
           "run": (14, 48, 2), "rbi": (14, 48, 2),
           "hrr2": (20, 76, 4), "hrr3": (8, 52, 4)}


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def actual_results(day: str) -> tuple[dict[int, dict], bool]:
    """({player_id: {hits, runs, rbi, pa}}, all_games_final) for a date."""
    sched = api_get("schedule", {"sportId": 1, "date": day})
    games = []
    for d in sched.get("dates", []):
        games.extend(d.get("games", []))
    if not games:
        return {}, False

    all_final = all(
        g.get("status", {}).get("abstractGameState") == "Final" for g in games)

    totals: dict[int, dict] = defaultdict(
        lambda: {"hits": 0, "runs": 0, "rbi": 0, "pa": 0})
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
                totals[pid]["pa"] += pa
                totals[pid]["hits"] += bat.get("hits") or 0
                totals[pid]["runs"] += bat.get("runs") or 0
                totals[pid]["rbi"] += bat.get("rbi") or 0
    return dict(totals), all_final


def grade_day(day: str, rows: list[dict]) -> list[dict] | None:
    actual, final = actual_results(day)
    if not final or not actual:
        return None

    graded = []
    for r in rows:
        got = actual.get(r["batter_id"])
        if not got:
            continue  # scratched, never batted — not a fair test
        graded.append({
            "date": day,
            "batter": r["batter"],
            "batter_id": r["batter_id"],
            "team": r["team"],
            "game": r["game"],
            "opp_sp": r["opp_sp"],
            "source": r["lineup_source"],
            "hit_prob": r["hit_prob"],
            "prob_2h": r.get("prob_2h", 0.0),
            "prob_run": r.get("prob_run", 0.0),
            "prob_rbi": r.get("prob_rbi", 0.0),
            "prob_hrr2": r.get("prob_hrr2", 0.0),
            "prob_hrr3": r.get("prob_hrr3", 0.0),
            "pa": got["pa"],
            "hits": got["hits"],
            "runs": got["runs"],
            "rbi": got["rbi"],
            "got_hit": 1 if got["hits"] >= 1 else 0,
            "got_2h": 1 if got["hits"] >= 2 else 0,
            "got_run": 1 if got["runs"] >= 1 else 0,
            "got_rbi": 1 if got["rbi"] >= 1 else 0,
            "hrr": got["hits"] + got["runs"] + got["rbi"],
            "got_hrr2": 1 if got["hits"] + got["runs"] + got["rbi"] >= 2 else 0,
            "got_hrr3": 1 if got["hits"] + got["runs"] + got["rbi"] >= 3 else 0,
        })
    return graded


def _has_rows(path: str) -> bool:
    """True only if the snapshot exists AND actually contains picks.

    A day that was built before a bug fix can be sitting there with zero rows.
    Those should be rebuilt, not skipped.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return isinstance(data, list) and len(data) > 0
    except Exception:
        return False


def backfill(days_back: int, anchor: str, asof: bool = True,
             skip_existing: bool = True) -> None:
    """Rebuild prediction snapshots for past days.

    With asof=True (the default) each day is scored using only stats from
    before that day, so the result is an honest backtest rather than a replay
    that already knows how the games went.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    built = skipped = empty = 0
    first = (_date.fromisoformat(anchor) - timedelta(days=days_back)).isoformat()
    last = (_date.fromisoformat(anchor) - timedelta(days=1)).isoformat()
    print(f"Backfill: checking {days_back} day(s), {first} through {last}"
          f"{' (point-in-time)' if asof else ' (LEAKS FUTURE STATS)'}",
          file=sys.stderr)

    for i in range(1, days_back + 1):
        day = (_date.fromisoformat(anchor) - timedelta(days=i)).isoformat()
        path = f"{PRED_DIR}/{day}.json"
        if skip_existing and _has_rows(path):
            skipped += 1
            continue
        if os.path.exists(path):
            empty += 1
            print(f"  rebuilding {day} (existing snapshot was empty)...",
                  file=sys.stderr)
        else:
            print(f"  building {day}...", file=sys.stderr)
        cmd = [sys.executable, os.path.join(here, "mlb_hit_probs.py"),
               "--date", day, "--min-prob", "0.70", "--keep-all",
               "--format", "json", "--out", path]
        if asof:
            cmd.append("--asof")
        subprocess.run(cmd, check=False)
        built += 1

    print(f"Backfill done: {built} built ({empty} were empty and got rebuilt), "
          f"{skipped} already had data.", file=sys.stderr)
    if built == 0 and skipped:
        print(f"  Nothing new — every day back to {first} already has picks. "
              f"Use a larger --backfill to reach further back.", file=sys.stderr)


def grade_pending() -> int:
    os.makedirs(GRADE_DIR, exist_ok=True)
    today = now_local_str()[:10]
    done = 0
    for path in sorted(glob.glob(f"{PRED_DIR}/*.json")):
        day = os.path.basename(path)[:-5]
        if day >= today:
            continue
        graded_path = f"{GRADE_DIR}/{day}.json"
        if _has_rows(graded_path):
            continue
        if not _has_rows(path):
            continue  # no picks to grade for this day
        if os.path.exists(graded_path):
            print(f"  {day}: previous grading was empty, redoing",
                  file=sys.stderr)
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
        sel = [g for g in graded if g.get("hit_prob", 0.0) >= 0.70]
        won = sum(g.get("got_hit", 0) for g in sel)
        print(f"  {day}: graded {len(graded)} rows — 70%+ hits went "
              f"{won}/{len(sel)}", file=sys.stderr)
        done += 1
    return done


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


PROB_KEYS = ("hit_prob", "prob_2h", "prob_run", "prob_rbi",
             "prob_hrr2", "prob_hrr3")

# Minimum graded picks before a market's calibration is trusted.
MIN_CALIBRATION_N = 400
CALIBRATION_PATH = "data/calibration.json"
RESULT_KEYS = ("got_hit", "got_2h", "got_run", "got_rbi",
               "got_hrr2", "got_hrr3")


def _normalize(r: dict) -> dict:
    """Make a row written by an older version safe for the current code.

    Early versions stored the hit probability as 'prob' and never recorded
    runs or RBI. Those days still count toward the 1+ hit record; for markets
    they have no data for, the probability is left at 0 so they never qualify
    rather than being scored as losses.
    """
    if "hit_prob" not in r and "prob" in r:
        r["hit_prob"] = r["prob"]
    for k in PROB_KEYS:
        r.setdefault(k, 0.0)
    for k in RESULT_KEYS:
        r.setdefault(k, 0)
    for k in ("hits", "runs", "rbi", "pa"):
        r.setdefault(k, 0)
    if "hrr" not in r:
        r["hrr"] = r["hits"] + r["runs"] + r["rbi"]
    r.setdefault("source", "POSTED")
    return r


def load_graded() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(f"{GRADE_DIR}/*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                rows.extend(_normalize(r) for r in json.load(fh))
        except Exception as exc:
            print(f"  ! skipping {path}: {exc}", file=sys.stderr)
    return rows


def tally(rows: list[dict], pkey: str, rkey: str) -> dict:
    n = len(rows)
    hits = sum(r.get(rkey, 0) for r in rows)
    exp = sum(r.get(pkey, 0.0) for r in rows)
    return {"n": n, "hits": hits, "miss": n - hits,
            "rate": (hits / n) if n else 0.0,
            "expected": (exp / n) if n else 0.0,
            "edge": ((hits - exp) / n) if n else 0.0}


def market_summary(rows: list[dict], mkey: str, pkey: str,
                   rkey: str, thresh: float) -> dict:
    lo, hi, step = BUCKETS[mkey]
    qualified = [r for r in rows if r.get(pkey, 0.0) >= thresh]

    by_bucket = {}
    for pct in range(lo, hi + 1, step):
        a, b = pct / 100.0, (pct + step) / 100.0
        sel = [r for r in rows if a <= r.get(pkey, 0.0) < b]
        if pct == hi:
            sel = [r for r in rows if r.get(pkey, 0.0) >= a]
        if sel:
            by_bucket[pct] = tally(sel, pkey, rkey)

    cumulative = {}
    for pct in range(lo, hi + 1, step):
        sel = [r for r in rows if r.get(pkey, 0.0) >= pct / 100.0]
        if len(sel) >= 3:
            cumulative[pct] = tally(sel, pkey, rkey)

    deciles = []
    scored = sorted((r for r in rows if r.get(pkey, 0.0) > 0),
                    key=lambda r: r.get(pkey, 0.0))
    if len(scored) >= 50:
        size = len(scored) // 10
        for i in range(10):
            chunk = scored[i * size:(i + 1) * size if i < 9 else len(scored)]
            if not chunk:
                continue
            t = tally(chunk, pkey, rkey)
            t["lo"] = chunk[0].get(pkey, 0.0)
            t["hi"] = chunk[-1].get(pkey, 0.0)
            deciles.append(t)

    return {"overall": tally(qualified, pkey, rkey),
            "threshold": thresh, "step": step, "deciles": deciles,
            "graded_n": len(scored),
            "by_bucket": by_bucket, "cumulative": cumulative}


def build_summary(rows: list[dict]) -> dict:
    posted = [r for r in rows if r["source"] == "POSTED"]

    markets = {}
    for mkey, pkey, rkey, label, thresh in MARKETS:
        markets[mkey] = market_summary(rows, mkey, pkey, rkey, thresh)
        markets[mkey]["label"] = label
        markets[mkey]["posted"] = tally(
            [r for r in posted if r.get(pkey, 0.0) >= thresh], pkey, rkey)

    def per_day(pkey, rkey, thresh):
        buckets = {}
        for r in rows:
            if r.get(pkey, 0.0) >= thresh:
                buckets.setdefault(r["date"], []).append(r)
        ordered = sorted(buckets.items(), reverse=True)
        return ({d: tally(v, pkey, rkey) for d, v in ordered},
                {d: sorted(v, key=lambda r: -r.get(pkey, 0.0)) for d, v in ordered[:14]})

    views = {}
    for mkey, pkey, rkey, label, thresh in DAY_VIEWS:
        daily, detail = per_day(pkey, rkey, thresh)
        views[mkey] = {"label": label, "pkey": pkey, "rkey": rkey,
                       "thresh": thresh, "daily": daily, "detail": detail}

    return {"generated": now_local_str(),
            "days_tracked": len({r["date"] for r in rows}),
            "markets": markets, "views": views}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_platt(pairs: list[tuple[float, int]], iters: int = 4000,
              lr: float = 0.05) -> tuple[float, float]:
    """Fit p_true = sigmoid(a * logit(p_raw) + b) by gradient descent.

    Two parameters only. `a` under 1 pulls extreme predictions toward the
    middle, which is the fix for a model that is confidently wrong at the
    edges; `b` shifts the overall level. Because the mapping is monotonic,
    it changes the numbers without disturbing the ranking.
    """
    if len(pairs) < 20:
        return 1.0, 0.0
    zs = [(_logit(p), y) for p, y in pairs]
    n = len(zs)
    a, b = 1.0, 0.0
    for i in range(iters):
        ga = gb = 0.0
        for z, y in zs:
            err = _sigmoid(a * z + b) - y
            ga += err * z
            gb += err
        a -= lr * ga / n
        b -= lr * gb / n
        if i % 500 == 499:
            lr *= 0.7
    return a, b


def fit_calibration(rows: list[dict]) -> dict:
    """Learn a correction per market from everything graded so far."""
    out = {}
    for mkey, pkey, rkey, label, _ in MARKETS:
        pairs = []
        for r in rows:
            # Always fit on the RAW model output. Once calibration is live the
            # stored probability is already corrected; fitting on that would
            # apply the correction twice.
            raw = r.get(pkey + "_raw")
            if raw is None:
                raw = r.get(pkey)
            if raw is None or raw <= 0:
                continue
            pairs.append((float(raw), int(r.get(rkey, 0))))
        if len(pairs) < MIN_CALIBRATION_N:
            continue
        a, b = fit_platt(pairs)
        before = sum(p for p, _ in pairs) / len(pairs)
        after = sum(_sigmoid(a * _logit(p) + b) for p, _ in pairs) / len(pairs)
        actual = sum(y for _, y in pairs) / len(pairs)
        out[pkey] = {"a": round(a, 5), "b": round(b, 5), "n": len(pairs),
                     "label": label,
                     "mean_raw": round(before, 4),
                     "mean_calibrated": round(after, 4),
                     "mean_actual": round(actual, 4)}
    return out


ODDS_PAGE = r"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Odds check</title>
<style>
 body{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      margin:18px;color:#1a1a1a;background:#fafaf8}
 h1{font-size:19px;margin:0 0 2px}
 p.sub{color:#666;margin:0 0 14px;font-size:12px}
 a{color:#1d4ed8}
 label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
       color:#555;margin:14px 0 5px}
 textarea{width:100%;min-height:150px;padding:10px;border:1px solid #ccc;
          border-radius:6px;font:13px ui-monospace,Menlo,monospace;box-sizing:border-box}
 select,button{font:14px inherit;padding:9px 12px;border-radius:6px;
               border:1px solid #ccc;background:#fff}
 button{background:#1f2937;color:#fff;border:none;font-weight:600;cursor:pointer}
 table{border-collapse:collapse;width:100%;background:#fff;margin-top:14px;
       box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:6px;overflow:hidden}
 th{background:#1f2937;color:#fff;text-align:left;padding:7px 9px;font-size:10px;
    letter-spacing:.05em;text-transform:uppercase}
 td{padding:7px 9px;border-top:1px solid #eee}
 td.n{font-weight:600}
 .pos{color:#15803d;font-weight:700}
 .neg{color:#b91c1c}
 .warn{background:#fef3c7;border:1px solid #fcd34d;padding:10px 12px;
        border-radius:6px;font-size:12px;margin:14px 0}
 .miss{color:#b91c1c;font-size:12px;margin-top:8px}
</style>
<h1>Odds check</h1>
<p class="sub"><span id="meta">loading...</span> &middot;
 <a href="index.html">board</a> &middot; <a href="results.html">track record</a></p>

<label>Market</label>
<select id="market">
  <option value="prob_hrr2">2+ hits+runs+RBI</option>
  <option value="hit_prob">1+ hit</option>
  <option value="prob_hrr3">3+ hits+runs+RBI</option>
  <option value="prob_2h">2+ hits</option>
  <option value="prob_run">1+ run</option>
  <option value="prob_rbi">1+ RBI</option>
</select>

<label>Paste players and odds &mdash; one per line</label>
<textarea id="input" placeholder="Yandy Diaz -150
Steven Kwan +105
Pete Alonso -120
Elly De La Cruz 130"></textarea>
<p class="sub">American odds. Plus sign optional; a bare number counts as positive.
 Anything else on the line is ignored.</p>
<button onclick="run()">Rank by value</button>

<div id="out"></div>

<script>
let PICKS = [];

function norm(s){
  return s.normalize("NFD").replace(/[\u0300-\u036f]/g,"")
          .toLowerCase().replace(/[^a-z ]/g,"").replace(/\s+/g," ").trim();
}
function lastName(s){ const p = norm(s).split(" "); return p[p.length-1]; }

fetch("today.json").then(r=>r.json()).then(d=>{
  PICKS = d.picks || [];
  document.getElementById("meta").textContent =
    PICKS.length + " batters on " + (d.date || "today");
}).catch(function(e){
  document.getElementById("meta").textContent = "could not load today's board";
});

function profit(odds){ return odds > 0 ? odds/100 : 100/Math.abs(odds); }
function implied(odds){ return 1/(1+profit(odds)); }
function breakeven(p){
  // profit per $1 the bet must return to break even
  const need = (1-p)/p;
  // American odds below even money are quoted as negative; there is no
  // such thing as "+76". Round so the reported price is still profitable.
  return need >= 1 ? "+" + Math.ceil(need*100)
                   : "-" + Math.floor(100/need);
}
function fmtOdds(o){ return o > 0 ? "+"+o : ""+o; }

function findPick(name){
  const n = norm(name);
  let hit = PICKS.find(function(p){ return norm(p.batter) === n; });
  if (hit) return hit;
  hit = PICKS.find(function(p){
    return norm(p.batter).startsWith(n) || n.startsWith(norm(p.batter)); });
  if (hit) return hit;
  const ln = lastName(name);
  const all = PICKS.filter(function(p){ return lastName(p.batter) === ln; });
  return all.length === 1 ? all[0] : null;
}

function run(){
  const key = document.getElementById("market").value;
  const lines = document.getElementById("input").value.split("\n");
  const rows = [], missing = [];

  for (const raw of lines){
    const line = raw.trim();
    if (!line) continue;
    const m = line.match(/([+-]?\d{2,5})\s*$/);
    if (!m){ missing.push(line + "  (no odds found)"); continue; }
    const odds = parseInt(m[1], 10);
    if (!odds || Math.abs(odds) < 100){ missing.push(line + "  (odds look wrong)"); continue; }
    const name = line.slice(0, m.index).replace(/[,\t|]+$/,"").trim();
    const pick = findPick(name);
    if (!pick){ missing.push(name + "  (not on today's board)"); continue; }
    const p = pick[key];
    if (p == null){ missing.push(name + "  (no number for this market)"); continue; }
    const imp = implied(odds);
    rows.push({name: pick.batter, team: pick.team, odds: odds, p: p, imp: imp,
               edge: p - imp, ev: 100*(p*profit(odds) - (1-p)), need: breakeven(p)});
  }

  rows.sort(function(a,b){ return b.ev - a.ev; });

  let html = "";
  if (rows.length){
    html += "<table><tr><th>Batter</th><th>Tm</th><th>Odds</th><th>Model</th>"
          + "<th>Book</th><th>Edge</th><th>EV/$100</th><th>Need</th></tr>";
    for (const r of rows){
      const cls = r.ev > 0 ? "pos" : "neg";
      html += "<tr><td class='n'>"+r.name+"</td><td>"+r.team+"</td>"
           + "<td>"+fmtOdds(r.odds)+"</td>"
           + "<td class='n'>"+(r.p*100).toFixed(1)+"%</td>"
           + "<td>"+(r.imp*100).toFixed(1)+"%</td>"
           + "<td class='"+cls+"'>"+(r.edge*100>=0?"+":"")+(r.edge*100).toFixed(1)+"</td>"
           + "<td class='"+cls+"'>"+(r.ev>=0?"+":"")+"$"+r.ev.toFixed(2)+"</td>"
           + "<td>"+r.need+"</td></tr>";
    }
    html += "</table>";
    const good = rows.filter(function(r){ return r.ev > 0; }).length;
    html += "<div class='warn'><b>"+good+" of "+rows.length+"</b> priced above what the "
         + "model thinks they are worth. <b>Need</b> is the shortest odds that would "
         + "still break even at the model's number.<br><br>"
         + "This only means something if the model's probabilities are accurate. "
         + "Check the track record page before trusting a positive edge - an "
         + "overconfident model invents edges that are not there.</div>";
  }
  if (missing.length){
    html += "<div class='miss'><b>Skipped:</b><br>" + missing.join("<br>") + "</div>";
  }
  document.getElementById("out").innerHTML = html || "<p>Nothing to rank.</p>";
}
</script>
"""


def write_odds_tools(rows, day):
    """Publish today's numbers plus the paste-in odds comparison page."""
    slim = [{"batter": r.get("batter"), "team": r.get("team"),
             "game": r.get("game"),
             "hit_prob": r.get("hit_prob"), "prob_2h": r.get("prob_2h"),
             "prob_run": r.get("prob_run"), "prob_rbi": r.get("prob_rbi"),
             "prob_hrr2": r.get("prob_hrr2"), "prob_hrr3": r.get("prob_hrr3")}
            for r in rows]
    with open(f"{DOCS_DIR}/today.json", "w", encoding="utf-8") as fh:
        json.dump({"date": day, "picks": slim}, fh)
    with open(f"{DOCS_DIR}/odds.html", "w", encoding="utf-8") as fh:
        fh.write(ODDS_PAGE)


# ---------------------------------------------------------------------------
# Results page
# ---------------------------------------------------------------------------


def _edge(v: float) -> str:
    color = "#15803d" if v >= 0 else "#b91c1c"
    return f"<span style='color:{color}'>{v * 100:+.1f}</span>"


def _bucket_table(d: dict, fmt: str) -> str:
    if not d:
        return ("<table><tr><th>&nbsp;</th></tr><tr><td>Not enough data yet."
                "</td></tr></table>")
    best = max((t["rate"] for t in d.values() if t["n"] >= 10), default=-1)
    rows = []
    for k, t in sorted(d.items()):
        star = " &#9733;" if t["n"] >= 10 and t["rate"] == best else ""
        bar = (f"<div style='background:#e5e7eb;border-radius:3px;height:8px;"
               f"width:70px'><div style='background:#2563eb;height:8px;"
               f"border-radius:3px;width:{t['rate'] * 70:.0f}px'></div></div>")
        rows.append(
            f"<tr><td class='n'>{fmt.format(k)}{star}</td>"
            f"<td class='y'>{t['hits']}</td><td class='x'>{t['miss']}</td>"
            f"<td>{t['n']}</td><td class='n'>{t['rate'] * 100:.1f}%</td>"
            f"<td>{t['expected'] * 100:.1f}%</td>"
            f"<td>{_edge(t['edge'])}</td><td>{bar}</td></tr>")
    head = ("<table><tr><th>Prob</th><th>Hit</th><th>Miss</th><th>Total</th>"
            "<th>Actual</th><th>Predicted</th><th>Diff</th><th></th></tr>")
    return head + "\n".join(rows) + "</table>"


def _market_block(m: dict, mkey: str) -> str:
    o = m["overall"]
    step = m["step"]
    thresh = int(m["threshold"] * 100)
    cards = (
        f"<div class='cards'>"
        f"<div class='card'><div class='big'>{o['hits']}&#8202;-&#8202;{o['miss']}</div>"
        f"<div class='lbl'>hits &ndash; misses</div></div>"
        f"<div class='card'><div class='big'>{o['rate'] * 100:.1f}%</div>"
        f"<div class='lbl'>actual</div></div>"
        f"<div class='card'><div class='big'>{o['expected'] * 100:.1f}%</div>"
        f"<div class='lbl'>predicted</div></div>"
        f"<div class='card'><div class='big'>{_edge(o['edge'])}</div>"
        f"<div class='lbl'>over / under</div></div></div>"
        f"<p class='sub'>All picks at {thresh}%+. Posted lineups only: "
        f"{m['posted']['hits']}/{m['posted']['n']} "
        f"({m['posted']['rate'] * 100:.1f}%).</p>")
    fmt = "{}%" if step == 1 else "{}%+"
    dec = ""
    if m.get("deciles"):
        rows_h = []
        for i, t in enumerate(m["deciles"], 1):
            bar = (f"<div style='background:#e5e7eb;border-radius:3px;height:8px;"
                   f"width:70px'><div style='background:#2563eb;height:8px;"
                   f"border-radius:3px;width:{t['rate'] * 70:.0f}px'></div></div>")
            rows_h.append(
                f"<tr><td class='n'>{i}</td>"
                f"<td>{t['lo'] * 100:.0f}&ndash;{t['hi'] * 100:.0f}%</td>"
                f"<td class='y'>{t['hits']}</td><td class='x'>{t['miss']}</td>"
                f"<td>{t['n']}</td><td class='n'>{t['rate'] * 100:.1f}%</td>"
                f"<td>{t['expected'] * 100:.1f}%</td>"
                f"<td>{_edge(t['edge'])}</td><td>{bar}</td></tr>")
        first, last = m["deciles"][0], m["deciles"][-1]
        spread = (last["rate"] - first["rate"]) * 100
        verdict = ("separates well" if spread >= 12 else
                   "some separation" if spread >= 6 else
                   "little separation &mdash; not ranking hitters")
        dec = (f"<h3>Ranking check &mdash; all {m['graded_n']} graded picks, "
               f"split into ten equal groups</h3>"
               f"<p class='sub'>If the model can rank, group 10 should hit far "
               f"more often than group 1. Spread here: "
               f"<b>{spread:+.1f} points</b> &mdash; {verdict}.</p>"
               f"<table><tr><th>Grp</th><th>Range</th><th>Hit</th><th>Miss</th>"
               f"<th>Total</th><th>Actual</th><th>Predicted</th><th>Diff</th>"
               f"<th></th></tr>{''.join(rows_h)}</table>")
    return (cards + dec
            + f"<h3>Each bucket</h3>{_bucket_table(m['by_bucket'], fmt)}"
            + f"<h3>Cumulative</h3>{_bucket_table(m['cumulative'], '&ge;{}%')}")


def render_results(s: dict) -> str:
    s_views = s["views"]
    blocks = []
    for mkey, _, _, label, _ in MARKETS:
        blocks.append(f"<h2>{label}</h2>" + _market_block(s["markets"][mkey], mkey))

    def day_panels(v):
        pkey, rkey = v["pkey"], v["rkey"]
        won_label = "WIN" if v["pkey"] == "prob_hrr2" else "HIT"
        out = []
        for i, (day, picks) in enumerate(v["detail"].items()):
            won = sum(p.get(rkey, 0) for p in picks)
            n = len(picks)
            body = "\n".join(
                f"<tr><td class='n'>{p.get(pkey, 0.0) * 100:.1f}%</td>"
                f"<td class='n'>{p['batter']}</td><td>{p['team']}</td>"
                f"<td>{p['opp_sp']}</td>"
                f"<td>{p.get('hits', 0)}-for-{p.get('pa', 0)}</td>"
                f"<td>{p.get('runs', 0)}</td><td>{p.get('rbi', 0)}</td>"
                f"<td class='n'>{p.get('hrr', 0)}</td>"
                f"<td class='{'y' if p.get(rkey) else 'x'}'>"
                f"{won_label if p.get(rkey) else 'no'}</td></tr>" for p in picks)
            out.append(
                f"<details {'open' if i == 0 else ''}><summary>{day} &mdash; "
                f"<b>{won}/{n}</b> ({(won / n * 100) if n else 0:.1f}%)</summary>"
                f"<table><tr><th>Prob</th><th>Batter</th><th>Tm</th>"
                f"<th>Opposing SP</th><th>Line</th><th>R</th><th>RBI</th>"
                f"<th>HRR</th><th>Result</th></tr>{body}</table></details>")
        return "\n".join(out) or "<p>No graded days yet.</p>"

    def daily_table(v):
        body = "\n".join(
            f"<tr><td class='n'>{d}</td><td>{t['hits']}/{t['n']}</td>"
            f"<td class='n'>{t['rate'] * 100:.1f}%</td>"
            f"<td>{t['expected'] * 100:.1f}%</td><td>{_edge(t['edge'])}</td></tr>"
            for d, t in list(v["daily"].items())[:30]
        ) or "<tr><td colspan='5'>No graded days yet.</td></tr>"
        return ("<table><tr><th>Date</th><th>Record</th><th>Actual</th>"
                "<th>Predicted</th><th>Diff</th></tr>" + body + "</table>")

    day_sections = []
    for mkey, _, _, label, thresh in DAY_VIEWS:
        v = s_views[mkey]
        day_sections.append(
            f"<h2>{label} &mdash; day by day</h2>"
            f"<p class='sub'>Every pick at {int(thresh * 100)}%+ on this market "
            f"only. Graded independently of the other markets.</p>"
            f"<h3>Each day's picks</h3>{day_panels(v)}"
            f"<h3>Daily summary</h3>{daily_table(v)}")
    day_block = "".join(day_sections)

    return f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Track record</title>
<style>
 body{{font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      margin:20px;color:#1a1a1a;background:#fafaf8}}
 h1{{font-size:19px;margin:0 0 2px}}
 h2{{font-size:15px;margin:30px 0 10px;padding-top:10px;
     border-top:2px solid #d4d4d0;color:#111}}
 h3{{font-size:11px;margin:16px 0 6px;text-transform:uppercase;
     letter-spacing:.06em;color:#666}}
 p.sub{{color:#666;margin:6px 0 12px;font-size:12px}}
 a{{color:#1d4ed8}}
 .cards{{display:flex;gap:8px;flex-wrap:wrap}}
 .card{{background:#fff;border-radius:8px;padding:10px 12px;flex:1;min-width:110px;
        box-shadow:0 1px 3px rgba(0,0,0,.08)}}
 .card .big{{font-size:21px;font-weight:700;line-height:1.1}}
 .card .lbl{{font-size:10px;color:#666;text-transform:uppercase;letter-spacing:.05em}}
 table{{border-collapse:collapse;width:100%;background:#fff;
        box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:6px;overflow:hidden}}
 th{{background:#1f2937;color:#fff;text-align:left;padding:7px 9px;
     font-size:10px;letter-spacing:.05em;text-transform:uppercase}}
 td{{padding:6px 9px;border-top:1px solid #eee}} td.n{{font-weight:600}}
 td.y{{color:#15803d;font-weight:700}} td.x{{color:#b91c1c}}
 details{{margin-bottom:10px;background:#fff;border-radius:6px;
          box-shadow:0 1px 3px rgba(0,0,0,.08);overflow:hidden}}
 summary{{padding:10px 12px;cursor:pointer;font-size:13px}}
 details table{{box-shadow:none;border-radius:0}}
</style>
<h1>Track record</h1>
<p class="sub">{s['days_tracked']} day(s) graded &middot; updated {s['generated']}
 &middot; <a href="index.html">today's board &rarr;</a>
 &middot; &#9733; marks the best bucket with 10+ tries</p>
{"".join(blocks)}
{day_block}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--today", help="path to today's prediction JSON")
    ap.add_argument("--date", help="date of that JSON, YYYY-MM-DD")
    ap.add_argument("--backfill", type=int, default=0, metavar="N",
                    help="build snapshots for the last N days (point-in-time)")
    ap.add_argument("--no-asof", action="store_true",
                    help="backfill using today's stats (leaks the future; "
                         "only for filling a gap you will not backtest on)")
    ap.add_argument("--rebuild", action="store_true",
                    help="overwrite existing snapshots during backfill")
    args = ap.parse_args()

    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)

    if args.backfill:
        backfill(args.backfill, args.date or now_local_str()[:10],
                 asof=not args.no_asof, skip_existing=not args.rebuild)

    if args.today and os.path.exists(args.today):
        with open(args.today, encoding="utf-8") as fh:
            rows = json.load(fh)
        with open(f"{DOCS_DIR}/index.html", "w", encoding="utf-8") as fh:
            fh.write(render_html(rows, args.date or ""))
        write_odds_tools(rows, args.date or "")
        print(f"Board: {len(rows)} rows", file=sys.stderr)

    print("Grading past days...", file=sys.stderr)
    grade_pending()

    graded = load_graded()
    calib = fit_calibration(graded)
    if calib:
        with open(CALIBRATION_PATH, "w", encoding="utf-8") as fh:
            json.dump(calib, fh, indent=1)
        print("Calibration refit:", file=sys.stderr)
        for k, c in calib.items():
            print(f"  {c['label']:20} a={c['a']:.3f} b={c['b']:+.3f}  "
                  f"mean {c['mean_raw']*100:.1f}% -> {c['mean_calibrated']*100:.1f}% "
                  f"(actual {c['mean_actual']*100:.1f}%)  n={c['n']}",
                  file=sys.stderr)

    summary = build_summary(graded)
    with open("data/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(f"{DOCS_DIR}/results.html", "w", encoding="utf-8") as fh:
        fh.write(render_results(summary))

    for mkey, _, _, label, _ in MARKETS:
        o = summary["markets"][mkey]["overall"]
        print(f"  {label:8} {o['hits']}-{o['miss']}  "
              f"actual {o['rate'] * 100:5.1f}%  predicted "
              f"{o['expected'] * 100:5.1f}%", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

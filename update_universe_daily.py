"""
update_universe_daily.py
------------------------
Daily maintenance of the point-in-time universes (Nifty MidSmallcap 400 and
Nifty Microcap 250).

For each universe, each run:
  1. Fetches the CURRENT official constituent list from NSE Indices.
  2. Saves that day's list to universe_snapshots/<key>_<date>.csv (the "who was
     in the index that day" file) and refreshes <key>_current.csv.
  3. Diffs today's list against the membership implied by its changes file and
     APPENDS any add/remove events (dated today) -- append-only.
  4. Rebuilds the combined universal master table (universe_membership.csv).

NSE only serves the CURRENT list (no historical feed), so this extends true
point-in-time history only FORWARD from when it first runs. Re-running the same
day is idempotent (build_universe_timeline applies post-snapshot events forward).

Usage:
    python update_universe_daily.py [--date YYYY-MM-DD] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import io
from datetime import datetime, timezone
from pathlib import Path

import requests

from universe import build_universe_timeline

ROOT = Path(__file__).resolve().parent
SNAP_DIR = ROOT / "universe_snapshots"

BASE = "https://www.niftyindices.com/IndexConstituent/"
UNIVERSES = {
    "midsmallcap400": {"url": BASE + "ind_niftymidsmallcap400list.csv", "changes": "changes.csv"},
    "microcap250":    {"url": BASE + "ind_niftymicrocap250_list.csv",   "changes": "changes_microcap250.csv"},
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.niftyindices.com/indices/equity/broad-based-indices",
}


def fetch_constituents(url: str, timeout: int = 60) -> list[dict]:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    members = []
    for row in csv.DictReader(io.StringIO(resp.text)):
        sym = (row.get("Symbol") or "").strip().upper()
        if not sym:
            continue
        members.append({
            "symbol": sym,
            "isin": (row.get("ISIN Code") or "").strip(),
            "company": (row.get("Company Name") or "").strip(),
            "industry": (row.get("Industry") or "").strip(),
        })
    if len(members) < 200:
        raise RuntimeError(f"Fetched list looks wrong: only {len(members)} constituents ({url}).")
    return members


def current_membership(changes_path: Path) -> set[str]:
    if not changes_path.exists():
        return set()
    timeline = build_universe_timeline(changes_path)
    return set(timeline[max(timeline)]) if timeline else set()


def save_snapshot(members: list[dict], key: str, date_str: str) -> Path:
    SNAP_DIR.mkdir(exist_ok=True)
    path = SNAP_DIR / f"{key}_{date_str}.csv"
    fieldnames = ["symbol", "isin", "company", "industry"]
    for target in (path, ROOT / f"{key}_current.csv"):
        with open(target, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(sorted(members, key=lambda m: m["symbol"]))
    return path


def append_events(changes_path: Path, adds: set[str], removes: set[str], date_str: str) -> int:
    existing = set()
    if changes_path.exists():
        with open(changes_path) as f:
            for row in csv.DictReader(f):
                existing.add((row["date"], row["symbol"].strip().upper(), row["action"]))
    new_rows = [(date_str, s, "add") for s in sorted(adds) if (date_str, s, "add") not in existing]
    new_rows += [(date_str, s, "remove") for s in sorted(removes) if (date_str, s, "remove") not in existing]
    if new_rows:
        with open(changes_path, "a", newline="") as f:
            csv.writer(f).writerows(new_rows)
    return len(new_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    changed_any = False
    for key, spec in UNIVERSES.items():
        changes_path = ROOT / spec["changes"]
        try:
            members = fetch_constituents(spec["url"])
        except Exception as exc:
            print(f"[{key}] fetch failed: {exc}")
            continue
        today = {m["symbol"] for m in members}
        known = current_membership(changes_path)
        adds, removes = today - known, known - today
        print(f"[{args.date}] {key}: fetched {len(today)}, known {len(known)} | "
              f"ADD {sorted(adds) or '-'} | REMOVE {sorted(removes) or '-'}")
        if args.dry_run:
            continue
        save_snapshot(members, key, args.date)
        n = append_events(changes_path, adds, removes, args.date)
        if n:
            changed_any = True
            print(f"  appended {n} event(s) to {spec['changes']}")

    if args.dry_run:
        print("dry-run: no files written.")
        return
    # Always rebuild the master view (cheap; reflects any appended events).
    from build_membership_master import main as rebuild_master
    rebuild_master()


if __name__ == "__main__":
    main()

"""
build_membership_master.py
--------------------------
Derives the "universal database" of index membership from the changes.csv event
logs: one row per membership SPELL, with entry and exit dates, for EVERY tracked
universe. A stock that left and rejoined has multiple rows.

Output: universe_membership.csv
    universe,isin,symbol,company,entry_date,exit_date,entry_approx,still_member

  - universe    : which index (midsmallcap400 | microcap250)
  - entry_date  : first date the stock was a member in that spell
  - exit_date   : first date it was no longer a member (blank if still in)
  - entry_approx: True when the spell begins before real tracking (dated to launch)
  - still_member: True when exit_date is blank

Regenerated from the changes files (the source of truth). ISIN/company are looked
up from the current constituent seeds where known.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from universe import build_universe_timeline

ROOT = Path(__file__).resolve().parent

# Each universe: its change-log and launch date (earliest membership can exist;
# base year differs from launch -- constituents were never published pre-launch).
UNIVERSES = {
    "midsmallcap400": {"changes": "changes.csv", "launch": "2016-04-01",
                       "seeds": ["current_list.csv", "midsmallcap400_current.csv"]},
    "microcap250":    {"changes": "changes_microcap250.csv", "launch": "2021-05-10",
                       "seeds": ["microcap250_current.csv"]},
}


def load_symbol_meta(seed_names: list[str]) -> dict[str, dict]:
    meta: dict[str, dict] = {}
    for name in seed_names:
        path = ROOT / name
        if not path.exists():
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                sym = (row.get("Symbol") or row.get("symbol") or "").strip().upper()
                if not sym:
                    continue
                meta[sym] = {
                    "isin": (row.get("ISIN Code") or row.get("isin") or "").strip(),
                    "company": (row.get("Company Name") or row.get("company") or "").strip(),
                }
    return meta


def spells_for_universe(name: str, changes_csv: Path, launch: str, seeds: list[str]) -> list[dict]:
    timeline = build_universe_timeline(changes_csv)
    dates = sorted(timeline.keys())
    if not dates:
        return []
    earliest = dates[0]
    meta = load_symbol_meta(seeds)
    ch = pd.read_csv(changes_csv, parse_dates=["date"])
    ch["symbol"] = ch["symbol"].str.strip().str.upper()
    added_at_earliest = set(ch[(ch["date"] == earliest) & (ch["action"] == "add")]["symbol"])

    symbols = sorted({s for members in timeline.values() for s in members})
    raw = []
    for sym in symbols:
        prev, entry = False, None
        for d in dates:
            present = sym in timeline[d]
            if present and not prev:
                entry = d
            elif (not present) and prev:
                raw.append((sym, entry, d))
                entry = None
            prev = present
        if prev:
            raw.append((sym, entry, None))

    out = []
    for sym, entry, exit_ in raw:
        info = meta.get(sym, {})
        founding = (entry == earliest) and (sym not in added_at_earliest)
        out.append({
            "universe": name,
            "isin": info.get("isin", ""),
            "symbol": sym,
            "company": info.get("company", ""),
            "entry_date": launch if founding else entry.date().isoformat(),
            "exit_date": exit_.date().isoformat() if exit_ is not None else "",
            "entry_approx": founding,
            "still_member": exit_ is None,
        })
    return out


def build_master() -> pd.DataFrame:
    rows: list[dict] = []
    for name, spec in UNIVERSES.items():
        path = ROOT / spec["changes"]
        if path.exists():
            rows += spells_for_universe(name, path, spec["launch"], spec["seeds"])
    cols = ["universe", "isin", "symbol", "company", "entry_date", "exit_date", "entry_approx", "still_member"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols].sort_values(["universe", "symbol", "entry_date"]).reset_index(drop=True)


def main() -> None:
    df = build_master()
    out = ROOT / "universe_membership.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df)} spells across {df['symbol'].nunique()} symbols.")
    for name, grp in df.groupby("universe"):
        print(f"  {name}: {len(grp)} spells, {grp['still_member'].sum()} current, "
              f"tracked from {grp['entry_date'].min()}")


if __name__ == "__main__":
    main()

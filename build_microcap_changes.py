"""
build_microcap_changes.py
-------------------------
Builds changes_microcap250.csv (point-in-time input for the Nifty Microcap 250
universe) from the SAME already-downloaded NSE press-release PDFs used for the
MidSmallcap 400 -- those PDFs carry a direct "Nifty Microcap 250" section. No
network needed; it re-parses pdfs/ locally.

Nifty Microcap 250 launched 2021-05-10, so genuine point-in-time membership only
exists from then. Output format matches changes.csv:
    date,symbol,action   with action in {snapshot_member, add, remove}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import pdfplumber

from build_universe_changes import ROW_PAT, parse_effective_date

ROOT = Path(__file__).resolve().parent
INDEX_NAME = "nifty microcap 250"


def extract_microcap_changes(text: str) -> list[tuple[str, str]]:
    """[(action, symbol)] for the Nifty Microcap 250 section of one press release."""
    in_section = False
    action = None
    out: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        low = line.lower()
        header = re.sub(r"^\d+\)\s*", "", low).strip().rstrip(":").strip()
        if header == INDEX_NAME or header.startswith(INDEX_NAME + " index"):
            in_section = True
            action = None
            continue
        # a header for any OTHER numbered index closes the microcap section
        if re.match(r"^\d+\)\s*nifty", low) and INDEX_NAME not in low:
            in_section = False
            action = None
            continue
        if not in_section:
            continue
        if "exclu" in low and ("compan" in low or "following" in low):
            action = "remove"
            continue
        if "inclu" in low and ("compan" in low or "following" in low):
            action = "add"
            continue
        if action:
            m = ROW_PAT.match(line)
            if m:
                out.append((action, m.group(2)))
            elif low.startswith(("sr. no", "sr no", "s. no")):
                continue
            elif line and not any(ch.isdigit() for ch in line[:3]):
                if len(line) > 60:
                    action = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="candidates.json")
    ap.add_argument("--pdf-dir", default="pdfs")
    ap.add_argument("--seed-csv", default="microcap250_current.csv")
    ap.add_argument("--seed-date", default=datetime.today().strftime("%Y-%m-%d"))
    ap.add_argument("--out", default="changes_microcap250.csv")
    args = ap.parse_args()

    titles = {Path(c.get("link", "")).name: c.get("title", "")
              for c in json.load(open(ROOT / args.candidates))}
    pdf_dir = ROOT / args.pdf_dir

    events = []
    reviewed = 0
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)
        except Exception:
            continue
        if INDEX_NAME not in text.lower():
            continue
        reviewed += 1
        changes = extract_microcap_changes(text)
        if not changes:
            continue
        eff = parse_effective_date(titles.get(pdf_path.name, ""), text)
        if eff is None:
            continue
        for action, sym in changes:
            events.append((eff.strftime("%Y-%m-%d"), sym.strip().upper(), action))

    # seed snapshot from the current constituent list
    with open(ROOT / args.seed_csv) as f:
        seed = sorted({row["Symbol"].strip().upper() for row in csv.DictReader(f) if row.get("Symbol", "").strip()})
    if len(seed) < 200:
        raise SystemExit(f"microcap seed looks wrong: only {len(seed)} symbols")

    seen = set()
    with open(ROOT / args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "symbol", "action"])
        for s in seed:
            w.writerow([args.seed_date, s, "snapshot_member"])
        for date, sym, action in sorted(events):
            key = (date, sym, action)
            if key in seen:
                continue
            seen.add(key)
            w.writerow([date, sym, action])

    print(f"Wrote {args.out}: {len(seed)} snapshot members + {len(seen)} events "
          f"from {reviewed} microcap-bearing PDFs.")


if __name__ == "__main__":
    main()

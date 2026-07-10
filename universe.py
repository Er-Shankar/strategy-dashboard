"""
universe.py
-----------
Builds the point-in-time Nifty Midsmallcap 400 constituent list for any given date.

IMPORTANT / HONEST LIMITATION:
NSE Indices does not publish a clean, free, bulk "constituents as of date X"
API. What they DO publish for free:

  1. The CURRENT constituent list (CSV download from niftyindices.com,
     under Indices > Nifty Midsmallcap 400 > "Download Full List").
  2. Individual "Index Maintenance" reconstitution press releases (PDFs),
     issued semi-annually (effective on/around the last business day of
     March and September for this index -- confirm exact cutover dates
     from niftyindices.com each cycle), listing stocks added/removed.

To get a genuine point-in-time universe for 10-15 years, you need to:
  a) Download the current list (seed the `changes.csv` file below with a
     `snapshot` row using today's date and all 400 current symbols).
  b) Walk backwards through each semi-annual reconstitution PDF and log
     every ADD/REMOVE event with its effective date into `changes.csv`.
     This is manual/semi-manual -- there is no bulk API for it.
  c) This script then reconstructs the universe for any historical date
     by starting from the seed snapshot and applying changes in reverse
     chronological order back to that date (or forward from an older
     snapshot if you have one).

If you don't complete step (b) for the full 10-15 year window, the
honest thing to do is document exactly which years have a verified
point-in-time universe and which years fall back to the nearest known
snapshot (flagged via the `approximated` column in the output) --
NOT silently assume today's constituents applied throughout.

changes.csv format:
    date,symbol,action        # action in {"snapshot_member","add","remove"}
    2026-07-01,RELIANCE,snapshot_member
    2026-07-01,...            # (all ~400 current members, action=snapshot_member)
    2021-09-30,XYZCORP,add
    2021-09-30,ABCLTD,remove
    ...
"""
import pandas as pd
from pathlib import Path


def load_changes(changes_csv: str) -> pd.DataFrame:
    df = pd.read_csv(changes_csv, parse_dates=["date"])
    df["symbol"] = df["symbol"].str.strip().str.upper()
    return df.sort_values("date")


def build_universe_timeline(changes_csv: str) -> dict[pd.Timestamp, set[str]]:
    """
    Returns {snapshot_date: set_of_symbols_from_that_point_forward}.
    Reconstructs by starting from the most recent 'snapshot_member' set and
    walking backward, reversing each add/remove event as we go further back
    in time (an 'add' on the way forward means the stock was NOT there before
    that date; a 'remove' means it WAS there before that date).
    """
    df = load_changes(changes_csv)
    snapshot_date = df.loc[df["action"] == "snapshot_member", "date"].max()
    current_set = set(df.loc[(df["action"] == "snapshot_member") & (df["date"] == snapshot_date), "symbol"])

    events = df[df["action"].isin(["add", "remove"])].sort_values("date", ascending=False)

    timeline = {snapshot_date: set(current_set)}
    running = set(current_set)
    for event_date, grp in events.groupby("date", sort=False):
        if event_date >= snapshot_date:
            continue
        for _, row in grp.iterrows():
            if row["action"] == "add":
                running.discard(row["symbol"])   # wasn't there before this add
            elif row["action"] == "remove":
                running.add(row["symbol"])       # was there before this remove
        timeline[event_date] = set(running)

    return dict(sorted(timeline.items()))


def get_universe_as_of(date, timeline: dict[pd.Timestamp, set[str]]) -> tuple[set[str], bool]:
    """
    Returns (universe_set, approximated_flag).
    approximated_flag=True if `date` is earlier than the earliest verified
    timeline entry (meaning we're falling back to the oldest known snapshot
    rather than a true point-in-time universe).
    """
    date = pd.to_datetime(date)
    dates_sorted = sorted(timeline.keys())
    applicable = [d for d in dates_sorted if d <= date]
    if not applicable:
        earliest = dates_sorted[0]
        return timeline[earliest], True   # approximated: using earliest known snapshot
    best = max(applicable)
    return timeline[best], False


if __name__ == "__main__":
    import sys
    changes_file = sys.argv[1] if len(sys.argv) > 1 else "changes.csv"
    if not Path(changes_file).exists():
        print(f"No {changes_file} found -- see the module docstring for the format required.")
        sys.exit(1)
    tl = build_universe_timeline(changes_file)
    for d, syms in tl.items():
        print(d.date(), "->", len(syms), "symbols")

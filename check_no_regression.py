"""Guard used by the daily-refresh workflow.

Yahoo Finance occasionally serves the most recent trading day with volume but
no finalized (adjusted) OHLC prices yet -- the row exists but Open/Close are
NaN, so refresh_market_data.py's dropna() silently drops that day for every
affected symbol. Because the pipeline always rebuilds the bundle from scratch
against a fixed git-committed baseline rather than the previous run's result,
that kind of upstream hiccup can make a fresh build cover FEWER trading days
than what's already live -- and without this check, the workflow would just
publish that regression over the better data already on GitHub Pages.

This compares the freshly-built site/data/metadata.json against the currently
deployed one and signals (via a regressed=true/false GitHub Actions output)
whether the deploy should be skipped, leaving the last good deployment in
place until a future run gets a complete day from Yahoo.
"""

import json
import os
import urllib.request

LIVE_METADATA_URL = "https://er-shankar.github.io/strategy-dashboard/data/metadata.json"


def main() -> None:
    new_meta = json.load(open("site/data/metadata.json"))
    new_end = new_meta.get("date_end", "")

    try:
        with urllib.request.urlopen(LIVE_METADATA_URL, timeout=15) as resp:
            live_meta = json.load(resp)
        live_end = live_meta.get("date_end", "")
    except Exception as exc:
        print(f"Could not fetch live metadata ({exc}); proceeding with deploy.")
        live_end = ""

    regressed = bool(live_end) and new_end < live_end
    print(f"live date_end={live_end!r} new date_end={new_end!r} regressed={regressed}")
    if regressed:
        print(
            f"::warning::New bundle ends {new_end}, older than the live site's {live_end} -- "
            "skipping deploy to avoid regressing (Yahoo likely served an incomplete final "
            "trading day). Will retry on the next run."
        )

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"regressed={'true' if regressed else 'false'}\n")


if __name__ == "__main__":
    main()

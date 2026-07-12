from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dashboard import SWEEP_LOOKBACK_SETS, json_safe, run_one_combo


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "site" / "data" / "simulation_results.json"
CHUNK_PREFIX = "simulation_results_part_"

SIMULATION_PARAMS: list[dict] = [
    {
        "key": "universe",
        "label": "Universe",
        "options": [
            {"value": ["midsmallcap400"], "label": "SmallMid 400"},
            {"value": ["microcap250"], "label": "Microcap 250"},
            {"value": ["midsmallcap400", "microcap250"], "label": "Both"},
        ],
    },
    {
        "key": "lookback_set",
        "label": "Lookback set",
        "options": [
            {"value": key, "label": key.replace("_", "+").upper()}
            for key in SWEEP_LOOKBACK_SETS
        ],
    },
    {
        "key": "skip_1m",
        "label": "Skip most recent 1M",
        "options": [{"value": False, "label": "No"}, {"value": True, "label": "Yes"}],
    },
    {
        "key": "vol_adjust",
        "label": "Volatility adjust",
        "options": [{"value": True, "label": "On"}, {"value": False, "label": "Off"}],
    },
    {
        "key": "top_n",
        "label": "Holdings",
        "options": [{"value": value, "label": str(value)} for value in (10, 15, 20, 30)],
    },
    {
        "key": "entry_rank",
        "label": "Entry rank",
        "options": [{"value": value, "label": str(value)} for value in (20, 32, 50)],
    },
    {
        "key": "exit_rank",
        "label": "Exit rank",
        "options": [{"value": value, "label": str(value)} for value in (32, 50)],
    },
    {
        "key": "rebalance_frequency",
        "label": "Frequency",
        "options": [
            {"value": "monthly", "label": "Monthly"},
            {"value": "quarterly", "label": "Quarterly"},
        ],
    },
    {
        "key": "weighting",
        "label": "Weighting",
        "options": [
            {"value": "equal", "label": "Equal"},
            {"value": "inverse_vol", "label": "Inverse vol"},
        ],
    },
    {
        "key": "trend_filter",
        "label": "Trend filter",
        "options": [{"value": "none", "label": "Off"}, {"value": "supertrend", "label": "On"}],
    },
    {
        "key": "supertrend_multiplier",
        "label": "Supertrend multiplier",
        "options": [{"value": value, "label": str(value)} for value in (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)],
    },
    {
        "key": "trend_timeframe",
        "label": "Supertrend timeframe",
        "options": [
            {"value": "weekly", "label": "Weekly"},
            {"value": "daily", "label": "Daily"},
        ],
    },
    {
        "key": "bearish_asset",
        "label": "Bearish destination",
        "options": [
            {"value": "cash", "label": "Cash"},
            {"value": "goldbees", "label": "GoldBees"},
        ],
    },
]


def all_combos(cartesian: bool = False) -> list[dict]:
    keys = [param["key"] for param in SIMULATION_PARAMS]
    value_lists = [[option["value"] for option in param["options"]] for param in SIMULATION_PARAMS]
    combos = [dict(zip(keys, values)) for values in itertools.product(*value_lists)]
    if cartesian:
        return combos

    # When the trend filter is off, Supertrend multiplier/timeframe and bearish
    # destination do not affect the backtest. Keep one canonical row instead of
    # shipping many duplicate results.
    unique: dict[str, dict] = {}
    for combo in combos:
        row = dict(combo)
        if row["trend_filter"] == "none":
            row["supertrend_multiplier"] = 2.5
            row["trend_timeframe"] = "weekly"
            row["bearish_asset"] = "cash"
        key = json.dumps(row, sort_keys=True)
        unique[key] = row
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate precomputed Simulation results for the static website.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N combinations for a quick smoke test.")
    parser.add_argument("--cartesian", action="store_true", help="Keep duplicate no-op combinations for a literal full Cartesian grid.")
    parser.add_argument("--chunk-size", type=int, default=25_000, help="Rows per result chunk. Use 0 to force one JSON file.")
    args = parser.parse_args()

    combos = all_combos(cartesian=args.cartesian)
    if args.limit:
        combos = combos[: args.limit]

    started = time.time()
    results: list[dict] = []
    print(f"Running {len(combos):,} simulation combinations with {args.workers} workers...", flush=True)

    if args.workers == 1:
        iterator = enumerate((run_one_combo({}, combo) for combo in combos), start=1)
        for index, row in iterator:
            results.append(row)
            if index == 1 or index % 100 == 0 or index == len(combos):
                elapsed = time.time() - started
                rate = index / elapsed if elapsed > 0 else 0
                remaining = (len(combos) - index) / rate if rate else 0
                print(f"{index:,}/{len(combos):,} done | elapsed {elapsed/60:.1f}m | eta {remaining/60:.1f}m", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_one_combo, {}, combo) for combo in combos]
            for index, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if index == 1 or index % 100 == 0 or index == len(futures):
                    elapsed = time.time() - started
                    rate = index / elapsed if elapsed > 0 else 0
                    remaining = (len(futures) - index) / rate if rate else 0
                    print(f"{index:,}/{len(futures):,} done | elapsed {elapsed/60:.1f}m | eta {remaining/60:.1f}m", flush=True)

    sort_keys = [param["key"] for param in SIMULATION_PARAMS]
    results.sort(key=lambda row: tuple(json.dumps(row.get(key), sort_keys=True) for key in sort_keys))

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 1),
        "total": len(combos),
        "completed": len(results),
        "params": SIMULATION_PARAMS,
        "cartesian": args.cartesian,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    for old_chunk in args.out.parent.glob(f"{CHUNK_PREFIX}*.json"):
        old_chunk.unlink()

    if args.chunk_size and len(results) > args.chunk_size:
        chunks = []
        for index, start in enumerate(range(0, len(results), args.chunk_size), start=1):
            chunk_rows = results[start:start + args.chunk_size]
            chunk_name = f"{CHUNK_PREFIX}{index:04d}.json"
            chunk_path = args.out.parent / chunk_name
            chunk_payload = {
                "schema_version": 1,
                "part": index,
                "rows": chunk_rows,
            }
            chunk_path.write_text(json.dumps(json_safe(chunk_payload), separators=(",", ":")))
            chunks.append({
                "file": chunk_name,
                "rows": len(chunk_rows),
                "bytes": chunk_path.stat().st_size,
            })
        payload["chunks"] = chunks
        args.out.write_text(json.dumps(json_safe(payload), separators=(",", ":")))
        total_bytes = args.out.stat().st_size + sum((args.out.parent / chunk["file"]).stat().st_size for chunk in chunks)
        print(f"Wrote {args.out} + {len(chunks)} chunks ({total_bytes / 1024 / 1024:.1f} MB total)", flush=True)
    else:
        payload["results"] = results
        args.out.write_text(json.dumps(json_safe(payload), separators=(",", ":")))
        print(f"Wrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()

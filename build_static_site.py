from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from universe import build_universe_timeline


ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"
DATA = SITE / "data"


def read_index_csv(path: Path) -> pd.Series:
    raw = path.read_text().splitlines()
    if len(raw) >= 4 and raw[0].lower().startswith("price,"):
        df = pd.read_csv(path, skiprows=3, names=["date", "close"])
    else:
        df = pd.read_csv(path)
        columns = {c.lower().strip(): c for c in df.columns}
        date_col = columns.get("date") or df.columns[0]
        close_col = columns.get("close") or columns.get("price") or df.columns[-1]
        df = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().sort_values("date").drop_duplicates("date")
    return pd.Series(df["close"].values, index=pd.DatetimeIndex(df["date"]), name=path.stem)


def read_ohlc_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns = {c.lower().strip(): c for c in df.columns}
    date_col = columns.get("date") or df.columns[0]
    rename = {
        date_col: "date",
        columns.get("open", "Open"): "open",
        columns.get("high", "High"): "high",
        columns.get("low", "Low"): "low",
        columns.get("close", "Close"): "close",
    }
    df = df.rename(columns=rename)
    df = df[[c for c in ["date", "open", "high", "low", "close"] if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().sort_values("date").drop_duplicates("date").set_index("date")


def round_or_none(value: object, ndigits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), ndigits)


def series_payload(series: pd.Series, dates: list[str]) -> list[float | None]:
    aligned = series.reindex(pd.to_datetime(dates))
    return [round_or_none(v, 4) for v in aligned.tolist()]


def matrix_payload(df: pd.DataFrame) -> list[list[float | None]]:
    # Column-major keeps each symbol contiguous, which is simpler for the browser engine.
    return [[round_or_none(v, 4) for v in df[col].tolist()] for col in df.columns]


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)

    close = pd.read_parquet(ROOT / "prices_wide_close.parquet").sort_index()
    open_wide = pd.read_parquet(ROOT / "prices_wide_open.parquet").sort_index()
    close.index = pd.to_datetime(close.index)
    open_wide.index = pd.to_datetime(open_wide.index)

    if (ROOT / "goldbees.csv").exists():
        gold = read_index_csv(ROOT / "goldbees.csv").rename("GOLDBEES").ffill()
        close = close.join(gold, how="outer")
        close["GOLDBEES"] = close["GOLDBEES"].ffill()

    close = close[close.notna().sum(axis=1) >= 200]
    open_wide = open_wide.reindex(close.index)
    if "GOLDBEES" in close.columns and "GOLDBEES" not in open_wide.columns:
        open_wide["GOLDBEES"] = close["GOLDBEES"]
    open_wide = open_wide.reindex(columns=close.columns)

    dates = [d.date().isoformat() for d in close.index]
    symbols = [str(c) for c in close.columns]
    symbol_index = {s: i for i, s in enumerate(symbols)}

    def build_universe(changes_path: Path) -> tuple[list[dict], set[str]]:
        timeline = build_universe_timeline(changes_path)
        snapshots, members_all = [], set()
        for date, members in timeline.items():
            members_all |= set(members)
            ids = sorted(symbol_index[s] for s in members if s in symbol_index)
            snapshots.append({"date": pd.to_datetime(date).date().isoformat(), "symbols": ids})
        return snapshots, members_all

    # Each selectable universe maps to its own point-in-time constituent timeline.
    universes: dict[str, list[dict]] = {}
    all_members: set[str] = set()
    universes["midsmallcap400"], m400 = build_universe(ROOT / "changes.csv")
    all_members |= m400
    if (ROOT / "changes_microcap250.csv").exists():
        universes["microcap250"], mmicro = build_universe(ROOT / "changes_microcap250.csv")
        all_members |= mmicro
    # Backward-compatible alias for the original single-universe key.
    universe = universes["midsmallcap400"]

    # Transparency: stocks that were once in a tracked universe but have no price
    # data (delisted/merged/defaulted names). They can't be traded, so historical
    # results carry mild survivorship bias.
    missing_universe = sorted(all_members - set(symbols))
    data_coverage = {
        "universe_symbols_total": len(all_members),
        "universe_symbols_with_prices": len(all_members) - len(missing_universe),
        "universe_symbols_missing": len(missing_universe),
        "missing_symbols": missing_universe,
        "note": ("Delisted/merged/defaulted stocks that left a tracked index and have "
                 "no price data. Excluded from backtests -> mild upward survivorship "
                 "bias in older years. Current constituents are fully covered."),
    }

    benchmarks: dict[str, dict] = {}
    if (ROOT / "nifty500.csv").exists():
        benchmarks["nifty500"] = {"close": series_payload(read_index_csv(ROOT / "nifty500.csv"), dates)}
    if (ROOT / "nifty50.csv").exists():
        benchmarks["nifty50"] = {"close": series_payload(read_index_csv(ROOT / "nifty50.csv"), dates)}
    if (ROOT / "nifty500_ohlc.csv").exists():
        # Ship OHLC on the benchmark's OWN trading calendar (not reindexed onto
        # the stock-universe calendar). The Supertrend trend filter must run on
        # the benchmark's real bars: reindexing inserted NaN-filled fake bars and
        # dropped genuine ones, corrupting ATR and shifting trend flips versus the
        # Python engine. The browser carries the native date axis to recompute it.
        ohlc = read_ohlc_csv(ROOT / "nifty500_ohlc.csv")
        benchmarks.setdefault("nifty500", {})
        benchmarks["nifty500"]["ohlc"] = {
            "dates": [d.date().isoformat() for d in ohlc.index],
            "open": [round_or_none(v, 4) for v in ohlc["open"].tolist()],
            "high": [round_or_none(v, 4) for v in ohlc["high"].tolist()],
            "low": [round_or_none(v, 4) for v in ohlc["low"].tolist()],
            "close": [round_or_none(v, 4) for v in ohlc["close"].tolist()],
        }

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dates": dates,
        "symbols": symbols,
        "close": matrix_payload(close),
        "open": matrix_payload(open_wide),
        "universe": universe,
        "universes": universes,
        "benchmarks": benchmarks,
        "data_coverage": data_coverage,
        "defaults": {
            "universe": ["midsmallcap400"],
            "start": "2016-04-01",
            "end": dates[-1],
            "lookbacks": ["3m", "6m", "9m"],
            "weights": {"1m": 0, "3m": 1, "6m": 1, "9m": 1, "12m": 1},
            "vol_adjust": True,
            "vol_lookback": "3m",
            "skip_1m": False,
            "top_n": 20,
            "entry_rank": 20,
            "exit_rank": 50,
            "rebalance_frequency": "monthly",
            "rebalance_day": 21,
            "weighting": "equal",
            "initial_capital": 1_000_000,
            "include_brokerage": True,
            "brokerage_rate": 0.0003,
            "include_stt": True,
            "stt_rate": 0.001,
            "include_sebi": True,
            "sebi_rate": 0.000001,
            "include_stamp": True,
            "stamp_duty_rate": 0.00015,
            "include_slippage": False,
            "slippage_rate": 0,
            "include_tax": True,
            "stcg_rate": 0.20,
            "ltcg_rate": 0.125,
            "ltcg_exemption": 125_000,
            "long_term_days": 365,
            "trend_filter": "none",
            "trend_source": "nifty500",
            "trend_timeframe": "weekly",
            "supertrend_atr_period": 1,
            "supertrend_multiplier": 2.5,
            "bearish_exposure": 0,
            "bearish_asset": "cash",
            "bearish_symbol": "GOLDBEES",
        },
        "warnings": [],
    }

    out = DATA / "market.json"
    out.write_text(json.dumps(payload, separators=(",", ":")))

    metadata = {
        "generated_at": payload["generated_at"],
        "date_start": dates[0],
        "date_end": dates[-1],
        "symbols": len(symbols),
        "trading_days": len(dates),
        "universe_snapshots": len(universe),
        "market_json_bytes": out.stat().st_size,
    }
    (DATA / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

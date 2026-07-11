from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent


def download_adjusted_ohlc(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    frames = []
    for i in range(0, len(symbols), 35):
        batch = symbols[i:i + 35]
        tickers = [f"{s}.NS" for s in batch if s != "GOLDBEES"]
        if not tickers:
            continue
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        for symbol in batch:
            if symbol == "GOLDBEES":
                continue
            ticker = f"{symbol}.NS"
            try:
                df = data[ticker] if len(tickers) > 1 else data
            except KeyError:
                continue
            if df is None or df.empty or df["Close"].dropna().empty:
                continue
            sub = df[["Open", "Close"]].dropna().reset_index()
            sub.columns = ["date", "open", "close"]
            sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None)
            sub["symbol"] = symbol
            frames.append(sub[["date", "symbol", "open", "close"]])
    if not frames:
        raise RuntimeError("No symbol data downloaded.")
    return pd.concat(frames, ignore_index=True)


def download_index(ticker: str, out: Path, ohlc_out: Path | None = None) -> None:
    df = yf.download(ticker, start="2011-01-01", auto_adjust=False, progress=False, threads=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    lines = ["Price,Close", f"Ticker,{ticker}", "Date,"]
    lines.extend(f"{idx.date().isoformat()},{float(value)}" for idx, value in close.items())
    out.write_text("\n".join(lines) + "\n")
    if ohlc_out is not None:
        ohlc = df[["Open", "High", "Low", "Close"]].copy()
        if isinstance(ohlc.columns, pd.MultiIndex):
            ohlc.columns = [c[0] for c in ohlc.columns]
        ohlc.dropna().to_csv(ohlc_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2011-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    if not (ROOT / "prices_wide_close.parquet").exists():
        raise SystemExit("prices_wide_close.parquet is required as the symbol source.")
    existing_close = pd.read_parquet(ROOT / "prices_wide_close.parquet").sort_index()
    existing_open = pd.read_parquet(ROOT / "prices_wide_open.parquet").sort_index()
    existing_close.index = pd.to_datetime(existing_close.index)
    existing_open.index = pd.to_datetime(existing_open.index)
    symbols = [str(c) for c in existing_close.columns if str(c) != "GOLDBEES"]
    panel = download_adjusted_ohlc(symbols, args.start, args.end)
    panel = panel.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])

    new_close = panel.pivot(index="date", columns="symbol", values="close").sort_index()
    new_open = panel.pivot(index="date", columns="symbol", values="open").sort_index()
    # Preserve every already-published historical cell for reproducible saved
    # backtests. Use fresh Yahoo downloads only for dates strictly after the
    # current dataset's last date; even filling old NaNs can move net/cost-tax
    # results because executions fall back from open to close when open is absent.
    last_existing = max(existing_close.index.max(), existing_open.index.max())
    new_close = new_close[new_close.index > last_existing]
    new_open = new_open[new_open.index > last_existing]
    columns = sorted(set(existing_close.columns) | set(new_close.columns))
    close = pd.concat(
        [existing_close.reindex(columns=columns), new_close.reindex(columns=columns)]
    ).sort_index()
    open_wide = pd.concat(
        [existing_open.reindex(columns=columns), new_open.reindex(columns=columns)]
    ).sort_index().reindex(index=close.index, columns=close.columns)

    close.to_parquet(ROOT / "prices_wide_close.parquet")
    open_wide.to_parquet(ROOT / "prices_wide_open.parquet")
    long_close = (
        close.stack()
        .rename("close")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "symbol"})
        .dropna(subset=["close"])
    )
    long_close["source"] = "yahoo_adjusted"
    long_close.to_parquet(ROOT / "prices.parquet")
    long_ohlc = (
        pd.concat({"open": open_wide.stack(), "close": close.stack()}, axis=1)
        .dropna()
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "symbol"})
    )
    long_ohlc.to_parquet(ROOT / "prices_ohlc.parquet")

    download_index("^CRSLDX", ROOT / "nifty500.csv", ROOT / "nifty500_ohlc.csv")
    download_index("^NSEI", ROOT / "nifty50.csv")
    gold = yf.download("GOLDBEES.NS", start=args.start, end=args.end, auto_adjust=True, progress=False, threads=False)
    if not gold.empty:
        close_gold = gold["Close"]
        if isinstance(close_gold, pd.DataFrame):
            close_gold = close_gold.iloc[:, 0]
        lines = ["Date,Close"]
        lines.extend(f"{idx.date().isoformat()},{float(value)}" for idx, value in close_gold.dropna().items())
        (ROOT / "goldbees.csv").write_text("\n".join(lines) + "\n")

    print(f"Updated {len(panel):,} rows across {panel['symbol'].nunique()} symbols.")


if __name__ == "__main__":
    main()

from __future__ import annotations

import itertools
import json
import math
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from universe import build_universe_timeline, get_universe_as_of


ROOT = Path(__file__).resolve().parent
TRADING_DAYS_PER_MONTH = 21
LOOKBACK_OPTIONS = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "9m": 189,
    "12m": 252,
}

DEFAULTS = {
    "start": "2016-05-01",
    "end": "2026-07-01",
    "lookbacks": ["3m", "6m", "9m"],
    "weights": {"3m": 1.0, "6m": 1.0, "9m": 1.0},
    "vol_adjust": True,
    "vol_lookback": "3m",
    "skip_1m": False,
    "top_n": 20,
    "entry_rank": 20,
    "exit_rank": 50,
    "rebalance_frequency": "monthly",
    "rebalance_day": 21,
    "weighting": "equal",
    "initial_capital": 1_000_000.0,
    "include_brokerage": True,
    "brokerage_rate": 0.0003,
    "include_stt": True,
    "stt_rate": 0.001,
    "include_sebi": True,
    "sebi_rate": 0.000001,
    "include_stamp": True,
    "stamp_duty_rate": 0.00015,
    "include_slippage": False,
    "slippage_rate": 0.0,
    "include_tax": True,
    "stcg_rate": 0.20,
    "ltcg_rate": 0.125,
    "ltcg_exemption": 125_000.0,
    "long_term_days": 365,
    "trend_filter": "none",
    "trend_source": "nifty500",
    "trend_timeframe": "weekly",
    "supertrend_atr_period": 1,
    "supertrend_multiplier": 2.5,
    "bearish_exposure": 0.0,
    "bearish_asset": "cash",
    "bearish_symbol": "GOLDBEES",
}


# --- Combination sweep -------------------------------------------------------
# A curated set of the dimensions that actually define the strategy's "edge"
# (signal, portfolio construction, trend filter) -- not every config field, and
# never the cost/tax fields, which should reflect reality rather than be tuned
# for a better-looking number. The default selection below is deliberately the
# small "Stage 1" grid (lookback set x skip_1m x vol_adjust, other dimensions
# pinned to a single value = 24 combinations) so opening the tab shows a fast,
# self-contained sweep rather than an accidental multi-thousand-run grid.
SWEEP_LOOKBACK_SETS: dict[str, tuple[list[str], dict[str, float]]] = {
    "3m_6m_9m": (["3m", "6m", "9m"], {"3m": 1.0, "6m": 1.0, "9m": 1.0}),
    "6m_12m": (["6m", "12m"], {"6m": 1.0, "12m": 1.0}),
    "12m": (["12m"], {"12m": 1.0}),
    "6m": (["6m"], {"6m": 1.0}),
    "3m_6m_9m_12m": (["3m", "6m", "9m", "12m"], {"3m": 1.0, "6m": 1.0, "9m": 1.0, "12m": 1.0}),
    "1m_3m_6m": (["1m", "3m", "6m"], {"1m": 1.0, "3m": 1.0, "6m": 1.0}),
}

SWEEP_PARAMS: list[dict] = [
    {"key": "lookback_set", "label": "Lookback set", "options": [
        {"value": k, "label": k.replace("_", "+").upper()} for k in SWEEP_LOOKBACK_SETS
    ]},
    {"key": "skip_1m", "label": "Skip most recent 1M", "options": [
        {"value": False, "label": "No"}, {"value": True, "label": "Yes"},
    ]},
    {"key": "vol_adjust", "label": "Volatility adjust", "options": [
        {"value": True, "label": "On"}, {"value": False, "label": "Off"},
    ]},
    {"key": "top_n", "label": "Holdings (top_n)", "options": [
        {"value": v, "label": str(v)} for v in (10, 15, 20, 30)
    ]},
    {"key": "rebalance_frequency", "label": "Rebalance frequency", "options": [
        {"value": "monthly", "label": "Monthly"}, {"value": "quarterly", "label": "Quarterly"},
    ]},
    {"key": "weighting", "label": "Weighting", "options": [
        {"value": "equal", "label": "Equal"}, {"value": "inverse_vol", "label": "Inverse vol"},
    ]},
    {"key": "trend_filter", "label": "Trend filter", "options": [
        {"value": "none", "label": "Off"}, {"value": "supertrend", "label": "On"},
    ]},
    {"key": "supertrend_multiplier", "label": "Supertrend multiplier", "options": [
        {"value": v, "label": str(v)} for v in (2.0, 2.5, 3.0)
    ]},
    {"key": "trend_timeframe", "label": "Supertrend timeframe", "options": [
        {"value": "weekly", "label": "Weekly"}, {"value": "daily", "label": "Daily"},
    ]},
    {"key": "bearish_asset", "label": "Bearish destination", "options": [
        {"value": "cash", "label": "Cash"}, {"value": "goldbees", "label": "GoldBees"},
    ]},
]

SWEEP_DEFAULT_SELECTION: dict[str, list] = {
    "lookback_set": list(SWEEP_LOOKBACK_SETS.keys()),
    "skip_1m": [False, True],
    "vol_adjust": [True, False],
    "top_n": [20],
    "rebalance_frequency": ["monthly"],
    "weighting": ["equal"],
    "trend_filter": ["none"],
    "supertrend_multiplier": [2.5],
    "trend_timeframe": ["weekly"],
    "bearish_asset": ["cash"],
}

# Today's absolute max (every option in every group checked) is 9,216 -- this
# ceiling exists only to guard against a genuine runaway/malformed request (e.g.
# a future parameter added without updating this), not to cap intentional use.
MAX_SWEEP_COMBOS = 200_000


_PRICE_WIDE: pd.DataFrame | None = None
_OPEN_WIDE: pd.DataFrame | None = None
_TIMELINE: dict | None = None
_TREND_CACHE: dict[str, pd.Series] = {}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    global _PRICE_WIDE, _OPEN_WIDE, _TIMELINE
    if _PRICE_WIDE is None:
        # Prefer pre-pivoted wide parquet files: pandas' pivot() on the 2.1M-row
        # long-format panel has a huge transient memory spike (~650MB, vs a ~23MB
        # result) which OOMs low-memory hosts. Pivoting once locally and reading
        # the wide result directly keeps runtime peak RSS under ~260MB.
        wide_close_path = ROOT / "prices_wide_close.parquet"
        wide_open_path = ROOT / "prices_wide_open.parquet"
        ohlc_path = ROOT / "prices_ohlc.parquet"
        if wide_close_path.exists() and wide_open_path.exists():
            close = pd.read_parquet(wide_close_path)
            open_wide = pd.read_parquet(wide_open_path)
        elif ohlc_path.exists():
            prices = pd.read_parquet(ohlc_path)
            close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
            open_wide = prices.pivot(index="date", columns="symbol", values="open").sort_index()
        else:
            prices = pd.read_parquet(ROOT / "prices.parquet")
            close = prices.pivot(index="date", columns="symbol", values="close").sort_index()
            open_wide = close.copy()  # no open data -> execution falls back to close
        gold_path = ROOT / "goldbees.csv"
        if gold_path.exists():
            gold = read_index_csv(gold_path).rename("GOLDBEES")
            close = close.join(gold, how="outer")
            # A single missing trading day inside GOLDBEES's own history (data
            # gap, not a market holiday -- the stock panel traded that day) made
            # value() treat a fully-invested-in-gold position as worth exactly
            # zero for that one day, fabricating a flash-crash-and-recover that
            # dominated every downstream vol/Sharpe/drawdown calculation. Forward
            # -fill only within its trading history: NaN before its first listed
            # date is untouched (still correctly "not tradable yet").
            close["GOLDBEES"] = close["GOLDBEES"].ffill()
        _PRICE_WIDE = close[close.notna().sum(axis=1) >= 200]
        _OPEN_WIDE = open_wide.reindex(_PRICE_WIDE.index)
    if _TIMELINE is None:
        _TIMELINE = build_universe_timeline(ROOT / "changes.csv")
    return _PRICE_WIDE, _OPEN_WIDE, _TIMELINE


def merged_config(payload: dict) -> dict:
    cfg = json.loads(json.dumps(DEFAULTS))
    for key, value in payload.items():
        if key == "weights" and isinstance(value, dict):
            cfg["weights"].update(value)
        else:
            cfg[key] = value
    cfg["top_n"] = int(cfg["top_n"])
    cfg["entry_rank"] = int(cfg["entry_rank"])
    cfg["exit_rank"] = int(cfg["exit_rank"])
    cfg["rebalance_day"] = int(cfg["rebalance_day"])
    cfg["initial_capital"] = float(cfg["initial_capital"])
    cfg["supertrend_atr_period"] = max(int(cfg["supertrend_atr_period"]), 1)
    cfg["supertrend_multiplier"] = float(cfg["supertrend_multiplier"])
    cfg["bearish_exposure"] = min(max(float(cfg["bearish_exposure"]), 0.0), 1.0)
    return cfg


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


def load_trend_series(source: str) -> tuple[pd.Series | None, str]:
    preferred = ROOT / "nifty500.csv" if source == "nifty500" else ROOT / "nifty50.csv"
    fallback = ROOT / "nifty50.csv"
    path = preferred if preferred.exists() else fallback
    if not path.exists():
        return None, "none"
    key = str(path)
    if key not in _TREND_CACHE:
        _TREND_CACHE[key] = read_index_csv(path)
    label = "Nifty 500" if path.name.lower() == "nifty500.csv" else "Nifty 50 fallback"
    return _TREND_CACHE[key], label


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
    keep = ["date", "open", "high", "low", "close"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna().sort_values("date").drop_duplicates("date").set_index("date")


def load_trend_ohlc(source: str) -> tuple[pd.DataFrame | None, str]:
    preferred = ROOT / "nifty500_ohlc.csv" if source == "nifty500" else ROOT / "nifty50_ohlc.csv"
    close_path = ROOT / "nifty500.csv" if source == "nifty500" else ROOT / "nifty50.csv"
    fallback = ROOT / "nifty50.csv"
    if preferred.exists():
        label = "Nifty 500" if preferred.name.lower().startswith("nifty500") else "Nifty 50"
        return read_ohlc_csv(preferred), label
    series, label = load_trend_series(source)
    if series is None and fallback.exists() and close_path != fallback:
        series, label = load_trend_series("nifty50")
    if series is None:
        return None, "missing"
    # Fallback for old close-only CSVs: usable for plumbing, but less accurate.
    df = pd.DataFrame({"open": series, "high": series, "low": series, "close": series})
    return df, f"{label} close-only"


def resample_ohlc(ohlc: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "daily":
        return ohlc
    # Mon-Sun calendar weeks (matches TradingView) so NSE weekend special
    # sessions (e.g. budget-Saturday 2020-02-01) count as that week's close
    # rather than falling into the next week as a W-FRI anchor would.
    return ohlc.resample("W-SUN").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()


def supertrend_bullish(ohlc: pd.DataFrame, atr_period: int, multiplier: float) -> pd.Series:
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean()
    hl2 = (high + low) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    bullish = pd.Series(True, index=ohlc.index)

    for i in range(1, len(ohlc)):
        if pd.isna(atr.iloc[i]):
            bullish.iloc[i] = bullish.iloc[i - 1]
            continue
        if basic_upper.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]
        if basic_lower.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

        if close.iloc[i] > final_upper.iloc[i - 1]:
            bullish.iloc[i] = True
        elif close.iloc[i] < final_lower.iloc[i - 1]:
            bullish.iloc[i] = False
        else:
            bullish.iloc[i] = bullish.iloc[i - 1]
    return bullish


_TREND_STATE_CACHE: dict = {}


def build_trend_state(price_index: pd.DatetimeIndex, cfg: dict) -> tuple[pd.Series, dict]:
    if cfg["trend_filter"] == "none":
        state = pd.Series(True, index=price_index)
        return state, {
            "enabled": False,
            "source": "None",
            "bearish_days": 0,
            "switches": 0,
            "current_mode": "Bullish",
            "changes": [],
        }

    # The result depends only on (trend_source, trend_timeframe, atr_period,
    # multiplier) -- NOT on start/end/lookbacks/etc -- so within a sweep that
    # varies unrelated dimensions across e.g. 4,608 trend_filter=on combos,
    # there are only a handful of truly distinct results (2 timeframes x 3
    # multipliers = 6 for the default sweep grid). Re-reading+reparsing the
    # OHLC CSV and re-running supertrend_bullish's O(n) loop plus the holiday
    # relabelling loop for every combo was measured at ~100ms/call, almost all
    # of it wasted recomputation of an identical prior result. (Cache is
    # per-process: under multiprocessing each worker builds its own copy, but
    # still collapses ~1000+ redundant calls per worker down to ~6.)
    cache_key = (cfg["trend_source"], cfg["trend_timeframe"], int(cfg["supertrend_atr_period"]), float(cfg["supertrend_multiplier"]), id(price_index))
    if cache_key in _TREND_STATE_CACHE:
        cached_state, cached_summary = _TREND_STATE_CACHE[cache_key]
        return cached_state, dict(cached_summary)

    ohlc, source_used = load_trend_ohlc(cfg["trend_source"])
    if ohlc is None or ohlc.empty:
        state = pd.Series(True, index=price_index)
        return state, {
            "enabled": False,
            "source": "missing",
            "bearish_days": 0,
            "switches": 0,
            "current_mode": "Bullish",
            "changes": [],
            "warning": "No trend OHLC CSV found.",
        }
    st_ohlc = resample_ohlc(ohlc, cfg["trend_timeframe"])
    raw_state = supertrend_bullish(
        st_ohlc,
        int(cfg["supertrend_atr_period"]),
        float(cfg["supertrend_multiplier"]),
    )
    # Weekly bars are labelled by calendar Friday (W-FRI). When that Friday is
    # a market holiday (e.g. 2020-05-01, 2025-04-18) the label is absent from
    # the trading calendar, so a plain ffill can't apply the new state until the
    # following Monday -- pushing the reported flip a week off the real close.
    # Re-stamp each bar to its actual last trading day so holiday weeks behave
    # exactly like normal weeks (normal Fridays are unchanged).
    if cfg["trend_timeframe"] != "daily" and len(price_index):
        relabelled = []
        for bar_end in raw_state.index:
            prior = price_index[price_index <= bar_end]
            relabelled.append(prior[-1] if len(prior) else bar_end)
        raw_state = pd.Series(raw_state.values, index=pd.DatetimeIndex(relabelled))
        raw_state = raw_state[~raw_state.index.duplicated(keep="last")].sort_index()
    trend_state = raw_state.reindex(price_index, method="ffill").fillna(True).astype(bool)
    switches = int((trend_state.astype(int).diff().abs() == 1).sum())
    summary = {
        "enabled": True,
        "source": source_used,
        "condition": f"{cfg['trend_timeframe']} supertrend",
        "bearish_days": int((~trend_state).sum()),
        "switches": switches,
        "current_mode": "Bullish" if bool(trend_state.iloc[-1]) else "Bearish",
        "changes": trend_changes(trend_state),
    }
    _TREND_STATE_CACHE[cache_key] = (trend_state, summary)
    return trend_state, dict(summary)


def trend_changes(trend_state: pd.Series) -> list[dict]:
    if trend_state.empty:
        return []
    rows = []
    previous = bool(trend_state.iloc[0])
    for date, value in trend_state.iloc[1:].items():
        current = bool(value)
        if current != previous:
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "mode": "Bullish" if current else "Bearish",
                    "previous": "Bullish" if previous else "Bearish",
                }
            )
            previous = current
    return rows


def trend_action_dates(index: pd.DatetimeIndex, trend_state: pd.Series, start: str, end: str) -> tuple[list[pd.Timestamp], dict[str, dict]]:
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    action_dates = []
    action_info: dict[str, dict] = {}
    for row in trend_changes(trend_state):
        signal_date = pd.to_datetime(row["date"])
        if signal_date < start_ts or signal_date > end_ts:
            continue
        candidates = index[index > signal_date]
        if not len(candidates):
            continue
        action_date = candidates[0]
        if action_date > end_ts:
            continue
        action_dates.append(action_date)
        action_info[action_date.date().isoformat()] = {
            "signal_date": row["date"],
            "action_date": action_date.date().isoformat(),
            "mode": row["mode"],
            "previous": row["previous"],
        }
    return action_dates, action_info


def get_rebalance_dates(index: pd.DatetimeIndex, start: str, end: str, frequency: str, day: int) -> list[pd.Timestamp]:
    start_ts, end_ts = pd.to_datetime(start), pd.to_datetime(end)
    if frequency == "weekly":
        anchors = pd.date_range(start_ts, end_ts, freq="W-MON")
    elif frequency == "quarterly":
        anchors = pd.date_range(start_ts, end_ts, freq="QS") + pd.Timedelta(days=max(day - 1, 0))
    else:
        anchors = pd.date_range(start_ts, end_ts, freq="MS") + pd.Timedelta(days=max(day - 1, 0))

    dates = []
    for target in anchors:
        candidates = index[index >= target]
        if len(candidates) and candidates[0] <= end_ts:
            dates.append(candidates[0])
    return sorted(set(dates))


def precompute_return_panels(price_wide: pd.DataFrame, cfg: dict) -> dict:
    """Vectorized once-per-backtest precompute of the lookback returns that
    compute_scores previously recomputed from a fresh window slice on every single
    rebalance date (100+ times per run). Same values, computed once across the
    full panel instead of once per rebalance -- pure speedup, no behavior change.

    Built via raw numpy (float64, same precision as before -- no ranking-relevant
    rounding introduced) rather than pandas' pct_change(): pandas' rolling/
    pct_change ops on a 729-column panel carry ~3-4x their result size in
    transient memory (internal upcasting/temporaries), which is what caused an
    OOM crash on a 512MB host the first time this was tried with pandas ops
    end-to-end. (float32 was tried and rejected: it shaves more memory but
    introduces up to ~1% relative score differences -- enough to flip which
    symbols land in the top-N ranking near the cutoff, silently changing
    backtest results.) Volatility is deliberately NOT precomputed the same way --
    it's cheap per-call on a small window already (see compute_scores), and
    precomputing it via pandas' rolling().std() alone added ~90MB of transient
    memory for a 22MB result, without a matching speed payoff.
    """
    lookbacks = [lb for lb in cfg["lookbacks"] if lb in LOOKBACK_OPTIONS]
    if not lookbacks:
        lookbacks = ["3m", "6m", "9m"]
    days_needed = {LOOKBACK_OPTIONS[lb] for lb in lookbacks}

    idx, cols = price_wide.index, price_wide.columns
    arr = price_wide.to_numpy(dtype=np.float64, copy=True)
    n, m = arr.shape
    ret_full = {}
    for days in days_needed:
        out = np.full((n, m), np.nan, dtype=np.float64)
        np.divide(arr[days:], arr[:-days], out=out[days:])
        out[days:] -= 1.0
        ret_full[days] = pd.DataFrame(out, index=idx, columns=cols, copy=False)

    return {"lookbacks": lookbacks, "ret_full": ret_full}


def compute_scores(price_wide: pd.DataFrame, date: pd.Timestamp, universe: set[str], cfg: dict, panels: dict) -> tuple[pd.Series, pd.Series]:
    if date not in price_wide.index:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    lookbacks = panels["lookbacks"]
    max_lb = max(LOOKBACK_OPTIONS[lb] for lb in lookbacks + [cfg["vol_lookback"]])
    if cfg["skip_1m"]:
        max_lb += LOOKBACK_OPTIONS["1m"]
    loc = price_wide.index.get_loc(date)
    if loc < max_lb:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    cols = [c for c in price_wide.columns if c in universe]
    p_now_loc = loc - (LOOKBACK_OPTIONS["1m"] if cfg["skip_1m"] else 0)
    p_now = price_wide.iloc[p_now_loc][cols]
    score = pd.Series(0.0, index=cols)
    total_weight = 0.0
    valid = p_now.notna()
    for lb in lookbacks:
        days = LOOKBACK_OPTIONS[lb]
        ret = panels["ret_full"][days].iloc[p_now_loc][cols]
        p_old = price_wide.iloc[p_now_loc - days][cols]
        weight = float(cfg["weights"].get(lb, 1.0))
        score = score.add(ret.fillna(0) * weight, fill_value=0)
        valid &= p_old.notna()
        total_weight += abs(weight)
    if total_weight:
        score = score / total_weight

    # Volatility is computed directly from a small window slice (matches the
    # original per-rebalance calculation exactly), not from a full-panel
    # precompute -- see the docstring on precompute_return_panels for why.
    vol_days = LOOKBACK_OPTIONS.get(cfg["vol_lookback"], 63)
    vol_window = price_wide.iloc[loc - vol_days + 1: loc + 1][cols].pct_change().dropna(how="all")
    vol = vol_window.std() * np.sqrt(252)
    if cfg["vol_adjust"]:
        valid &= vol.notna() & (vol > 0)
        score = score / vol

    return score[valid].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False), vol


def metric_block(equity: pd.Series) -> dict:
    returns = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = returns.std() * np.sqrt(252)
    drawdown = equity / equity.cummax() - 1
    return {
        "cagr": float(round(cagr * 100, 2)),
        "vol": float(round(vol * 100, 2)),
        "sharpe": float(round((returns.mean() * 252) / vol, 2)) if vol > 0 else 0.0,
        "maxdd": float(round(drawdown.min() * 100, 2)),
        "final": float(round(equity.iloc[-1], 2)),
    }


def equity_points(series: pd.Series) -> list[list]:
    return [[d.date().isoformat(), round(float(v), 2)] for d, v in series.items()]


def drawdown_points(equity: pd.Series) -> list[list]:
    dd = (equity / equity.cummax() - 1) * 100
    return equity_points(dd)


def monthly_returns(equity: pd.Series) -> dict:
    monthly = equity.resample("ME").last().pct_change().dropna() * 100
    out: dict[str, dict[str, float]] = {}
    for date, value in monthly.items():
        out.setdefault(str(date.year), {})[str(date.month)] = round(float(value), 1)
    return out


def yearly_returns(equity: pd.Series) -> list[list]:
    yearly = equity.resample("YE").last().pct_change().dropna() * 100
    return [[int(date.year), round(float(value), 1)] for date, value in yearly.items()]


def benchmark_equity(index: pd.DatetimeIndex, source: str = "nifty500") -> tuple[pd.Series | None, dict]:
    series, label = load_trend_series(source)
    if series is None or series.empty:
        return None, {"label": label, "warning": "Benchmark CSV not found."}
    aligned = series.reindex(index, method="ffill").dropna()
    if aligned.empty:
        return None, {"label": label, "warning": "Benchmark has no overlap with backtest window."}
    equity = aligned / aligned.iloc[0] * 100
    return equity, {"label": label}


def financial_year(date: pd.Timestamp) -> str:
    start = date.year if date.month >= 4 else date.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def tax_for_gain(gain: float, is_long: bool, date: pd.Timestamp, state: dict, cfg: dict) -> float:
    if not cfg["include_tax"]:
        return 0.0
    fy = financial_year(date)
    state["ltcg_used"].setdefault(fy, 0.0)
    if gain <= 0:
        state["lt_loss" if is_long else "st_loss"] += -gain
        return 0.0
    taxable = gain
    if is_long:
        offset = min(taxable, state["lt_loss"])
        taxable -= offset
        state["lt_loss"] -= offset
        offset = min(taxable, state["st_loss"])
        taxable -= offset
        state["st_loss"] -= offset
        exempt_left = max(float(cfg["ltcg_exemption"]) - state["ltcg_used"][fy], 0.0)
        exempt = min(taxable, exempt_left)
        taxable -= exempt
        state["ltcg_used"][fy] += exempt
        return taxable * float(cfg["ltcg_rate"])
    offset = min(taxable, state["st_loss"])
    taxable -= offset
    state["st_loss"] -= offset
    return taxable * float(cfg["stcg_rate"])


def charge_rates(cfg: dict) -> tuple[float, float]:
    common = 0.0
    common += float(cfg["brokerage_rate"]) if cfg["include_brokerage"] else 0.0
    common += float(cfg["stt_rate"]) if cfg["include_stt"] else 0.0
    common += float(cfg["sebi_rate"]) if cfg["include_sebi"] else 0.0
    common += float(cfg["slippage_rate"]) if cfg["include_slippage"] else 0.0
    buy = common + (float(cfg["stamp_duty_rate"]) if cfg["include_stamp"] else 0.0)
    return buy, common


def run_dashboard_backtest(cfg: dict) -> dict:
    price_wide, open_wide, timeline = load_inputs()
    scheduled_dates = get_rebalance_dates(price_wide.index, cfg["start"], cfg["end"], cfg["rebalance_frequency"], cfg["rebalance_day"])
    dates = scheduled_dates
    if len(dates) < 2:
        raise ValueError("Not enough rebalance dates for selected period.")
    trend_state, trend_summary = build_trend_state(price_wide.index, cfg)
    trend_event_dates, trend_event_info = trend_action_dates(price_wide.index, trend_state, cfg["start"], cfg["end"])
    if trend_summary.get("enabled"):
        signal_to_action = {info["signal_date"]: info["action_date"] for info in trend_event_info.values()}
        dates = sorted(set(scheduled_dates + trend_event_dates))
        trend_summary["changes"] = [
            {
                **row,
                "action_date": signal_to_action.get(row["date"]),
            }
            for row in trend_summary.get("changes", [])
            if pd.to_datetime(cfg["start"]) <= pd.to_datetime(row["date"]) <= pd.to_datetime(cfg["end"])
        ]
    warnings = []

    current = set()
    gross_returns = []
    holdings_rows = []
    change_rows = []
    cash = float(cfg["initial_capital"])
    positions: dict[str, list[dict]] = {}
    net_points = []
    contributor_returns: dict[str, float] = {}
    total_charges = 0.0
    total_tax = 0.0
    tax_state = {"st_loss": 0.0, "lt_loss": 0.0, "ltcg_used": {}}
    buy_rate, sell_rate = charge_rates(cfg)

    def shares(symbol: str) -> float:
        return sum(lot["shares"] for lot in positions.get(symbol, []))

    def value(symbol: str, date: pd.Timestamp) -> float:
        if symbol not in price_wide.columns or pd.isna(price_wide.at[date, symbol]):
            return 0.0
        return shares(symbol) * float(price_wide.at[date, symbol])

    def total_value(date: pd.Timestamp) -> float:
        return cash + sum(value(symbol, date) for symbol in positions)

    def exec_price(symbol: str, date: pd.Timestamp) -> float | None:
        # Trades fill at the rebalance-day OPEN; fall back to close where open is
        # unavailable (e.g. GOLDBEES, or any symbol missing from the OHLC panel).
        if symbol in open_wide.columns and date in open_wide.index:
            px = open_wide.at[date, symbol]
            if pd.notna(px):
                return float(px)
        if symbol in price_wide.columns and pd.notna(price_wide.at[date, symbol]):
            return float(price_wide.at[date, symbol])
        return None

    def sell(symbol: str, date: pd.Timestamp, shares_to_sell: float) -> None:
        nonlocal cash, total_charges, total_tax
        if shares_to_sell <= 0 or symbol not in positions:
            return
        price = exec_price(symbol, date)
        if price is None:
            return
        remaining = min(shares_to_sell, shares(symbol))
        sale, tax = 0.0, 0.0
        kept = []
        for lot in positions[symbol]:
            take = min(lot["shares"], remaining)
            if take > 0:
                frac = take / lot["shares"]
                lot_sale = take * price
                lot_cost = lot["cost"] * frac
                sale += lot_sale
                tax += tax_for_gain(
                    lot_sale - lot_cost,
                    (date - lot["entry"]).days >= int(cfg["long_term_days"]),
                    date,
                    tax_state,
                    cfg,
                )
                lot["shares"] -= take
                lot["cost"] *= 1 - frac
                remaining -= take
            if lot["shares"] > 1e-10:
                kept.append(lot)
        if kept:
            positions[symbol] = kept
        else:
            positions.pop(symbol, None)
        charges = sale * sell_rate
        cash += sale - charges - tax
        total_charges += charges
        total_tax += tax

    def buy(symbol: str, date: pd.Timestamp, amount: float) -> None:
        nonlocal cash, total_charges
        price = exec_price(symbol, date)
        if amount <= 0 or price is None or price <= 0:
            return
        gross = min(cash, amount) / (1 + buy_rate)
        charges = gross * buy_rate
        if gross <= 0:
            return
        cash -= gross + charges
        total_charges += charges
        positions.setdefault(symbol, []).append(
            {"shares": gross / price, "cost": gross, "entry": date}
        )

    panels = precompute_return_panels(price_wide, cfg)
    for i, date in enumerate(dates):
        action_key = date.date().isoformat()
        is_trend_event = action_key in trend_event_info
        universe, approximated = get_universe_as_of(date, timeline)
        # Rank on the PREVIOUS close (data known before the entry-day open), then
        # enter at this day's open -- no look-ahead.
        loc = price_wide.index.get_loc(date)
        rank_date = price_wide.index[loc - 1] if loc > 0 else date
        scores, vols = compute_scores(price_wide, rank_date, universe, cfg, panels)
        ranks = {sym: n + 1 for n, sym in enumerate(scores.index)}
        is_bullish = bool(trend_state.loc[date]) if date in trend_state.index else True
        gold_symbol = str(cfg.get("bearish_symbol", "GOLDBEES")).upper()
        use_gold = (not is_bullish) and cfg.get("bearish_asset") == "goldbees" and gold_symbol in price_wide.columns
        if (not is_bullish) and cfg.get("bearish_asset") == "goldbees" and gold_symbol not in price_wide.columns:
            warning = f"{gold_symbol} data not found; bearish periods stayed in cash."
            if warning not in warnings:
                warnings.append(warning)
        exposure = 1.0 if is_bullish or use_gold else float(cfg["bearish_exposure"])
        survivors = {s for s in current if ranks.get(s, 999999) <= int(cfg["exit_rank"])}
        if is_bullish:
            candidates = [s for s in scores.index[: int(cfg["entry_rank"])] if s not in survivors]
            new_adds = candidates[: max(int(cfg["top_n"]) - len(survivors), 0)]
            target = set(list(survivors) + list(new_adds))
        elif use_gold:
            new_adds = [gold_symbol] if gold_symbol not in current else []
            target = {gold_symbol}
        else:
            new_adds = []
            target = survivors if exposure > 0 else set()

        for symbol in sorted(target - current):
            change_rows.append({"date": date.date().isoformat(), "action": "ENTER", "symbol": symbol, "rank": ranks.get(symbol)})
        for symbol in sorted(current - target):
            change_rows.append({"date": date.date().isoformat(), "action": "EXIT", "symbol": symbol, "rank": ranks.get(symbol)})

        current = target
        if not current:
            weights = pd.Series(dtype=float)
        elif cfg["weighting"] == "score":
            raw = scores.reindex(list(current)).clip(lower=0)
            weights = raw / raw.sum() if raw.sum() > 0 else pd.Series(1 / len(current), index=list(current))
        elif cfg["weighting"] == "inverse_vol":
            raw = 1 / vols.reindex(list(current)).replace(0, np.nan)
            raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
            weights = raw / raw.sum() if raw.sum() > 0 else pd.Series(1 / len(current), index=list(current))
        else:
            weights = pd.Series(1 / len(current), index=list(current))
        weights = weights * exposure

        for symbol in sorted(set(positions) - current):
            sell(symbol, date, shares(symbol))
        account_value = total_value(date)
        for symbol in sorted(current):
            target_value = account_value * float(weights.get(symbol, 0.0))
            cur_value = value(symbol, date)
            if cur_value > target_value * 1.01:
                sell(symbol, date, (cur_value - target_value) / float(price_wide.at[date, symbol]))
        account_value = total_value(date)
        for symbol in sorted(current):
            target_value = account_value * float(weights.get(symbol, 0.0))
            cur_value = value(symbol, date)
            if cur_value < target_value * 0.99:
                buy(symbol, date, target_value - cur_value)

        next_date = dates[i + 1] if i + 1 < len(dates) else price_wide.index[price_wide.index <= pd.to_datetime(cfg["end"])][-1]
        if current:
            period = price_wide.loc[date:next_date, list(current)]
            period_rets = period.pct_change().iloc[1:]
            period_weights = weights.reindex(period.columns).fillna(0)
            weighted = period_rets.mul(period_weights, axis=1)
            gross_returns.append(weighted.sum(axis=1))
            for symbol, contribution in weighted.sum().items():
                contributor_returns[symbol] = contributor_returns.get(symbol, 0.0) + float(contribution)
        else:
            days = price_wide.loc[(price_wide.index > date) & (price_wide.index <= next_date)].index
            gross_returns.append(pd.Series(0.0, index=days))
        # Cash and share counts are fixed for the whole holding period (no trades
        # occur until the next rebalance), so the daily mark-to-market can be one
        # matrix multiply instead of calling total_value() per day per holding
        # (previously ~51,000 scalar .at[] lookups across a full backtest).
        period_index = price_wide.loc[date:next_date].index
        held_symbols = [s for s in positions if s in price_wide.columns]
        if held_symbols:
            shares_vec = pd.Series({s: shares(s) for s in held_symbols})
            position_value = price_wide.loc[period_index, held_symbols].mul(shares_vec, axis=1).sum(axis=1)
        else:
            position_value = pd.Series(0.0, index=period_index)
        net_points.extend(zip(period_index, cash + position_value))

        holdings_rows.append(
            {
                "date": date.date().isoformat(),
                "holdings": sorted(current),
                "universe_size": len(universe),
                "approximated": bool(approximated),
                "trend_mode": "Bullish" if is_bullish else "Bearish",
                "exposure": exposure,
                "bearish_asset": gold_symbol if use_gold else ("Cash" if not is_bullish and exposure == 0 else "Momentum"),
                "action_type": "Trend switch" if is_trend_event else "Scheduled rebalance",
                "signal_date": trend_event_info.get(action_key, {}).get("signal_date"),
            }
        )

    gross_ret = pd.concat(gross_returns).sort_index()
    gross_ret = gross_ret[~gross_ret.index.duplicated(keep="first")]
    gross_equity = 100 * (1 + gross_ret.fillna(0)).cumprod()
    net_value = pd.Series([v for _, v in net_points], index=pd.DatetimeIndex([d for d, _ in net_points]))
    net_value = net_value[~net_value.index.duplicated(keep="last")].sort_index()
    net_equity = net_value / float(cfg["initial_capital"]) * 100
    benchmark, benchmark_summary = benchmark_equity(net_equity.index, "nifty500")
    contributors = sorted(
        ((symbol, round(value * 100, 1)) for symbol, value in contributor_returns.items()),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "gross_metrics": metric_block(gross_equity),
        "net_metrics": metric_block(net_equity),
        "gross_equity": equity_points(gross_equity),
        "net_equity": equity_points(net_equity),
        "drawdown": {
            "gross": drawdown_points(gross_equity),
            "net": drawdown_points(net_equity),
            "nifty500": drawdown_points(benchmark) if benchmark is not None else [],
        },
        "benchmark": {
            "label": benchmark_summary["label"],
            "metrics": metric_block(benchmark) if benchmark is not None else {},
            "equity": equity_points(benchmark) if benchmark is not None else [],
            "drawdown": drawdown_points(benchmark) if benchmark is not None else [],
            "warning": benchmark_summary.get("warning"),
        },
        "monthly": monthly_returns(net_equity),
        "yearly": yearly_returns(net_equity),
        "contributors": {
            "top": contributors[:12],
            "bottom": sorted(contributors, key=lambda item: item[1])[:12],
        },
        "summary": {
            "rebalances": len(holdings_rows),
            "scheduled_rebalances": len(scheduled_dates),
            "trend_actions": len(trend_event_dates),
            "entries_exits": len(change_rows),
            "total_charges": round(total_charges, 2),
            "total_tax": round(total_tax, 2),
            "net_final_value": round(float(net_value.iloc[-1]), 2),
            "st_loss_carry": round(tax_state["st_loss"], 2),
            "lt_loss_carry": round(tax_state["lt_loss"], 2),
            "trend": trend_summary,
            "warnings": warnings,
        },
        "latest_holdings": holdings_rows[-1]["holdings"] if holdings_rows else [],
        "changes": change_rows,
        "recent_changes": change_rows[-80:],
    }


# --- Combination sweep job runner --------------------------------------------
# One sweep runs at a time (personal-use tool, no auth). State lives in a plain
# module-level dict guarded by a lock; a background thread drives a small
# ThreadPoolExecutor over the combos (numpy/pandas release the GIL during the
# heavy array ops, so this gets real parallelism on a multi-core dev machine;
# on a throttled single-core host it just runs sequentially without penalty).
_SWEEP_LOCK = threading.Lock()
_SWEEP_STATE: dict = {
    "job_id": None,
    "running": False,
    "total": 0,
    "completed": 0,
    "started_at": None,
    "finished_at": None,
    "results": [],
    "cancel": False,
    "error": None,
    "base": None,
}
SWEEP_RESULTS_PATH = ROOT / "sweep_results.json"


def expand_sweep_combos(selection: dict) -> list[dict]:
    keys = [p["key"] for p in SWEEP_PARAMS if selection.get(p["key"])]
    value_lists = [selection[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def combo_to_overrides(combo: dict) -> dict:
    overrides = {}
    for key, value in combo.items():
        if key == "lookback_set":
            lookbacks, weights = SWEEP_LOOKBACK_SETS[value]
            overrides["lookbacks"] = lookbacks
            overrides["weights"] = weights
        else:
            overrides[key] = value
    return overrides


def run_one_combo(base_payload: dict, combo: dict) -> dict:
    row = dict(combo)
    try:
        cfg = merged_config({**base_payload, **combo_to_overrides(combo)})
        result = run_dashboard_backtest(cfg)
        row.update({
            "cagr": result["net_metrics"]["cagr"],
            "gross_cagr": result["gross_metrics"]["cagr"],
            "sharpe": result["net_metrics"]["sharpe"],
            "vol": result["net_metrics"]["vol"],
            "maxdd": result["net_metrics"]["maxdd"],
            "gross_maxdd": result["gross_metrics"]["maxdd"],
            "final": result["net_metrics"]["final"],
            "rebalances": result["summary"]["rebalances"],
        })
    except Exception as exc:
        row["error"] = str(exc)
    return row


def sweep_worker_count() -> int:
    # Measured directly: threads give ZERO real speedup here (6 threads was
    # slightly SLOWER than plain sequential, 7.55s vs 6.71s for 6 identical
    # backtests) -- the per-rebalance simulation loop (trade/tax-lot bookkeeping)
    # is pure Python and holds the GIL for nearly the whole runtime, so "workers"
    # just take turns rather than running concurrently. On the hosted 512MB host
    # we stay at 1 worker anyway (memory, not CPU, is the constraint there -- see
    # sweep_executor()). Locally, real parallelism requires separate processes
    # (each with its own GIL) -- see sweep_executor().
    if os.environ.get("PORT"):
        return 1
    return max(1, (os.cpu_count() or 4) - 1)  # leave one core for the HTTP server


def sweep_executor_class():
    # Threads on the hosted host: a full process per worker would duplicate the
    # ~300MB baseline (price panels, pandas/numpy import) per worker, which is
    # exactly the OOM mechanism already fixed once -- not worth it when hosted
    # mode is pinned to 1 worker anyway. Processes locally: each gets its own
    # GIL, so this is where the real parallelism actually comes from; local
    # machines have enough RAM (verified 16GB here) that N x ~300MB is a
    # non-issue.
    return ThreadPoolExecutor if os.environ.get("PORT") else ProcessPoolExecutor


def run_sweep_job(job_id: str, base_payload: dict, combos: list[dict]) -> None:
    load_inputs()  # warm the shared cache once before fanning out workers
    workers = sweep_worker_count()
    # Not using the executor as a context manager: its __exit__ calls
    # shutdown(wait=True), which blocks until every already-submitted future
    # finishes -- so breaking out of the loop below on cancel wouldn't actually
    # stop anything, it would just stop collecting results while the pool kept
    # grinding through the rest in the background. Explicit shutdown(wait=False,
    # cancel_futures=True) in `finally` actually drops the not-yet-started work.
    executor = sweep_executor_class()(max_workers=workers)
    try:
        futures = {executor.submit(run_one_combo, base_payload, combo): combo for combo in combos}
        for future in as_completed(futures):
            with _SWEEP_LOCK:
                if _SWEEP_STATE["job_id"] != job_id:
                    return  # superseded by a newer job
                if _SWEEP_STATE["cancel"]:
                    break
            row = future.result()
            with _SWEEP_LOCK:
                if _SWEEP_STATE["job_id"] != job_id:
                    return
                _SWEEP_STATE["results"].append(row)
                _SWEEP_STATE["completed"] += 1
    except Exception as exc:
        with _SWEEP_LOCK:
            if _SWEEP_STATE["job_id"] == job_id:
                _SWEEP_STATE["error"] = f"{exc}\n{traceback.format_exc(limit=3)}"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        with _SWEEP_LOCK:
            if _SWEEP_STATE["job_id"] == job_id:
                _SWEEP_STATE["running"] = False
                _SWEEP_STATE["finished_at"] = time.time()


def sweep_status_payload() -> dict:
    with _SWEEP_LOCK:
        state = dict(_SWEEP_STATE)
    elapsed = (state["finished_at"] or time.time()) - state["started_at"] if state["started_at"] else 0.0
    completed, total = state["completed"], state["total"]
    eta = (elapsed / completed) * (total - completed) if completed and total > completed and state["running"] else 0.0
    return {
        "job_id": state["job_id"],
        "running": state["running"],
        "total": total,
        "completed": completed,
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1),
        "cancelled": state["cancel"],
        "error": state["error"],
        "results": state["results"],
    }


def json_safe(obj):
    # Python's json.dumps writes NaN/Infinity as bare (invalid-JSON) tokens by
    # default -- harmless until a browser's strict JSON.parse hits one and
    # aborts on the ENTIRE response, not just the offending field. A backtest
    # over a very short/degenerate equity series (e.g. a sweep combo that ends
    # up with almost no rebalances) can produce NaN vol/sharpe, so this is a
    # real, reachable case, not just a defensive nicety. Recursively replace
    # NaN/Infinity with null (valid JSON, renders as "-" client-side).
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(json_safe(payload)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/dashboard.html"):
            body = (ROOT / "dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/sweep/meta":
            self._send_json(200, {
                "params": SWEEP_PARAMS,
                "defaults": SWEEP_DEFAULT_SELECTION,
                "max_combos": MAX_SWEEP_COMBOS,
            })
            return
        if path == "/api/sweep/status":
            self._send_json(200, sweep_status_payload())
            return
        if path == "/api/sweep/saved":
            if SWEEP_RESULTS_PATH.exists():
                self._send_json(200, json.loads(SWEEP_RESULTS_PATH.read_text()))
            else:
                self._send_json(200, {"results": [], "saved_at": None})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/backtest":
                payload = self._read_json_body()
                result = run_dashboard_backtest(merged_config(payload))
                self._send_json(200, result)
                return

            if path == "/api/sweep/start":
                payload = self._read_json_body()
                base_payload = payload.get("base", {})
                selection = payload.get("selection", {})
                combos = expand_sweep_combos(selection)
                if not combos:
                    self._send_json(400, {"error": "Select at least one value for every sweep parameter."})
                    return
                if len(combos) > MAX_SWEEP_COMBOS:
                    self._send_json(400, {"error": f"{len(combos)} combinations exceeds the {MAX_SWEEP_COMBOS} cap. Narrow your selection."})
                    return
                with _SWEEP_LOCK:
                    if _SWEEP_STATE["running"]:
                        self._send_json(409, {"error": "A sweep is already running. Cancel it first."})
                        return
                    job_id = uuid.uuid4().hex
                    _SWEEP_STATE.update({
                        "job_id": job_id, "running": True, "total": len(combos), "completed": 0,
                        "started_at": time.time(), "finished_at": None, "results": [],
                        "cancel": False, "error": None, "base": base_payload,
                    })
                threading.Thread(target=run_sweep_job, args=(job_id, base_payload, combos), daemon=True).start()
                self._send_json(200, {"job_id": job_id, "total": len(combos)})
                return

            if path == "/api/sweep/cancel":
                with _SWEEP_LOCK:
                    _SWEEP_STATE["cancel"] = True
                self._send_json(200, {"stopped": True})
                return

            if path == "/api/sweep/save":
                with _SWEEP_LOCK:
                    results = list(_SWEEP_STATE["results"])
                    saved = {
                        "saved_at": time.time(),
                        "total": _SWEEP_STATE["total"],
                        "completed": len(results),
                        "base": _SWEEP_STATE["base"],
                        "results": results,
                    }
                if not results:
                    self._send_json(400, {"error": "No completed sweep results to save yet."})
                    return
                SWEEP_RESULTS_PATH.write_text(json.dumps(json_safe(saved)))
                self._send_json(200, {"saved": True, "completed": len(results)})
                return

            self.send_error(404)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def main() -> None:
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    port = int(os.environ.get("PORT", 8765))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
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
    "trend_condition": "close_above_ma",
    "trend_ma_type": "sma",
    "trend_ma_days": 200,
    "trend_fast_ma_days": 50,
    "trend_timeframe": "weekly",
    "supertrend_atr_period": 1,
    "supertrend_multiplier": 2.5,
    "trend_buffer_pct": 0.0,
    "trend_confirmation_days": 1,
    "bearish_exposure": 0.0,
    "bearish_asset": "cash",
    "bearish_symbol": "GOLDBEES",
}


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
    cfg["trend_ma_days"] = int(cfg["trend_ma_days"])
    cfg["trend_fast_ma_days"] = int(cfg["trend_fast_ma_days"])
    cfg["supertrend_atr_period"] = max(int(cfg["supertrend_atr_period"]), 1)
    cfg["supertrend_multiplier"] = float(cfg["supertrend_multiplier"])
    cfg["trend_confirmation_days"] = max(int(cfg["trend_confirmation_days"]), 1)
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


def moving_average(series: pd.Series, days: int, ma_type: str) -> pd.Series:
    if ma_type == "ema":
        return series.ewm(span=days, min_periods=days, adjust=False).mean()
    return series.rolling(days, min_periods=days).mean()


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

    if cfg["trend_condition"] == "weekly_supertrend":
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
        return trend_state, {
            "enabled": True,
            "source": source_used,
            "condition": f"{cfg['trend_timeframe']} supertrend",
            "bearish_days": int((~trend_state).sum()),
            "switches": switches,
            "current_mode": "Bullish" if bool(trend_state.iloc[-1]) else "Bearish",
            "changes": trend_changes(trend_state),
        }

    trend, source_used = load_trend_series(cfg["trend_source"])
    if trend is None or trend.empty:
        state = pd.Series(True, index=price_index)
        return state, {
            "enabled": False,
            "source": "missing",
            "bearish_days": 0,
            "switches": 0,
            "current_mode": "Bullish",
            "changes": [],
            "warning": "No trend index CSV found.",
        }

    slow = moving_average(trend, int(cfg["trend_ma_days"]), cfg["trend_ma_type"])
    fast = moving_average(trend, int(cfg["trend_fast_ma_days"]), cfg["trend_ma_type"])
    buffer = float(cfg["trend_buffer_pct"]) / 100.0
    if cfg["trend_condition"] == "fast_above_slow":
        raw_bull = fast > slow * (1 + buffer)
        raw_bear = fast < slow * (1 - buffer)
    elif cfg["trend_condition"] == "close_above_ma_and_ma_rising":
        raw_bull = (trend > slow * (1 + buffer)) & (slow.diff() > 0)
        raw_bear = (trend < slow * (1 - buffer)) | (slow.diff() < 0)
    else:
        raw_bull = trend > slow * (1 + buffer)
        raw_bear = trend < slow * (1 - buffer)

    confirm = int(cfg["trend_confirmation_days"])
    if confirm > 1:
        bull = raw_bull.rolling(confirm).sum() >= confirm
        bear = raw_bear.rolling(confirm).sum() >= confirm
    else:
        bull, bear = raw_bull, raw_bear

    regime = []
    mode = True
    for date in trend.index:
        if bool(bull.loc[date]):
            mode = True
        elif bool(bear.loc[date]):
            mode = False
        regime.append(mode)
    trend_state = pd.Series(regime, index=trend.index).reindex(price_index, method="ffill").fillna(True).astype(bool)
    switches = int((trend_state.astype(int).diff().abs() == 1).sum())
    return trend_state, {
        "enabled": True,
        "source": source_used,
        "bearish_days": int((~trend_state).sum()),
        "switches": switches,
        "current_mode": "Bullish" if bool(trend_state.iloc[-1]) else "Bearish",
        "changes": trend_changes(trend_state),
    }


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


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlparse(self.path).path in ("/", "/dashboard.html"):
            body = (ROOT / "dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/backtest":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = run_dashboard_backtest(merged_config(payload))
            body = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            body = json.dumps({"error": str(exc)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> None:
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    port = int(os.environ.get("PORT", 8765))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Dashboard running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

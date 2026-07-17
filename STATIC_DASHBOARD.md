# Static client-side dashboard

This is the free/no-card production path for the strategy dashboard.

## How it works

- `refresh_market_data.py` updates local price/index files from free Yahoo Finance endpoints.
- `build_static_site.py` converts the latest Parquet/CSV data into browser-ready JSON under `site/data/`.
- `site/index.html` runs the backtest in the browser and stores saved runs in IndexedDB.
- `.github/workflows/daily-refresh.yml` refreshes data daily and deploys `site/` to GitHub Pages.

The public app does not need Render, a database, persistent disk, or an always-running backend.

## Local build

```bash
venv/bin/python build_static_site.py
python3 -m http.server 8000 --directory site
```

Open `http://127.0.0.1:8000`.

## Production

Enable GitHub Pages with **GitHub Actions** as the source. The daily workflow publishes the `site/` directory. If a refresh fails, GitHub Pages keeps serving the previous successful deployment.

## Paper trading

The **Paper Trading** tab tracks the strategy forward with simulated money.

- **Create**: set the sidebar to the strategy rules you want, open the tab, choose
  starting capital and inception date. The sidebar settings are snapshotted as the
  portfolio's locked rules.
- **Orders**: when a rebalance is due (initial buy, or the configured rebalance day
  has passed), the tab shows a whole-share order list ranked off the latest closes.
  Execute at the next market open, record actual fill prices, and confirm. Nothing
  trades automatically; confirming with all rows unticked records a no-trade
  rebalance.
- **State**: the trade ledger is the source of truth. It lives in the browser
  (IndexedDB); use **Export JSON** and commit the file as
  `site/data/paper_portfolio.json` to make it durable, auditable in git, and
  visible on other devices. On load, the newer of the repo file and the local copy
  wins. Fix mistakes (bad fills, splits/bonuses) via export → edit → import.
- **Tracking**: daily NAV is replayed from the ledger against the refreshed price
  data and charted against the backtest engine over the same window (pre-tax — the
  parity check) and the Nifty 500. Tiles show P&L, CAGR (after 90 days), drawdown,
  cash, charges, and a simplified capital-gains tax estimate (reported, not
  deducted from cash). Warnings flag negative cash, missing prices, and fills whose
  recorded price has drifted >12% from the back-adjusted close (likely split/bonus
  — scale the quantity in the JSON).

## Notes

- Saved backtests are local to the browser/device via IndexedDB.
- The browser app is intended for interactive experimentation, not huge hosted parameter sweeps.
- Keep the repository public for the best free GitHub Actions/GitHub Pages behavior.

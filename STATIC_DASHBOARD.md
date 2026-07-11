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

## Notes

- Saved backtests are local to the browser/device via IndexedDB.
- The browser app is intended for interactive experimentation, not huge hosted parameter sweeps.
- Keep the repository public for the best free GitHub Actions/GitHub Pages behavior.

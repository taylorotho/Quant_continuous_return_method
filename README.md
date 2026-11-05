# Quant continuous-return method

## Russell 3000 current-constituent daily prices

`download_russell3000_prices.py` downloads 25 calendar years of daily prices
from Yahoo Finance through `yfinance`. The security universe is the latest
equity holdings snapshot from the iShares Russell 3000 ETF (IWV), so only
securities present in that current snapshot are retained.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python download_russell3000_prices.py
```

Outputs are under `data/russell3000/`:

- `current_constituents.csv`: labeled current-universe snapshot and Yahoo ticker
- `iwv_holdings_raw.csv`: unmodified source snapshot
- `daily_prices/year=YYYY/*.parquet`: one row per ticker/day, with the canonical
  `adjusted_close_usd` price and an audit `raw_close_usd` field
- `manifest.json`: dates, definitions, row counts, source, and caveats
- `failures.csv`: constituents for which Yahoo returned no daily history
- `validation_report.json`: row, coverage, uniqueness, date, and price checks

The job is resumable by batch. Use `--refresh` to overwrite all batches.
Run `.venv/bin/python validate_russell3000_prices.py` after downloading.

This dataset intentionally has survivorship bias: it applies today's universe
to the entire historical window. It is not point-in-time historical membership.

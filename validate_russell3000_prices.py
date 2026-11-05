#!/usr/bin/env python3
"""Validate schema, uniqueness, partitions, coverage, and manifest row counts."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path("data/russell3000")
EXPECTED_COLUMNS = ["date", "ticker", "adjusted_close_usd", "raw_close_usd"]


def main() -> None:
    constituents = pd.read_csv(ROOT / "current_constituents.csv")
    failures = pd.read_csv(ROOT / "failures.csv")
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    files = sorted((ROOT / "daily_prices").glob("year=*/*.parquet"))
    assert files, "No Parquet files found"

    row_count = 0
    observed_tickers: set[str] = set()
    minimum_date = None
    maximum_date = None
    duplicate_count = 0
    invalid_price_count = 0
    for path in files:
        frame = pd.read_parquet(path)
        assert frame.columns.tolist() == EXPECTED_COLUMNS, f"Bad schema: {path}"
        year = int(path.parent.name.split("=", 1)[1])
        assert frame["date"].dt.year.eq(year).all(), f"Wrong year partition: {path}"
        row_count += len(frame)
        duplicate_count += int(frame.duplicated(["date", "ticker"]).sum())
        observed_tickers.update(frame["ticker"].unique())
        invalid_price_count += int(
            (~frame["adjusted_close_usd"].map(lambda value: math.isfinite(value) and value > 0)).sum()
        )
        file_min = frame["date"].min()
        file_max = frame["date"].max()
        minimum_date = file_min if minimum_date is None else min(minimum_date, file_min)
        maximum_date = file_max if maximum_date is None else max(maximum_date, file_max)

    expected_tickers = set(constituents["yahoo_ticker"]) - set(failures["yahoo_ticker"])
    missing_successes = sorted(expected_tickers - observed_tickers)
    unexpected_tickers = sorted(observed_tickers - set(constituents["yahoo_ticker"]))
    assert duplicate_count == 0, f"Found {duplicate_count} duplicate ticker-days"
    assert invalid_price_count == 0, f"Found {invalid_price_count} invalid adjusted closes"
    assert not missing_successes, f"Successful tickers absent from prices: {missing_successes}"
    assert not unexpected_tickers, f"Prices outside current universe: {unexpected_tickers}"
    assert row_count == manifest["row_count"], (row_count, manifest["row_count"])

    report = {
        "status": "PASS",
        "parquet_file_count": len(files),
        "row_count": row_count,
        "constituent_snapshot_count": len(constituents),
        "priced_ticker_count": len(observed_tickers),
        "unpriced_ticker_count": len(failures),
        "minimum_date": minimum_date.date().isoformat(),
        "maximum_date": maximum_date.date().isoformat(),
        "duplicate_ticker_day_count": duplicate_count,
        "invalid_adjusted_close_count": invalid_price_count,
    }
    (ROOT / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download 25 years of daily prices for today's Russell 3000 constituents.

The current universe is proxied by the equity holdings of IWV, the iShares
Russell 3000 ETF. Price history comes only from yfinance. The canonical price
is Yahoo's split- and dividend-adjusted daily close (one row/security/day).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


HOLDINGS_URL = (
    "https://www.ishares.com/us/products/239714/"
    "ishares-russell-3000-etf/latest-holdings.csv"
)
PRICE_COLUMN = "adjusted_close_usd"


def yahoo_symbol(symbol: str) -> str:
    """Translate the common class-share separator to Yahoo's separator."""
    return re.sub(r"[ .]", "-", symbol.strip())


def fetch_constituents(output_dir: Path) -> tuple[pd.DataFrame, str]:
    request = urllib.request.Request(HOLDINGS_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        raw = response.read().decode("utf-8-sig")

    lines = raw.splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("Ticker,Name,"))
    metadata = list(csv.reader(lines[:header_index]))
    as_of = next(row[1] for row in metadata if row and row[0] == "Fund Holdings as of")
    holdings = pd.read_csv(io.StringIO("\n".join(lines[header_index:])))
    holdings = holdings.loc[holdings["Asset Class"].eq("Equity")].copy()
    holdings = holdings.loc[holdings["Ticker"].notna() & holdings["Ticker"].ne("-")].copy()
    holdings["ticker"] = holdings["Ticker"].astype(str).str.strip()
    holdings["yahoo_ticker"] = holdings["ticker"].map(yahoo_symbol)
    holdings = holdings.rename(
        columns={
            "Name": "company_name",
            "Sector": "sector",
            "Location": "location",
            "Exchange": "exchange",
            "Currency": "currency",
            "Weight (%)": "iwv_weight_percent",
        }
    )
    result = holdings[
        [
            "ticker",
            "yahoo_ticker",
            "company_name",
            "sector",
            "location",
            "exchange",
            "currency",
            "iwv_weight_percent",
        ]
    ].drop_duplicates("yahoo_ticker", keep="first")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "current_constituents.csv", index=False)
    (output_dir / "iwv_holdings_raw.csv").write_text(raw, encoding="utf-8")
    return result, as_of


def extract_ticker_frame(download: pd.DataFrame, ticker: str, multi_ticker: bool) -> pd.DataFrame:
    if download.empty:
        return pd.DataFrame()
    if multi_ticker:
        if ticker not in download.columns.get_level_values(1):
            return pd.DataFrame()
        frame = download.xs(ticker, axis=1, level=1, drop_level=True).copy()
    else:
        frame = download.copy()
    if "Adj Close" not in frame or "Close" not in frame:
        return pd.DataFrame()
    frame = frame[["Adj Close", "Close"]].dropna(subset=["Adj Close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    frame = frame.reset_index()
    frame.columns = ["date", PRICE_COLUMN, "raw_close_usd"]
    frame.insert(1, "ticker", ticker)
    return frame


def download_batch(tickers: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        actions=False,
        repair=True,
        group_by="column",
        threads=True,
        progress=False,
        timeout=30,
        multi_level_index=True,
    )
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in tickers:
        frame = extract_ticker_frame(data, ticker, multi_ticker=True)
        if frame.empty:
            missing.append(ticker)
        else:
            frames[ticker] = frame
    return frames, missing


def retry_individually(tickers: list[str], start: str, end: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    recovered: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for ticker in tickers:
        frame = pd.DataFrame()
        error = None
        for attempt in range(3):
            try:
                # Some malformed corporate-action records break yfinance's
                # optional repair path. The last attempt deliberately falls
                # back to Yahoo's un-repaired adjusted close instead of losing
                # the entire otherwise-valid ticker.
                use_repair = attempt < 2
                data = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    actions=False,
                    repair=use_repair,
                    progress=False,
                    threads=False,
                    timeout=30,
                    multi_level_index=False,
                )
                frame = extract_ticker_frame(data, ticker, multi_ticker=False)
                if not frame.empty:
                    break
            except Exception as exc:
                error = exc
            time.sleep(2**attempt)
        if frame.empty:
            logging.warning("No history for %s%s", ticker, f": {error}" if error else "")
            failed.append(ticker)
        else:
            recovered[ticker] = frame
    return recovered, failed


def write_batch(frames: dict[str, pd.DataFrame], output_dir: Path, batch_number: int) -> int:
    if not frames:
        return 0
    combined = pd.concat(frames.values(), ignore_index=True)
    combined = combined.drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"])
    combined["year"] = combined["date"].dt.year
    rows = 0
    for year, yearly in combined.groupby("year", sort=True):
        partition = output_dir / "daily_prices" / f"year={year}"
        partition.mkdir(parents=True, exist_ok=True)
        target = partition / f"batch-{batch_number:04d}.parquet"
        yearly.drop(columns="year").to_parquet(target, index=False, compression="zstd")
        rows += len(yearly)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/russell3000"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--years", type=int, default=25)
    parser.add_argument("--refresh", action="store_true", help="Redownload completed batches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output_dir: Path = args.output_dir
    constituents, holdings_as_of = fetch_constituents(output_dir)

    today = date.today()
    try:
        start_date = today.replace(year=today.year - args.years)
    except ValueError:
        start_date = today.replace(year=today.year - args.years, day=28)
    end_exclusive = today + timedelta(days=1)
    tickers = constituents["yahoo_ticker"].tolist()
    failures: list[str] = []
    total_rows = 0
    completed_batches = 0

    for offset in range(0, len(tickers), args.batch_size):
        batch_number = offset // args.batch_size
        batch = tickers[offset : offset + args.batch_size]
        marker = output_dir / ".completed" / f"batch-{batch_number:04d}.json"
        if marker.exists() and not args.refresh:
            marker_state = json.loads(marker.read_text(encoding="utf-8"))
            if not marker_state.get("failed"):
                logging.info("Skipping completed batch %d", batch_number)
                continue
            logging.info("Retrying batch %d because its marker has failures", batch_number)
        logging.info("Downloading batch %d (%d tickers)", batch_number, len(batch))
        frames, missing = download_batch(batch, start_date.isoformat(), end_exclusive.isoformat())
        still_missing: list[str] = []
        if missing:
            logging.info("Retrying %d missing tickers individually", len(missing))
            recovered, still_missing = retry_individually(missing, start_date.isoformat(), end_exclusive.isoformat())
            frames.update(recovered)
            failures.extend(still_missing)
        rows = write_batch(frames, output_dir, batch_number)
        total_rows += rows
        completed_batches += 1
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"tickers": batch, "rows": rows, "failed": still_missing}, indent=2),
            encoding="utf-8",
        )
        logging.info("Wrote %s rows for batch %d", f"{rows:,}", batch_number)

    marker_states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / ".completed").glob("batch-*.json"))
    ]
    all_failures = sorted({ticker for state in marker_states for ticker in state.get("failed", [])})
    all_rows = sum(int(state.get("rows", 0)) for state in marker_states)
    failure_rows = constituents.loc[constituents["yahoo_ticker"].isin(all_failures)]
    failure_rows.to_csv(output_dir / "failures.csv", index=False)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe_definition": "Current equity holdings of iShares Russell 3000 ETF (IWV)",
        "universe_source": HOLDINGS_URL,
        "universe_as_of": holdings_as_of,
        "constituent_count": len(constituents),
        "requested_start_date": start_date.isoformat(),
        "requested_end_date_inclusive": today.isoformat(),
        "price_source": "Yahoo Finance via yfinance",
        "canonical_price_column": PRICE_COLUMN,
        "canonical_price_definition": "Daily close adjusted for splits and cash dividends",
        "row_granularity": "One row per current constituent per available trading day",
        "output_format": "Hive-partitioned Parquet by calendar year, Zstandard compression",
        "completed_batch_count": len(marker_states),
        "row_count": all_rows,
        "failed_ticker_count": len(all_failures),
        "failed_tickers": all_failures,
        "survivorship_bias_warning": (
            "The universe is intentionally filtered to constituents present at the snapshot date; "
            "it is not the historical Russell 3000 membership and therefore has survivorship bias."
        ),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Finished: %d current constituents; %d failures", len(constituents), len(all_failures))


if __name__ == "__main__":
    main()

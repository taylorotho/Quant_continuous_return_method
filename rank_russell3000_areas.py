#!/usr/bin/env python3
"""Rank Russell 3000 firms by six-month log-price-path area."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from six_month_price_area import calculate_six_month_area


DEFAULT_DATA_ROOT = Path("data/russell3000")
DEFAULT_OUTPUT_DIR = Path("output/russell3000_areas")
RANKING_COLUMNS = [
    "rank",
    "ticker",
    "yahoo_ticker",
    "company_name",
    "area",
    "start_price",
    "end_price",
    "six_month_return",
]
SELECTION_COLUMNS = [
    "selection_rank",
    "area_rank",
    "ticker",
    "yahoo_ticker",
    "company_name",
    "area",
    "start_price",
    "end_price",
    "six_month_return",
]
EXCLUSION_COLUMNS = [
    "ticker",
    "yahoo_ticker",
    "company_name",
    "reason",
    "observation_count",
    "months_present",
]


def _window_bounds(month: int, year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
        raise ValueError("month must be an integer from 1 through 12")
    if isinstance(year, bool) or not isinstance(year, int) or year < 1:
        raise ValueError("year must be a positive integer")

    try:
        endpoint = pd.Timestamp(year=year, month=month, day=1)
    except (ValueError, OverflowError) as exc:
        raise ValueError("month and year must form a valid pandas date") from exc
    return endpoint - pd.DateOffset(months=6), endpoint


def _validate_dataset_coverage(
    data_root: Path, window_start: pd.Timestamp, endpoint: pd.Timestamp
) -> None:
    report_path = data_root / "validation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Dataset validation report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    minimum_date = pd.Timestamp(report["minimum_date"])
    maximum_date = pd.Timestamp(report["maximum_date"])

    # The first and last months containing the dataset boundaries may be
    # partial. Only windows wholly inside the intervening complete months are
    # accepted.
    earliest_complete_start = minimum_date.to_period("M").start_time + pd.DateOffset(months=1)
    latest_complete_endpoint = maximum_date.to_period("M").start_time
    if window_start < earliest_complete_start:
        raise ValueError(
            f"window starts before the first complete dataset month "
            f"({earliest_complete_start:%Y-%m-%d})"
        )
    if endpoint > latest_complete_endpoint:
        raise ValueError(
            f"endpoint is after the latest complete dataset boundary "
            f"({latest_complete_endpoint:%Y-%m-%d})"
        )


def _load_window_prices(
    data_root: Path, window_start: pd.Timestamp, endpoint: pd.Timestamp
) -> pd.DataFrame:
    final_included_day = endpoint - pd.offsets.Day(1)
    files: list[Path] = []
    for partition_year in range(window_start.year, final_included_day.year + 1):
        files.extend(
            sorted(
                (data_root / "daily_prices" / f"year={partition_year}").glob("*.parquet")
            )
        )
    if not files:
        raise FileNotFoundError(
            f"No price Parquet files found for {window_start:%Y-%m-%d} through "
            f"{final_included_day:%Y-%m-%d}"
        )

    prices = pd.concat(
        [
            pd.read_parquet(
                path,
                columns=["date", "ticker", "adjusted_close_usd"],
            )
            for path in files
        ],
        ignore_index=True,
    )
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    return prices.loc[
        prices["date"].ge(window_start) & prices["date"].lt(endpoint)
    ].copy()


def rank_russell3000_areas(
    month: int,
    year: int,
    data_root: Path = DEFAULT_DATA_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return an ascending six-month area ranking and an exclusions table.

    The requested month is an exclusive first-of-month endpoint. For example,
    ``month=10, year=2026`` measures observations from 2026-04-01 through
    2026-09-30. Firms need at least two observations in every one of those six
    calendar months to enter the ranking.
    """
    data_root = Path(data_root)
    window_start, endpoint = _window_bounds(month, year)
    _validate_dataset_coverage(data_root, window_start, endpoint)

    constituents_path = data_root / "current_constituents.csv"
    if not constituents_path.exists():
        raise FileNotFoundError(f"Constituent file not found: {constituents_path}")
    constituents = pd.read_csv(constituents_path)
    required_constituent_columns = {"ticker", "yahoo_ticker", "company_name"}
    missing_columns = required_constituent_columns - set(constituents.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"current_constituents.csv is missing column(s): {missing}")
    if constituents["yahoo_ticker"].isna().any() or constituents["yahoo_ticker"].duplicated().any():
        raise ValueError("current_constituents.csv must contain unique non-missing yahoo_ticker values")

    prices = _load_window_prices(data_root, window_start, endpoint)
    grouped_prices = {ticker: group for ticker, group in prices.groupby("ticker", sort=False)}
    expected_months = set(pd.period_range(window_start, endpoint - pd.offsets.Day(1), freq="M"))

    ranking_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    for constituent in constituents.itertuples(index=False):
        yahoo_ticker = constituent.yahoo_ticker
        identity = {
            "ticker": constituent.ticker,
            "yahoo_ticker": yahoo_ticker,
            "company_name": constituent.company_name,
        }
        ticker_prices = grouped_prices.get(yahoo_ticker)
        if ticker_prices is None or ticker_prices.empty:
            exclusion_rows.append(
                {
                    **identity,
                    "reason": "no observations in window",
                    "observation_count": 0,
                    "months_present": 0,
                }
            )
            continue

        month_values = ticker_prices["date"].dt.to_period("M")
        observed_months = set(month_values.unique())
        monthly_counts = month_values.value_counts()
        months_present = len(observed_months)
        if observed_months != expected_months or (monthly_counts < 2).any():
            exclusion_rows.append(
                {
                    **identity,
                    "reason": "incomplete six-month history",
                    "observation_count": len(ticker_prices),
                    "months_present": months_present,
                }
            )
            continue

        ordered_prices = ticker_prices.sort_values("date")
        try:
            area = calculate_six_month_area(ordered_prices)
        except (TypeError, ValueError) as exc:
            exclusion_rows.append(
                {
                    **identity,
                    "reason": f"invalid price history: {exc}",
                    "observation_count": len(ticker_prices),
                    "months_present": months_present,
                }
            )
            continue
        start_price = float(ordered_prices["adjusted_close_usd"].iloc[0])
        end_price = float(ordered_prices["adjusted_close_usd"].iloc[-1])
        ranking_rows.append(
            {
                **identity,
                "area": area,
                "start_price": start_price,
                "end_price": end_price,
                "six_month_return": end_price / start_price - 1.0,
            }
        )

    ranking = pd.DataFrame(ranking_rows, columns=RANKING_COLUMNS[1:])
    if ranking.empty:
        ranking = pd.DataFrame(columns=RANKING_COLUMNS)
    else:
        ranking = ranking.sort_values(["area", "ticker"], kind="stable").reset_index(drop=True)
        ranking.insert(0, "rank", range(1, len(ranking) + 1))

    exclusions = pd.DataFrame(exclusion_rows, columns=EXCLUSION_COLUMNS)
    if not exclusions.empty:
        exclusions = exclusions.sort_values("ticker", kind="stable").reset_index(drop=True)
    return ranking, exclusions


def select_low_area_high_return(area_ranking: pd.DataFrame) -> pd.DataFrame:
    """Apply the 20%-lowest-area then 20%-highest-return screen."""
    if not isinstance(area_ranking, pd.DataFrame):
        raise TypeError("area_ranking must be a pandas DataFrame")
    missing_columns = set(RANKING_COLUMNS) - set(area_ranking.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"area_ranking is missing column(s): {missing}")
    if area_ranking.empty:
        return pd.DataFrame(columns=SELECTION_COLUMNS)

    numeric_columns = ["area", "start_price", "end_price", "six_month_return"]
    numeric = area_ranking[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not numeric.map(math.isfinite).all().all():
        raise ValueError("area and price-return columns must contain finite numbers")

    low_area_count = max(1, math.floor(len(area_ranking) * 0.20))
    low_area = (
        area_ranking.sort_values(["area", "ticker"], kind="stable")
        .head(low_area_count)
        .copy()
    )
    selected_count = max(1, math.floor(len(low_area) * 0.20))
    selected = (
        low_area.sort_values(
            ["six_month_return", "area", "ticker"],
            ascending=[False, True, True],
            kind="stable",
        )
        .head(selected_count)
        .rename(columns={"rank": "area_rank"})
        .reset_index(drop=True)
    )
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    return selected[SELECTION_COLUMNS]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank current Russell 3000 firms by log-price six-month path area."
    )
    parser.add_argument("--month", type=int, required=True, help="Exclusive endpoint month (1-12)")
    parser.add_argument("--year", type=int, required=True, help="Exclusive endpoint year")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    window_start, endpoint = _window_bounds(args.month, args.year)
    ranking, exclusions = rank_russell3000_areas(
        month=args.month,
        year=args.year,
        data_root=args.data_root,
    )

    selected = select_low_area_high_return(ranking)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint_label = endpoint.strftime("%Y-%m-%d")
    ranking_path = args.output_dir / f"areas_{endpoint_label}.csv"
    exclusions_path = args.output_dir / f"exclusions_{endpoint_label}.csv"
    selected_path = args.output_dir / f"selected_{endpoint_label}.csv"
    ranking.to_csv(ranking_path, index=False)
    exclusions.to_csv(exclusions_path, index=False)
    selected.to_csv(selected_path, index=False)

    print(f"Window: [{window_start:%Y-%m-%d}, {endpoint:%Y-%m-%d})")
    print(f"Ranked firms: {len(ranking):,}")
    print(f"Excluded firms: {len(exclusions):,}")
    print(f"Selected firms: {len(selected):,}")
    print(f"Ranking: {ranking_path}")
    print(f"Exclusions: {exclusions_path}")
    print(f"Selection: {selected_path}")


if __name__ == "__main__":
    main()

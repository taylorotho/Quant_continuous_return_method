"""Tests for the Russell 3000 six-month area ranking."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rank_russell3000_areas import (
    RANKING_COLUMNS,
    main,
    rank_russell3000_areas,
    select_low_area_high_return,
)


class RankRussell3000AreasTests(unittest.TestCase):
    def make_dataset(self, root: Path, endpoint: str = "2026-10-01") -> None:
        endpoint_date = pd.Timestamp(endpoint)
        months = pd.period_range(endpoint_date - pd.DateOffset(months=6), periods=6, freq="M")
        rows: list[dict[str, object]] = []
        for month in months:
            start = month.start_time
            dates = [start + pd.offsets.Day(1), start + pd.offsets.Day(14), month.end_time.normalize()]
            for ticker, values in (("AAA", [10.0, 20.0, 10.0]), ("BBB", [20.0, 20.0, 20.0])):
                rows.extend(
                    {"date": date, "ticker": ticker, "adjusted_close_usd": value}
                    for date, value in zip(dates, values)
                )

        # This October value must not affect a ranking ending at October 1.
        rows.append(
            {"date": endpoint_date, "ticker": "AAA", "adjusted_close_usd": 10_000.0}
        )
        # CCC has only the final month; DDD has no price observations.
        final_month = months[-1]
        rows.extend(
            [
                {
                    "date": final_month.start_time + pd.offsets.Day(1),
                    "ticker": "CCC",
                    "adjusted_close_usd": 30.0,
                },
                {
                    "date": final_month.end_time.normalize(),
                    "ticker": "CCC",
                    "adjusted_close_usd": 31.0,
                },
            ]
        )
        prices = pd.DataFrame(rows)
        for partition_year, yearly in prices.groupby(prices["date"].dt.year):
            partition = root / "daily_prices" / f"year={partition_year}"
            partition.mkdir(parents=True, exist_ok=True)
            yearly.to_parquet(partition / "batch-0000.parquet", index=False)

        pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC", "DDD"],
                "yahoo_ticker": ["AAA", "BBB", "CCC", "DDD"],
                "company_name": ["Alpha", "Beta", "Gamma", "Delta"],
            }
        ).to_csv(root / "current_constituents.csv", index=False)
        report = {
            "minimum_date": (months[0].start_time - pd.DateOffset(months=1)).date().isoformat(),
            "maximum_date": (endpoint_date + pd.DateOffset(months=1)).date().isoformat(),
        }
        (root / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")

    def test_ranks_valid_firms_and_reports_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_dataset(root)

            ranking, exclusions = rank_russell3000_areas(10, 2026, root)

            self.assertEqual(ranking["ticker"].tolist(), ["BBB", "AAA"])
            self.assertEqual(ranking["rank"].tolist(), [1, 2])
            self.assertAlmostEqual(ranking.loc[0, "area"], 0.0)
            self.assertAlmostEqual(ranking.loc[1, "area"], 6.0)
            self.assertEqual(ranking["company_name"].tolist(), ["Beta", "Alpha"])
            self.assertAlmostEqual(ranking.loc[0, "six_month_return"], 0.0)
            self.assertAlmostEqual(ranking.loc[1, "six_month_return"], 0.0)
            self.assertEqual(exclusions["ticker"].tolist(), ["CCC", "DDD"])
            self.assertEqual(
                exclusions.set_index("ticker").loc["CCC", "reason"],
                "incomplete six-month history",
            )
            self.assertEqual(
                exclusions.set_index("ticker").loc["DDD", "reason"],
                "no observations in window",
            )

    def test_cross_year_window_reads_both_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_dataset(root, endpoint="2026-03-01")

            ranking, exclusions = rank_russell3000_areas(3, 2026, root)

            self.assertEqual(len(ranking), 2)
            self.assertEqual(len(exclusions), 2)
            self.assertEqual(ranking["ticker"].tolist(), ["BBB", "AAA"])

    def test_rejects_invalid_or_unavailable_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            self.make_dataset(root)

            for month, year in ((0, 2026), (13, 2026), (10, 0), (4, 2026), (12, 2026)):
                with self.subTest(month=month, year=year):
                    with self.assertRaises(ValueError):
                        rank_russell3000_areas(month, year, root)

    def test_cli_writes_ranking_and_exclusion_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory)
            root = base / "data"
            output = base / "output"
            root.mkdir()
            self.make_dataset(root)

            main(
                [
                    "--month",
                    "10",
                    "--year",
                    "2026",
                    "--data-root",
                    str(root),
                    "--output-dir",
                    str(output),
                ]
            )

            ranking = pd.read_csv(output / "areas_2026-10-01.csv")
            exclusions = pd.read_csv(output / "exclusions_2026-10-01.csv")
            selected = pd.read_csv(output / "selected_2026-10-01.csv")
            self.assertEqual(ranking["ticker"].tolist(), ["BBB", "AAA"])
            self.assertEqual(exclusions["ticker"].tolist(), ["CCC", "DDD"])
            self.assertEqual(selected["ticker"].tolist(), ["BBB"])


    def make_ranking(self, size: int) -> pd.DataFrame:
        rows = []
        for index in range(size):
            start_price = 100.0
            six_month_return = (index % 11 - 5) / 10.0
            rows.append(
                {
                    "rank": index + 1,
                    "ticker": f"T{index:04d}",
                    "yahoo_ticker": f"T{index:04d}",
                    "company_name": f"Firm {index}",
                    "area": float(index),
                    "start_price": start_price,
                    "end_price": start_price * (1.0 + six_month_return),
                    "six_month_return": six_month_return,
                }
            )
        return pd.DataFrame(rows, columns=RANKING_COLUMNS)

    def test_twenty_five_firms_produce_one_selection(self) -> None:
        ranking = self.make_ranking(25)

        selected = select_low_area_high_return(ranking)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.loc[0, "ticker"], "T0004")
        self.assertEqual(selected.loc[0, "area_rank"], 5)

    def test_2551_firms_produce_102_selections(self) -> None:
        ranking = self.make_ranking(2551)

        selected = select_low_area_high_return(ranking)

        self.assertEqual(len(selected), 102)
        self.assertLessEqual(selected["area"].max(), 509.0)
        self.assertTrue(selected["six_month_return"].is_monotonic_decreasing)

    def test_empty_and_small_rankings(self) -> None:
        empty = pd.DataFrame(columns=RANKING_COLUMNS)
        self.assertTrue(select_low_area_high_return(empty).empty)

        one_firm = self.make_ranking(1)
        selected = select_low_area_high_return(one_firm)
        self.assertEqual(selected["ticker"].tolist(), ["T0000"])

    def test_return_ties_use_area_then_ticker(self) -> None:
        ranking = self.make_ranking(25)
        ranking.loc[:4, "six_month_return"] = -0.25
        ranking.loc[:4, "end_price"] = 75.0

        selected = select_low_area_high_return(ranking)

        self.assertEqual(selected["ticker"].tolist(), ["T0000"])


if __name__ == "__main__":
    unittest.main()

"""Tests for the six-month stock path area measure."""

from __future__ import annotations

import math
import unittest

import pandas as pd

from six_month_price_area import calculate_six_month_area


def six_month_frame() -> pd.DataFrame:
    dates = [
        "2026-01-02",
        "2026-01-15",
        "2026-01-30",
        "2026-02-02",
        "2026-02-27",
        "2026-03-02",
        "2026-03-31",
        "2026-04-01",
        "2026-04-30",
        "2026-05-01",
        "2026-05-29",
        "2026-06-01",
        "2026-06-12",
    ]
    prices = [10.0, 20.0, 10.0] + [10.0] * 10
    return pd.DataFrame(
        {"date": dates, "ticker": "TEST", "adjusted_close_usd": prices}
    )


class CalculateSixMonthAreaTests(unittest.TestCase):
    def test_straight_monthly_paths_have_zero_area(self) -> None:
        frame = six_month_frame()
        frame["adjusted_close_usd"] = [
            10.0,
            math.sqrt(200.0),
            20.0,
            20.0,
            25.0,
            25.0,
            30.0,
            30.0,
            20.0,
            20.0,
            10.0,
            10.0,
            15.0,
        ]

        self.assertAlmostEqual(calculate_six_month_area(frame), 0.0)

    def test_known_curved_path_area(self) -> None:
        self.assertAlmostEqual(calculate_six_month_area(six_month_frame()), math.log(2.0))

    def test_crossing_is_split_into_absolute_triangles(self) -> None:
        frame = six_month_frame()
        extra_january_row = pd.DataFrame(
            {
                "date": ["2026-01-22"],
                "ticker": ["TEST"],
                "adjusted_close_usd": [10.0],
            }
        )
        frame = pd.concat([frame, extra_january_row], ignore_index=True)
        january = frame["date"].astype(str).str.startswith("2026-01")
        frame.loc[january, "adjusted_close_usd"] = [20.0, 30.0, 20.0, 10.0]

        a = math.log(1.5)
        b = abs(math.log(0.5))
        expected = a / 2 + (a**2 + b**2) / (2 * (a + b)) + b / 2
        self.assertAlmostEqual(calculate_six_month_area(frame), expected)

    def test_price_scale_does_not_change_area(self) -> None:
        frame = six_month_frame()
        scaled = frame.copy()
        scaled["adjusted_close_usd"] *= 37.0

        self.assertAlmostEqual(
            calculate_six_month_area(frame), calculate_six_month_area(scaled)
        )

    def test_unsorted_input_is_sorted_internally(self) -> None:
        frame = six_month_frame()
        shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)

        self.assertAlmostEqual(
            calculate_six_month_area(frame), calculate_six_month_area(shuffled)
        )

    def test_partial_edge_months_are_accepted(self) -> None:
        frame = six_month_frame()
        self.assertEqual(frame["date"].iloc[-1], "2026-06-12")

        self.assertIsInstance(calculate_six_month_area(frame), float)

    def test_invalid_inputs_are_rejected(self) -> None:
        base = six_month_frame()
        cases: dict[str, pd.DataFrame] = {}

        cases["five months"] = base[~base["date"].astype(str).str.startswith("2026-06")]

        seven_months = pd.concat(
            [
                base,
                pd.DataFrame(
                    {
                        "date": ["2026-07-01", "2026-07-31"],
                        "ticker": ["TEST", "TEST"],
                        "adjusted_close_usd": [10.0, 10.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        cases["seven months"] = seven_months

        nonconsecutive = base.copy()
        nonconsecutive.loc[
            nonconsecutive["date"].astype(str).str.startswith("2026-06"), "date"
        ] = ["2026-07-01", "2026-07-12"]
        cases["nonconsecutive months"] = nonconsecutive

        cases["duplicate date"] = pd.concat([base, base.iloc[[0]]], ignore_index=True)

        multiple_tickers = base.copy()
        multiple_tickers.loc[0, "ticker"] = "OTHER"
        cases["multiple tickers"] = multiple_tickers

        invalid_price = base.copy()
        invalid_price.loc[0, "adjusted_close_usd"] = 0.0
        cases["invalid price"] = invalid_price

        one_february_observation = base.drop(index=3)
        cases["month with one observation"] = one_february_observation

        for label, frame in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    calculate_six_month_area(frame)


if __name__ == "__main__":
    unittest.main()

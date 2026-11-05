"""Measure how far a six-month stock-price path bends from monthly chords."""

from __future__ import annotations

import math

import pandas as pd


DATE_COLUMN = "date"
PRICE_COLUMN = "adjusted_close_usd"


def _absolute_linear_area(left_gap: float, right_gap: float) -> float:
    """Integrate the absolute value of a linear gap over a unit interval."""
    left_magnitude = abs(left_gap)
    right_magnitude = abs(right_gap)

    if left_gap == 0.0 or right_gap == 0.0 or left_gap * right_gap > 0.0:
        return (left_magnitude + right_magnitude) / 2.0

    # The lines cross inside the interval. The absolute region is two
    # triangles rather than a trapezoid whose signed parts cancel.
    return (left_magnitude**2 + right_magnitude**2) / (
        2.0 * (left_magnitude + right_magnitude)
    )


def calculate_six_month_area(prices: pd.DataFrame) -> float:
    """Return the normalized area between a stock path and its monthly chords.

    ``prices`` must contain one stock's daily observations for exactly six
    consecutive represented calendar months. Partial first and last months are
    allowed, but every month needs at least two observations. Every price is
    divided by the first price in the six-month period before measuring area.

    Within each month, a chord joins the first and last normalized prices. The
    observed prices and the chord are both treated as linear between adjacent
    trading observations, with every adjacent pair one horizontal unit apart.
    The returned scalar is the sum of the absolute areas between those lines.

    Required columns are ``date`` and ``adjusted_close_usd``. If a ``ticker``
    column is supplied, it must contain exactly one non-missing ticker.

    Raises:
        TypeError: If ``prices`` is not a pandas DataFrame.
        ValueError: If the input schema or observations violate the contract.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    missing_columns = {DATE_COLUMN, PRICE_COLUMN} - set(prices.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"prices is missing required column(s): {missing}")
    if prices.empty:
        raise ValueError("prices must not be empty")

    frame = prices.copy()
    if "ticker" in frame.columns:
        if frame["ticker"].isna().any() or frame["ticker"].nunique() != 1:
            raise ValueError("ticker must contain exactly one non-missing value")

    try:
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("date must contain valid datetime values") from exc
    if frame[DATE_COLUMN].isna().any():
        raise ValueError("date must not contain missing values")
    if frame[DATE_COLUMN].duplicated().any():
        raise ValueError("date must contain unique observations")

    frame[PRICE_COLUMN] = pd.to_numeric(frame[PRICE_COLUMN], errors="coerce")
    if not frame[PRICE_COLUMN].map(lambda value: math.isfinite(value) and value > 0.0).all():
        raise ValueError("adjusted_close_usd must contain only finite positive values")

    frame = frame.sort_values(DATE_COLUMN).reset_index(drop=True)
    frame["_month"] = frame[DATE_COLUMN].dt.to_period("M")
    months = frame["_month"].drop_duplicates().tolist()
    if len(months) != 6:
        raise ValueError("prices must represent exactly six calendar months")
    if any(right.ordinal - left.ordinal != 1 for left, right in zip(months, months[1:])):
        raise ValueError("the six represented calendar months must be consecutive")

    observations_per_month = frame.groupby("_month", observed=True).size()
    if (observations_per_month < 2).any():
        raise ValueError("every represented month must contain at least two observations")

    first_price = float(frame.loc[0, PRICE_COLUMN])
    frame["_normalized_price"] = frame[PRICE_COLUMN] / first_price

    total_area = 0.0
    for _, monthly in frame.groupby("_month", sort=False, observed=True):
        monthly_prices = monthly["_normalized_price"].tolist()
        interval_count = len(monthly_prices) - 1
        chord_start = monthly_prices[0]
        chord_change = monthly_prices[-1] - chord_start
        gaps = [
            price - (chord_start + chord_change * index / interval_count)
            for index, price in enumerate(monthly_prices)
        ]
        total_area += sum(
            _absolute_linear_area(left, right)
            for left, right in zip(gaps, gaps[1:])
        )

    return float(total_area)


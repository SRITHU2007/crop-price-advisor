
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def build_market_series(
    df: pd.DataFrame,
    market: str | None = None
) -> pd.Series:
    """Build a daily tomato modal-price series for a selected market."""

    required = {"arrival_date", "modal_price"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    data = df.copy()
    data["arrival_date"] = pd.to_datetime(
        data["arrival_date"], errors="coerce"
    )
    data["modal_price"] = pd.to_numeric(
        data["modal_price"], errors="coerce"
    )

    if market is not None and "market" in data.columns:
        data = data[
            data["market"].astype(str).str.casefold()
            == str(market).casefold()
        ]

    data = data.dropna(subset=["arrival_date", "modal_price"])
    data = data[data["modal_price"] >= 0]

    if data.empty:
        return pd.Series(dtype=float, name="modal_price")

    series = (
        data.groupby(data["arrival_date"].dt.normalize())["modal_price"]
        .mean()
        .sort_index()
    )

    series.name = "modal_price"
    return series


def _make_features(values, dates):
    """Create lag and calendar features for a price series."""

    rows = []

    for i, value in enumerate(values):
        date = pd.Timestamp(dates[i])

        rows.append({
            "lag_1": values[i - 1] if i >= 1 else value,
            "lag_2": values[i - 2] if i >= 2 else value,
            "lag_3": values[i - 3] if i >= 3 else value,
            "lag_7": values[i - 7] if i >= 7 else value,
            "lag_14": values[i - 14] if i >= 14 else value,
            "rolling_mean_7": float(np.mean(values[max(0, i - 7):i]))
            if i > 0 else value,
            "day_of_week": date.dayofweek,
            "month": date.month,
            "day_of_year": date.dayofyear,
        })

    return pd.DataFrame(rows)


def forecast_prices(
    series: pd.Series,
    days: int = 7,
    horizon: int | None = None
) -> pd.DataFrame:
    """
    Forecast future prices using Random Forest when enough history exists.
    Otherwise, use a recent-trend fallback.

    Returns date, predicted_price, lower_bound, upper_bound and model.
    """

    if horizon is not None:
        days = horizon

    if days < 1:
        raise ValueError("Forecast days must be at least 1.")

    clean = pd.Series(series).copy()
    clean.index = pd.to_datetime(clean.index, errors="coerce")
    clean = pd.to_numeric(clean, errors="coerce").dropna()
    clean = clean[~clean.index.isna()].sort_index()

    if clean.empty:
        return pd.DataFrame(columns=[
            "date", "predicted_price", "forecast_price",
            "price", "lower_bound", "upper_bound", "model"
        ])

    clean = clean.groupby(clean.index.normalize()).mean()
    clean = clean.asfreq("D").interpolate(method="linear").ffill().bfill()

    values = clean.to_numpy(dtype=float)
    dates = list(clean.index)

    future_dates = pd.date_range(
        start=clean.index[-1] + pd.Timedelta(days=1),
        periods=days,
        freq="D"
    )

    predictions = []
    model_name = "Recent trend fallback"

    if len(values) >= 18:
        features = _make_features(values, dates)
        target = values

        # Exclude the first row because it has no genuine prior observations.
        X_train = features.iloc[1:]
        y_train = target[1:]

        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        working_values = list(values)
        working_dates = list(clean.index)

        for future_date in future_dates:
            # Build one feature row using the future date and known price history.
            i = len(working_values)
            history = np.asarray(working_values, dtype=float)

            row = pd.DataFrame([{
                "lag_1": history[-1],
                "lag_2": history[-2] if len(history) >= 2 else history[-1],
                "lag_3": history[-3] if len(history) >= 3 else history[-1],
                "lag_7": history[-7] if len(history) >= 7 else history[-1],
                "lag_14": history[-14] if len(history) >= 14 else history[-1],
                "rolling_mean_7": float(np.mean(history[-7:])),
                "day_of_week": future_date.dayofweek,
                "month": future_date.month,
                "day_of_year": future_date.dayofyear,
            }])

            prediction = float(model.predict(row)[0])
            prediction = max(0.0, prediction)

            predictions.append(prediction)
            working_values.append(prediction)
            working_dates.append(future_date)

        model_name = "Random Forest (experimental)"

    else:
        # Simple recent trend, limited to avoid extreme extrapolation.
        recent = values[-min(7, len(values)):]
        trend = (
            float(np.mean(np.diff(recent)))
            if len(recent) >= 2 else 0.0
        )
        trend = float(np.clip(trend, -0.10 * values[-1],
                              0.10 * values[-1]))

        for step in range(1, days + 1):
            predictions.append(max(0.0, values[-1] + trend * step))

    # Approximate uncertainty band based on recent price variability.
    recent_changes = np.diff(values[-min(30, len(values)):])
    volatility = (
        float(np.std(recent_changes))
        if len(recent_changes) > 1 else 0.0
    )

    lower = []
    upper = []

    for i, prediction in enumerate(predictions, start=1):
        margin = 1.96 * volatility * np.sqrt(i)
        lower.append(max(0.0, prediction - margin))
        upper.append(max(0.0, prediction + margin))

    result = pd.DataFrame({
        "date": future_dates,
        "predicted_price": predictions,
        "forecast_price": predictions,
        "price": predictions,
        "lower_bound": lower,
        "upper_bound": upper,
        "model": model_name,
    })

    return result


def evaluate_baseline(series: pd.Series) -> dict:
    """
    Evaluate a naive forecast that predicts the previous day's price.
    Uses a chronological holdout; this is not an evaluation of the
    Random Forest model itself.
    """

    clean = pd.Series(series).copy()
    clean = pd.to_numeric(clean, errors="coerce").dropna()

    if len(clean) < 3:
        return {
            "MAE": None,
            "RMSE": None,
            "mae": None,
            "rmse": None,
            "message": "At least 3 observations are needed."
        }

    values = clean.to_numpy(dtype=float)

    split = max(2, int(len(values) * 0.8))
    split = min(split, len(values) - 1)

    actual = values[split:]
    predicted = values[split - 1:-1]

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "mae": mae,
        "rmse": rmse,
        "message": "Naive previous-price baseline evaluation."
    }

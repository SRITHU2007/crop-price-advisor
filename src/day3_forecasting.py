
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT / "data" / "processed" / "historical_tomato_prices.csv"
OUTPUT_DIR = ROOT / "data" / "processed"

FORECAST_DAYS = 7
TEST_FRACTION = 0.20
MIN_DATES = 10


def load_daily_prices():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Cleaned historical file not found. "
            "Complete Day 2 first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["arrival_date"])

    required = {"market", "arrival_date", "modal_price"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["modal_price"] = pd.to_numeric(
        df["modal_price"], errors="coerce"
    )
    df = df.dropna(subset=["market", "arrival_date", "modal_price"])
    df = df[df["modal_price"] > 0]

    if df.empty:
        raise ValueError("No valid price records were found.")

    # Select the market with the most distinct observation dates.
    dates_per_market = df.groupby("market")["arrival_date"].nunique()
    market = dates_per_market.idxmax()

    market_df = df[df["market"] == market].copy()

    # If multiple varieties are reported on a date, average their
    # modal prices for this initial baseline.
    daily = (
        market_df.groupby("arrival_date", as_index=False)["modal_price"]
        .mean()
        .sort_values("arrival_date")
        .reset_index(drop=True)
    )

    # Reindex to calendar days. Missing prices remain missing;
    # they are not filled with invented prices.
    daily = daily.set_index("arrival_date").asfreq("D")
    daily.index.name = "arrival_date"

    # For a simple baseline, use observed days only.
    daily = daily.dropna(subset=["modal_price"]).reset_index()

    return market, daily


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    market, daily = load_daily_prices()

    print("Selected market:", market)
    print("Observed dates:", len(daily))
    print("First date:", daily["arrival_date"].min().date())
    print("Last date:", daily["arrival_date"].max().date())

    if len(daily) < MIN_DATES:
        raise ValueError(
            f"Only {len(daily)} distinct price dates are available. "
            f"At least {MIN_DATES} are required for this demonstration. "
            "Download more historical data."
        )

    # Chronological split: older observations for training,
    # newer observations for testing.
    split = int(len(daily) * (1 - TEST_FRACTION))
    split = max(1, min(split, len(daily) - 1))

    train = daily.iloc[:split].copy()
    test = daily.iloc[split:].copy()

    # Naive one-step forecast: use the previous observed price.
    history = list(train["modal_price"].astype(float))
    predictions = []

    for actual_price in test["modal_price"].astype(float):
        predictions.append(history[-1])
        history.append(actual_price)

    actual = test["modal_price"].to_numpy(dtype=float)
    predicted = np.array(predictions, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero])
                       / actual[nonzero])) * 100
        if nonzero.any()
        else float("nan")
    )

    results = test[["arrival_date", "modal_price"]].copy()
    results = results.rename(columns={"modal_price": "actual_price"})
    results["predicted_price"] = predicted
    results.to_csv(
        OUTPUT_DIR / "baseline_test_results.csv", index=False
    )

    # Predict the next seven calendar days using the latest observed
    # price. This is a baseline, not a trend-aware model.
    last_price = float(daily["modal_price"].iloc[-1])
    last_date = daily["arrival_date"].iloc[-1]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D"
    )

    forecast = pd.DataFrame({
        "forecast_date": future_dates,
        "predicted_modal_price": [last_price] * FORECAST_DAYS
    })

    forecast.to_csv(
        OUTPUT_DIR / "baseline_7_day_forecast.csv", index=False
    )

    # Save a chart showing observed prices and the baseline forecast.
    plt.figure(figsize=(12, 5))
    plt.plot(
        daily["arrival_date"], daily["modal_price"],
        marker=".", label="Observed daily price"
    )
    plt.plot(
        forecast["forecast_date"],
        forecast["predicted_modal_price"],
        marker="o", linestyle="--", label="Naive 7-day baseline"
    )
    plt.title(f"Tomato Price Baseline Forecast — {market}")
    plt.xlabel("Date")
    plt.ylabel("Modal price (source unit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_file = OUTPUT_DIR / "day3_baseline_forecast.png"
    plt.savefig(chart_file, dpi=150)
    plt.close()

    print("\nBASELINE EVALUATION")
    print("-------------------")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print("\nSaved test results: data/processed/baseline_test_results.csv")
    print("Saved forecast: data/processed/baseline_7_day_forecast.csv")
    print("Saved chart:", chart_file)
    print("\nForecast:")
    print(forecast.to_string(index=False))
    print(
        "\nNote: this baseline repeats the last observed price. "
        "It does not model trends or seasonality."
    )


if __name__ == "__main__":
    main()


from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FORECAST_FILE = ROOT / "data" / "processed" / "baseline_7_day_forecast.csv"


def recommend():
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(
            "Forecast not found. Run src/day3_forecasting.py first."
        )

    forecast = pd.read_csv(
        FORECAST_FILE, parse_dates=["forecast_date"]
    )

    print("\nSELL NOW vs HOLD CALCULATOR")
    print("---------------------------")

    # Ask the user for their own selling and storage assumptions.
    current_price = float(input("Current selling price per quintal: "))
    quantity = float(input("Quantity available (quintals): "))
    hold_days = int(input("Days to hold (1 to 7): "))
    spoilage_percent = float(input("Expected spoilage per day (%): "))
    storage_cost = float(input("Storage cost per quintal per day: "))

    if current_price <= 0:
        raise ValueError("Current selling price must be greater than zero.")
    if quantity <= 0:
        raise ValueError("Quantity must be greater than zero.")
    if not 1 <= hold_days <= 7:
        raise ValueError("Hold days must be between 1 and 7.")
    if not 0 <= spoilage_percent < 100:
        raise ValueError("Daily spoilage must be at least 0 and less than 100.")
    if storage_cost < 0:
        raise ValueError("Storage cost cannot be negative.")

    # Select the predicted price for the chosen holding period.
    row = forecast.iloc[hold_days - 1]
    predicted_price = float(row["predicted_modal_price"])

    # Selling now: no future spoilage or storage costs.
    sell_now = current_price * quantity

    # Holding reduces the sellable quantity due to spoilage.
    remaining_quantity = quantity * (
        1 - spoilage_percent / 100
    ) ** hold_days

    gross_future = predicted_price * remaining_quantity
    total_storage = storage_cost * quantity * hold_days
    hold_net = gross_future - total_storage

    difference = hold_net - sell_now

    print("\nRESULTS")
    print("-------")
    print(f"Current price per quintal: Rs. {current_price:.2f}")
    print(f"Predicted price per quintal: Rs. {predicted_price:.2f}")
    print(f"Original quantity: {quantity:.2f} quintals")
    print(f"Estimated remaining quantity: {remaining_quantity:.2f} quintals")
    print(f"Selling now proceeds: Rs. {sell_now:.2f}")
    print(f"Holding net proceeds: Rs. {hold_net:.2f}")
    print(f"Difference (hold - sell now): Rs. {difference:.2f}")

    if difference > 0:
        print("\nCALCULATOR RESULT: HOLD may give higher estimated proceeds.")
    elif difference < 0:
        print("\nCALCULATOR RESULT: SELL NOW may give higher estimated proceeds.")
    else:
        print("\nCALCULATOR RESULT: Both options have equal estimated proceeds.")

    print(
        "\nImportant: This result depends on your assumptions and the baseline "
        "forecast. It does not account for every market risk or guarantee profit."
    )


if __name__ == "__main__":
    recommend()

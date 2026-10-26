
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "data" / "processed" / "historical_tomato_prices.csv"
)
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "tomato_price_trend.png"
)


def plot_prices():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Run prepare_historical_data.py first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["arrival_date"])

    # Choose the market with the most records
    market = df["market"].value_counts().index[0]
    selected = df[df["market"] == market].copy()

    # Average prices when multiple varieties exist on a date
    daily = (
        selected.groupby("arrival_date", as_index=False)["modal_price"]
        .mean()
        .sort_values("arrival_date")
    )

    if daily.empty:
        raise ValueError("No prices available to plot.")

    plt.figure(figsize=(12, 5))
    plt.plot(
        daily["arrival_date"],
        daily["modal_price"],
        marker="o"
    )

    plt.title(f"Tomato Price Trend - {market}")
    plt.xlabel("Arrival Date")
    plt.ylabel("Modal Price (Rs. per quintal)")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=150)

    print("Selected market:", market)
    print("Number of dates:", len(daily))
    print("Chart saved to:", OUTPUT_FILE)

    plt.show()


if __name__ == "__main__":
    plot_prices()

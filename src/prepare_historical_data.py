
from pathlib import Path
import csv
import pandas as pd

# Find the main project folder
ROOT = Path(__file__).resolve().parent.parent

# Input and output file locations
INPUT_FILE = ROOT / "data" / "raw" / "historical_mandi_prices.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "historical_tomato_prices.csv"


def prepare_data():
    # 1. Check whether the downloaded file exists
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}\n"
            "Put historical_mandi_prices.csv inside data/raw."
        )

    records = []
    current_market = None

    # 2. Read the special market-wise CSV format
    with open(INPUT_FILE, "r", encoding="utf-8-sig", newline="") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines and report headings
            if not line:
                continue

            if line.startswith('"Date Wise Prices'):
                continue

            if line.startswith('"Commodity :'):
                continue

            # Remember the market for the rows that follow
            if line.startswith("Market Name :"):
                current_market = line.split(":", 1)[1].strip()
                continue

            # Skip the table header
            if line.startswith("Arrival Date,"):
                continue

            # Read one data row
            parts = next(csv.reader([line]))

            # Expected fields:
            # Date, Arrivals, Variety, Min Price, Max Price, Modal Price
            if len(parts) != 6:
                print("Skipping unexpected row:", line)
                continue

            if current_market is None:
                print("Skipping row without a market:", line)
                continue

            records.append({
                "market": current_market,
                "arrival_date": parts[0].strip(),
                "arrivals_metric_tonnes": parts[1].strip(),
                "variety": parts[2].strip(),
                "min_price": parts[3].strip(),
                "max_price": parts[4].strip(),
                "modal_price": parts[5].strip(),
            })

    # 3. Convert the extracted records to a DataFrame
    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "No price records were read. Check the downloaded CSV format."
        )

    print("Records read:", len(df))

    # 4. Convert dates to proper dates
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date"],
        dayfirst=True,
        errors="coerce"
    )

    # 5. Convert prices and arrivals to numbers
    numeric_columns = [
        "arrivals_metric_tonnes",
        "min_price",
        "max_price",
        "modal_price",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 6. Remove rows with missing or invalid essential data
    df = df.dropna(
        subset=["market", "arrival_date", "modal_price"]
    )

    df = df[
        (df["modal_price"] > 0)
        & (df["min_price"].isna() | (df["min_price"] >= 0))
        & (df["max_price"].isna() | (df["max_price"] >= 0))
    ].copy()

    # 7. Remove exact duplicate rows
    df = df.drop_duplicates()

    # 8. Sort the records
    df = df.sort_values(
        ["market", "arrival_date", "variety"]
    ).reset_index(drop=True)

    # 9. Create the output folder and save the clean CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, date_format="%Y-%m-%d")

    # 10. Print a summary
    print("\nDATA CLEANING COMPLETED")
    print("-----------------------")
    print("Clean records:", len(df))
    print("Number of markets:", df["market"].nunique())
    print("Number of distinct dates:", df["arrival_date"].nunique())
    print("First date:", df["arrival_date"].min().date())
    print("Last date:", df["arrival_date"].max().date())
    print("Output saved to:", OUTPUT_FILE)

    print("\nMarkets with the most records:")
    print(df["market"].value_counts().head(10).to_string())

    print("\nFirst five cleaned records:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    prepare_data()

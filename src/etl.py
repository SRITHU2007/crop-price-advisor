
from pathlib import Path

import pandas as pd

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "crop_price_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_crop_prices.csv"


def clean_data():
    """Read, validate, clean, and save crop-price data."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}"
        )

    # Read the original CSV
    df = pd.read_csv(RAW_FILE)
    print(f"Original rows: {len(df)}")

    # Remove duplicate records
    df = df.drop_duplicates()

    # Remove leading/trailing spaces from text columns
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype("string").str.strip()

    # Convert arrival dates
    df["Arrival_Date"] = pd.to_datetime(
        df["Arrival_Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Convert prices to numeric values
    price_columns = [
        "Min_x0020_Price",
        "Max_x0020_Price",
        "Modal_x0020_Price",
    ]

    for column in price_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Remove rows with invalid dates or missing prices
    df = df.dropna(
        subset=["Arrival_Date"] + price_columns
    )

    # Remove impossible negative prices
    df = df[(df[price_columns] >= 0).all(axis=1)]

    # Create output directory if needed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save cleaned data
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Cleaned rows: {len(df)}")
    print(f"Distinct dates: {df['Arrival_Date'].nunique()}")
    print(f"Saved cleaned dataset to: {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_data()

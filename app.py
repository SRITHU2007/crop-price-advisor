
from pathlib import Path

import pandas as pd
import streamlit as st

# Project paths
DATA_DIR = Path("data/processed")

HISTORY_FILE = DATA_DIR / "historical_tomato_prices.csv"
FORECAST_FILE = DATA_DIR / "baseline_7_day_forecast.csv"
METRICS_FILE = DATA_DIR / "baseline_test_results.csv"

st.set_page_config(
    page_title="Crop Price Advisor",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Crop Price and Selling-Day Advisor")
st.write(
    "Explore historical tomato prices, view a baseline forecast, "
    "and compare selling now with holding your crop."
)

st.caption(
    "Forecasts are experimental estimates, not guaranteed market prices."
)


def load_csv(path):
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {path}: {exc}")
        return None


history = load_csv(HISTORY_FILE)
forecast = load_csv(FORECAST_FILE)
metrics = load_csv(METRICS_FILE)

if history is None or history.empty:
    st.error(
        "Historical data is missing or empty. "
        "Complete Day 2 before using this dashboard."
    )
    st.stop()

required = {"market", "arrival_date", "modal_price"}
if not required.issubset(history.columns):
    st.error(
        "Historical CSV must contain: market, arrival_date, modal_price."
    )
    st.write("Columns found:", history.columns.tolist())
    st.stop()

history["arrival_date"] = pd.to_datetime(
    history["arrival_date"], errors="coerce"
)
history["modal_price"] = pd.to_numeric(
    history["modal_price"], errors="coerce"
)
history = history.dropna(
    subset=["market", "arrival_date", "modal_price"]
)
history = history[history["modal_price"] > 0]

if history.empty:
    st.error("No valid historical prices were found.")
    st.stop()

# Sidebar filters
st.sidebar.header("Dashboard Filters")

markets = sorted(history["market"].astype(str).unique())
selected_market = st.sidebar.selectbox(
    "Select a market",
    markets
)

market_data = history[
    history["market"].astype(str) == selected_market
].copy()

daily = (
    market_data.groupby("arrival_date", as_index=False)["modal_price"]
    .mean()
    .sort_values("arrival_date")
)

# Summary cards
col1, col2, col3 = st.columns(3)

col1.metric("Selected market", selected_market)
col2.metric("Historical records", f"{len(market_data):,}")

col3.metric(
    "Latest recorded modal price",
    f"₹{daily.iloc[-1]['modal_price']:,.2f}/quintal"
)

st.subheader("1. Historical Tomato Prices")

st.line_chart(
    daily.set_index("arrival_date")["modal_price"],
    y_label="Average modal price (₹/quintal)"
)

st.caption(
    "If several varieties were recorded on the same date, "
    "their modal prices are averaged for this chart."
)

with st.expander("View historical price data"):
    st.dataframe(daily, use_container_width=True)

st.subheader("2. Seven-Day Baseline Forecast")

if forecast is None or forecast.empty:
    st.warning(
        "Forecast file not found. Run "
        "`python src\\day3_forecasting.py` first."
    )
else:
    st.write("Forecast file columns:", forecast.columns.tolist())

    # Detect common date and predicted-price column names
    date_candidates = [
        c for c in forecast.columns
        if "date" in c.lower()
    ]
    price_candidates = [
        c for c in forecast.columns
        if any(word in c.lower()
               for word in ["forecast", "predicted", "prediction"])
        and any(word in c.lower()
                for word in ["price", "modal", "value"])
    ]

    # Fallback to common price column names
    if not price_candidates:
        price_candidates = [
            c for c in forecast.columns
            if c.lower() in [
                "modal_price", "predicted_price",
                "forecast_price", "price"
            ]
        ]

    if date_candidates and price_candidates:
        date_col = date_candidates[0]
        price_col = price_candidates[0]

        forecast[date_col] = pd.to_datetime(
            forecast[date_col], errors="coerce"
        )
        forecast[price_col] = pd.to_numeric(
            forecast[price_col], errors="coerce"
        )

        forecast_view = forecast.dropna(
            subset=[date_col, price_col]
        )

        if not forecast_view.empty:
            st.line_chart(
                forecast_view.set_index(date_col)[price_col],
                y_label="Estimated price (₹/quintal)"
            )
            st.dataframe(
                forecast_view, use_container_width=True
            )
        else:
            st.warning("No valid forecast rows were found.")
    else:
        st.warning(
            "The forecast column names could not be identified. "
            "Please share the columns shown above so we can adjust "
            "the dashboard."
        )
        st.dataframe(forecast, use_container_width=True)

st.subheader("3. Forecast Evaluation")

if metrics is not None and not metrics.empty:
    st.dataframe(metrics, use_container_width=True)
else:
    st.info(
        "Evaluation results are not available yet. "
        "Run the Day 3 forecasting script."
    )

st.subheader("4. Sell Now or Hold?")

st.write(
    "Enter your own market offer and storage estimates. "
    "The forecast price is only an estimate."
)

with st.form("selling_calculator"):
    current_price = st.number_input(
        "Current selling price (₹/quintal)",
        min_value=0.0,
        value=2500.0,
        step=50.0
    )

    quantity = st.number_input(
        "Quantity (quintals)",
        min_value=0.1,
        value=10.0,
        step=1.0
    )

    hold_days = st.slider(
        "Days to hold the crop",
        min_value=1,
        max_value=7,
        value=3
    )

    daily_spoilage = st.number_input(
        "Estimated spoilage per day (%)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.5
    )

    storage_cost = st.number_input(
        "Storage cost (₹/quintal/day)",
        min_value=0.0,
        value=10.0,
        step=5.0
    )

    submitted = st.form_submit_button("Compare options")

if submitted:
    sell_now = current_price * quantity

    # Use the last available baseline forecast price
    # when a valid forecast price column exists.
    forecast_price = None

    if forecast is not None and not forecast.empty:
        candidates = [
            c for c in forecast.columns
            if any(word in c.lower()
                   for word in ["forecast", "predicted", "prediction"])
            and any(word in c.lower()
                    for word in ["price", "modal", "value"])
        ]

        if not candidates:
            candidates = [
                c for c in forecast.columns
                if c.lower() in [
                    "modal_price", "predicted_price",
                    "forecast_price", "price"
                ]
            ]

        if candidates:
            values = pd.to_numeric(
                forecast[candidates[0]], errors="coerce"
            ).dropna()

            if not values.empty:
                forecast_price = float(values.iloc[-1])

    if forecast_price is None:
        st.warning(
            "A valid forecast price is unavailable. "
            "Please run the Day 3 forecasting script and check "
            "its output columns."
        )
    else:
        net_hold = (
            forecast_price
            * quantity
            * (1 - daily_spoilage / 100) ** hold_days
            - storage_cost * quantity * hold_days
        )

        a, b = st.columns(2)
        a.metric("Sell now", f"₹{sell_now:,.2f}")
        b.metric(
            f"Estimated net after holding {hold_days} days",
            f"₹{net_hold:,.2f}",
            delta=f"₹{net_hold - sell_now:,.2f} vs sell now"
        )

        if net_hold > sell_now:
            st.success(
                "Based on these inputs, holding has the higher "
                "estimated net proceeds. This is not a guarantee."
            )
        elif net_hold < sell_now:
            st.info(
                "Based on these inputs, selling now has the higher "
                "estimated proceeds."
            )
        else:
            st.info("Both options have equal estimated proceeds.")

st.divider()
st.caption(
    "Educational project only. Actual prices depend on market "
    "conditions, quality, transport, demand, and other costs."
)

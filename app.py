
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.price_model import (
    build_market_series,
    forecast_prices,
    evaluate_baseline,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Crop Price Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = (
    Path(__file__).parent
    / "data"
    / "processed"
    / "historical_tomato_prices.csv"
)


# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------

@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Load and clean the historical tomato price dataset."""

    df = pd.read_csv(file_path)

    required_columns = {"arrival_date", "modal_price"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required dataset columns: {sorted(missing)}"
        )

    df["arrival_date"] = pd.to_datetime(
        df["arrival_date"], errors="coerce"
    )
    df["modal_price"] = pd.to_numeric(
        df["modal_price"], errors="coerce"
    )

    df = df.dropna(subset=["arrival_date", "modal_price"])
    df = df[df["modal_price"] >= 0]

    if "market" not in df.columns:
        df["market"] = "All Markets"

    df["market"] = df["market"].fillna("Unknown").astype(str)

    return df.sort_values("arrival_date").reset_index(drop=True)


def rupees(value: float) -> str:
    """Format a price in Indian rupees."""

    return f"₹{value:,.2f}"


def make_download_csv(df: pd.DataFrame) -> bytes:
    """Convert a dataframe to downloadable CSV bytes."""

    return df.to_csv(index=False).encode("utf-8")


def get_market_series(
    data: pd.DataFrame, market: str
) -> pd.Series:
    """Get a daily price series for a market."""

    market_data = data[data["market"] == market].copy()

    series = build_market_series(market_data)

    if not isinstance(series, pd.Series):
        raise TypeError(
            "build_market_series() must return a pandas Series."
        )

    series.index = pd.to_datetime(series.index, errors="coerce")
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[~series.index.isna()].sort_index()

    return series


# --------------------------------------------------
# LOAD DATA SAFELY
# --------------------------------------------------

st.title("🌾 Crop Price Advisor")
st.caption(
    "Historical tomato prices, experimental forecasting, "
    "market comparisons, and sell-or-hold decision support."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Place historical_tomato_prices.csv inside "
        "data/processed/ and restart the application."
    )
    st.stop()

try:
    df = load_data(str(DATA_PATH))
except Exception as exc:
    st.error(f"Unable to load the dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The dataset has no valid price records.")
    st.stop()


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Settings")

markets = sorted(df["market"].unique().tolist())

selected_market = st.sidebar.selectbox(
    "Select market",
    markets,
    index=0,
)

minimum_date = df["arrival_date"].min().date()
maximum_date = df["arrival_date"].max().date()

date_range = st.sidebar.date_input(
    "Historical date range",
    value=(minimum_date, maximum_date),
    min_value=minimum_date,
    max_value=maximum_date,
)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = minimum_date
    end_date = maximum_date

forecast_days = st.sidebar.slider(
    "Forecast horizon (days)",
    min_value=7,
    max_value=30,
    value=7,
)

st.sidebar.caption(
    "Forecasts use historical data only. They are estimates, "
    "not guaranteed future market prices."
)


# --------------------------------------------------
# SELECTED MARKET DATA
# --------------------------------------------------

market_data = df[df["market"] == selected_market].copy()

daily = get_market_series(df, selected_market)

if daily.empty:
    st.warning("No valid prices are available for this market.")
    st.stop()

filtered_daily = daily.loc[
    (daily.index.date >= start_date)
    & (daily.index.date <= end_date)
]

latest_price = float(daily.iloc[-1])
latest_date = daily.index[-1]

previous_price = (
    float(daily.iloc[-2]) if len(daily) >= 2 else latest_price
)

price_change = latest_price - previous_price

change_percent = (
    (price_change / previous_price) * 100
    if previous_price != 0
    else 0.0
)

# Generate forecast using the complete available market history.
try:
    forecast_df = forecast_prices(daily, days=forecast_days)
except Exception as exc:
    forecast_df = pd.DataFrame()
    forecast_error = str(exc)
else:
    forecast_error = None


# --------------------------------------------------
# TOP-LEVEL METRICS
# --------------------------------------------------

st.subheader(f"Market Overview — {selected_market}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latest recorded price",
        rupees(latest_price),
        help="Latest available modal price in the dataset.",
    )

with col2:
    st.metric(
        "Change from previous day",
        rupees(price_change),
        delta=f"{change_percent:+.2f}%",
    )

with col3:
    st.metric(
        "Latest recorded date",
        latest_date.strftime("%d %b %Y"),
    )

with col4:
    st.metric(
        "Available daily observations",
        len(daily),
    )

st.caption(
    "Prices are expressed in the dataset's original unit, "
    "typically ₹ per quintal. Check the source dataset to confirm."
)


# --------------------------------------------------
# DASHBOARD TABS
# --------------------------------------------------

overview_tab, forecast_tab, decision_tab, comparison_tab, data_tab = (
    st.tabs([
        "📊 Overview",
        "🔮 Forecast",
        "💰 Sell or Hold",
        "🏪 Market Comparison",
        "📁 Data & Evaluation",
    ])
)


# ==================================================
# TAB 1: HISTORICAL OVERVIEW
# ==================================================

with overview_tab:
    st.subheader("Historical Price Trend")

    if filtered_daily.empty:
        st.warning("No records exist in the selected date range.")
    else:
        chart_data = filtered_daily.rename("Price (₹)").to_frame()

        st.line_chart(chart_data)

        st.download_button(
            "Download historical prices CSV",
            data=make_download_csv(
                filtered_daily.rename("price").rename_axis("date")
                .reset_index()
            ),
            file_name="historical_tomato_prices.csv",
            mime="text/csv",
        )

        st.subheader("Recent Price Records")

        history_table = (
            filtered_daily.rename("Price")
            .rename_axis("Date")
            .reset_index()
            .sort_values("Date", ascending=False)
        )

        st.dataframe(
            history_table,
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Price Summary")

    summary_cols = st.columns(4)

    with summary_cols[0]:
        st.metric(
            "Average price",
            rupees(float(daily.mean())),
        )

    with summary_cols[1]:
        st.metric(
            "Highest price",
            rupees(float(daily.max())),
        )

    with summary_cols[2]:
        st.metric(
            "Lowest price",
            rupees(float(daily.min())),
        )

    with summary_cols[3]:
        st.metric(
            "Median price",
            rupees(float(daily.median())),
        )


# ==================================================
# TAB 2: FORECAST
# ==================================================

with forecast_tab:
    st.subheader("Future Tomato Price Forecast")

    st.warning(
        "Experimental forecast: results are based on historical "
        "prices and may not capture weather, arrivals, demand, "
        "transport costs, or sudden market changes."
    )

    if forecast_error:
        st.error(f"Forecast could not be generated: {forecast_error}")

    elif forecast_df.empty:
        st.warning(
            "Not enough valid data is available to produce a forecast."
        )

    else:
        forecast_display = forecast_df.copy()

        forecast_display["date"] = pd.to_datetime(
            forecast_display["date"]
        )

        forecast_display = forecast_display.rename(
            columns={
                "date": "Date",
                "predicted_price": "Predicted Price",
                "lower_bound": "Lower Estimate",
                "upper_bound": "Upper Estimate",
                "model": "Model",
            }
        )

        chart_columns = [
            col for col in ["Predicted Price"]
            if col in forecast_display.columns
        ]

        if chart_columns:
            st.line_chart(
                forecast_display.set_index("Date")[chart_columns]
            )

        predicted_col = (
            "Predicted Price"
            if "Predicted Price" in forecast_display.columns
            else "forecast_price"
        )

        if predicted_col in forecast_display.columns:
            predicted_values = pd.to_numeric(
                forecast_display[predicted_col], errors="coerce"
            ).dropna()

            if not predicted_values.empty:
                f1, f2, f3 = st.columns(3)

                with f1:
                    st.metric(
                        "Average forecast price",
                        rupees(float(predicted_values.mean())),
                    )

                with f2:
                    st.metric(
                        "Final forecast price",
                        rupees(float(predicted_values.iloc[-1])),
                    )

                with f3:
                    st.metric(
                        "Forecast horizon",
                        f"{len(predicted_values)} days",
                    )

        st.dataframe(
            forecast_display,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download forecast CSV",
            data=make_download_csv(forecast_display),
            file_name="tomato_price_forecast.csv",
            mime="text/csv",
        )

        model_name = (
            str(forecast_df["model"].iloc[0])
            if "model" in forecast_df.columns
            and not forecast_df.empty
            else "Unknown"
        )

        st.info(f"Forecast method: {model_name}")


# ==================================================
# TAB 3: SELL NOW VS HOLD CALCULATOR
# ==================================================

with decision_tab:
    st.subheader("Sell Now vs Hold Calculator")

    st.write(
        "Estimate potential revenue from selling now versus "
        "holding the crop and selling at an assumed future price."
    )

    with st.form("sell_hold_form"):
        quantity = st.number_input(
            "Quantity available (quintals)",
            min_value=0.1,
            value=10.0,
            step=1.0,
        )

        current_market_price = st.number_input(
            "Current selling price (₹ per quintal)",
            min_value=0.0,
            value=float(latest_price),
            step=10.0,
        )

        expected_future_price = st.number_input(
            "Expected future selling price (₹ per quintal)",
            min_value=0.0,
            value=(
                float(forecast_df["predicted_price"].iloc[-1])
                if not forecast_df.empty
                and "predicted_price" in forecast_df.columns
                else float(latest_price)
            ),
            step=10.0,
        )

        holding_days = st.number_input(
            "Holding period (days)",
            min_value=1,
            max_value=30,
            value=7,
            step=1,
        )

        spoilage_percent = st.slider(
            "Expected spoilage (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
        )

        storage_cost = st.number_input(
            "Storage cost (₹ per quintal per day)",
            min_value=0.0,
            value=2.0,
            step=0.5,
        )

        additional_cost = st.number_input(
            "Other holding costs (₹ total)",
            min_value=0.0,
            value=0.0,
            step=10.0,
        )

        submitted = st.form_submit_button("Calculate recommendation")

    if submitted:
        current_revenue = quantity * current_market_price

        remaining_quantity = quantity * (
            1.0 - spoilage_percent / 100.0
        )

        future_revenue = (
            remaining_quantity * expected_future_price
        )

        total_storage_cost = (
            quantity * holding_days * storage_cost
        )

        future_net_revenue = (
            future_revenue - total_storage_cost - additional_cost
        )

        difference = future_net_revenue - current_revenue

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Revenue if sold now",
                rupees(current_revenue),
            )

        with b:
            st.metric(
                "Estimated net revenue if held",
                rupees(future_net_revenue),
            )

        with c:
            st.metric(
                "Estimated difference",
                rupees(difference),
                delta=rupees(difference),
            )

        if difference > 0:
            st.success(
                "Under these assumptions, holding has the higher "
                "estimated net revenue."
            )
        elif difference < 0:
            st.info(
                "Under these assumptions, selling now has the "
                "higher estimated revenue."
            )
        else:
            st.info(
                "Both options have the same estimated revenue "
                "under these assumptions."
            )

        st.caption(
            "This is a scenario calculator, not a guaranteed "
            "recommendation. It excludes any costs you have not "
            "entered and does not account for uncertainty in future prices."
        )


# ==================================================
# TAB 4: MARKET COMPARISON
# ==================================================

with comparison_tab:
    st.subheader("Compare Tomato Prices Across Markets")

    selected_markets = st.multiselect(
        "Choose markets to compare",
        options=markets,
        default=markets[:min(5, len(markets))],
    )

    if not selected_markets:
        st.info("Select at least one market.")
    else:
        comparison_rows = []

        for market_name in selected_markets:
            market_series = get_market_series(df, market_name)

            if market_series.empty:
                continue

            comparison_rows.append({
                "Market": market_name,
                "Latest Date": market_series.index[-1],
                "Latest Price": float(market_series.iloc[-1]),
                "Average Price": float(market_series.mean()),
                "Minimum Price": float(market_series.min()),
                "Maximum Price": float(market_series.max()),
                "Observations": len(market_series),
            })

        if not comparison_rows:
            st.warning("No valid prices were found for the selected markets.")
        else:
            comparison_df = pd.DataFrame(comparison_rows)

            st.bar_chart(
                comparison_df.set_index("Market")[["Latest Price"]]
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download market comparison CSV",
                data=make_download_csv(comparison_df),
                file_name="market_price_comparison.csv",
                mime="text/csv",
            )

            cheapest = comparison_df.loc[
                comparison_df["Latest Price"].idxmin()
            ]

            costliest = comparison_df.loc[
                comparison_df["Latest Price"].idxmax()
            ]

            st.write(
                f"**Lowest latest recorded price:** "
                f"{cheapest['Market']} — "
                f"{rupees(float(cheapest['Latest Price']))}"
            )

            st.write(
                f"**Highest latest recorded price:** "
                f"{costliest['Market']} — "
                f"{rupees(float(costliest['Latest Price']))}"
            )

            st.caption(
                "Market observations may have different arrival dates. "
                "Compare dates before making a selling decision."
            )


# ==================================================
# TAB 5: DATA AND EVALUATION
# ==================================================

with data_tab:
    st.subheader("Dataset Information")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric("Valid records", f"{len(df):,}")

    with info2:
        st.metric("Markets", df["market"].nunique())

    with info3:
        st.metric(
            "Date coverage",
            f"{df['arrival_date'].min():%d %b %Y} – "
            f"{df['arrival_date'].max():%d %b %Y}",
        )

    st.write("**Dataset columns**")

    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(dtype) for dtype in df.dtypes],
            "Missing values": [
                int(df[column].isna().sum())
                for column in df.columns
            ],
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download cleaned dataset CSV",
        data=make_download_csv(df),
        file_name="cleaned_tomato_prices.csv",
        mime="text/csv",
    )

    st.subheader("Naive Baseline Evaluation")

    st.write(
        "This evaluation compares the previous day's price with "
        "the actual price on a chronological holdout."
    )

    try:
        evaluation = evaluate_baseline(daily)

        eval1, eval2 = st.columns(2)

        with eval1:
            mae = evaluation.get("MAE", evaluation.get("mae"))

            st.metric(
                "Mean Absolute Error (MAE)",
                f"{mae:.2f}" if mae is not None else "Unavailable",
            )

        with eval2:
            rmse = evaluation.get("RMSE", evaluation.get("rmse"))

            st.metric(
                "Root Mean Squared Error (RMSE)",
                f"{rmse:.2f}" if rmse is not None else "Unavailable",
            )

        if evaluation.get("message"):
            st.caption(evaluation["message"])

    except Exception as exc:
        st.warning(f"Baseline evaluation unavailable: {exc}")

    st.subheader("Important Limitations")

    st.markdown(
        """
        - Forecasts are experimental and may be inaccurate.
        - Historical data does not guarantee future prices.
        - Weather, supply, demand, transport and spoilage can affect prices.
        - Price-threshold notifications are not sent automatically.
        - This dashboard uses the available CSV dataset; it does not
          automatically fetch live market prices.
        - The naive baseline evaluation does not measure the Random Forest
          model's predictive accuracy.
        """
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Crop Price Advisor | Decision-support tool for historical "
    "tomato market prices. Verify current prices with the relevant "
    "market before making financial decisions."
)

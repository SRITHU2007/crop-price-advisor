# 🌾 Crop Price Advisor

A web-based agricultural decision-support application that helps farmers analyze historical tomato prices, forecast potential future prices, compare markets, and estimate whether selling immediately or holding produce may be more profitable.

The application is built using Python and Streamlit, uses Machine Learning for experimental price forecasting, and is deployed on Render Cloud using Docker.

## 📌 Project Overview

Farmers need access to market price information to make informed selling decisions. Crop prices can fluctuate due to supply, demand, seasonal changes, transportation costs, and other market conditions.

Crop Price Advisor analyzes historical tomato prices and provides visualizations, experimental forecasts, market comparisons, and a revenue calculator to support decision-making.

## 🎯 Objectives

- Analyze historical tomato market prices.
- Forecast potential future prices using Machine Learning.
- Compare prices across different markets.
- Estimate revenue from selling now versus holding produce.
- Account for estimated spoilage and storage costs.
- Display results through an interactive web dashboard.
- Deploy the application on a cloud platform for online access.

## ✨ Features

### 1. Interactive Dashboard

Displays the latest recorded tomato price, recent price changes, historical trends, and summary statistics.

### 2. Historical Price Analysis

- Interactive historical price charts.
- Date-range filtering.
- Average, minimum, maximum, and median price statistics.
- Downloadable historical price data in CSV format.

### 3. Machine Learning-Based Price Forecasting

Uses Random Forest Regression to estimate future tomato prices from historical price patterns and time-based features.

The forecasting model uses:

- Previous-day prices.
- Previous price observations at different lags.
- Recent rolling averages.
- Day of the week.
- Month and day of the year.

When sufficient historical observations are unavailable, the application uses a simple recent-trend fallback.

**Note:** Forecasts are experimental estimates and are not guaranteed future market prices.

### 4. Sell Now vs Hold Calculator

Helps users compare two selling scenarios by considering:

- Available crop quantity.
- Current selling price.
- Expected future selling price.
- Holding duration.
- Estimated spoilage percentage.
- Storage costs.
- Additional holding costs.

The calculator estimates the potential revenue difference between selling now and holding the crop.

### 5. Market Comparison

- Compare recorded prices across selected markets.
- View latest available prices and historical statistics.
- Identify markets with lower or higher recorded prices.
- Download market comparison results.

Market observation dates should be checked before comparing prices.

### 6. Data Analysis and Evaluation

- Displays dataset coverage and record counts.
- Shows market coverage and data quality information.
- Calculates Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for a naive previous-price baseline.
- Supports downloading cleaned data.

The baseline metrics do not represent the Random Forest model's predictive accuracy.

## 🛠️ Technologies Used

| Technology                  | Purpose                                         |
| --------------------------- | ----------------------------------------------- |
| Python                      | Application logic and data processing           |
| Streamlit                   | Interactive web dashboard                       |
| Pandas                      | Data cleaning and analysis                      |
| NumPy                       | Numerical calculations                          |
| Scikit-learn                | Random Forest Regression and evaluation metrics |
| Matplotlib/Streamlit charts | Data visualization, where applicable            |
| Git                         | Version control                                 |
| GitHub                      | Source-code hosting                             |
| Docker                      | Application packaging                           |
| Render Cloud                | Web application hosting                         |

## 🧠 Machine Learning Algorithm

### Random Forest Regression

Random Forest is a supervised Machine Learning algorithm that combines predictions from multiple decision trees.

In this project, Random Forest Regression is used to estimate future tomato prices using historical price observations and calendar-based features.

The implementation uses parameters such as:

- `n_estimators=150`
- `max_depth=6`
- `min_samples_leaf=2`
- `random_state=42`

These parameters configure the number of trees, tree depth, minimum leaf size, and reproducibility.

The forecast is generated recursively, meaning that each predicted price can be used as an input for the next forecast step. Consequently, forecast uncertainty may increase with the prediction horizon.

## 📂 Project Structure

```text
crop_price_advisor_project/
│
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
│
├── src/
│   ├── __init__.py
│   └── price_model.py
│
└── data/
    ├── raw/
    └── processed/
        └── historical_tomato_prices.csv
```

*Note: The structure above shows the expected main files. Your actual folders may contain additional files.*

## 📊 Dataset

The application uses a historical tomato price CSV dataset.

Expected columns:

| Column         | Description                            |
| -------------- | -------------------------------------- |
| `market`       | Name of the market                     |
| `arrival_date` | Date of the market observation         |
| `modal_price`  | Modal price recorded for the commodity |

The application cleans invalid dates and price values before analysis.

The dashboard uses the dataset packaged with the application. It does not automatically retrieve live prices from market APIs.

## ⚙️ Installation and Setup

### Prerequisites

Install the following:

- Python 3.10 or a compatible version supported by your dependencies.
- Git (optional, for cloning the repository).
- Visual Studio Code or another Python editor.
- Docker Desktop

### Step 1: Clone the Repository

```bash
git clone https://github.com/SRITHU2007/crop-price-advisor.git
cd crop-price-advisor
```

Alternatively, open your existing project folder in VS Code.

### Step 2: Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the Python commands using `.venv\Scripts\python.exe` directly.

### Step 3: Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

Ensure `requirements.txt` includes the packages used by the application, including Streamlit, Pandas, NumPy, and Scikit-learn.

### Step 4: Run the Application

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## ☁️ Cloud Deployment

The application can be deployed on Render Cloud using Docker and GitHub.

### Deployment workflow

1. Push the project code to GitHub.
2. Sign in to Render.
3. Create or select the existing Web Service connected to the repository.
4. Choose Docker as the runtime environment.
5. Configure the Dockerfile path and build settings.
6. Deploy the application.
7. Open the generated public URL and test the dashboard.

### Deployment architecture

```text
User's Browser
      |
      v
Render Cloud Web Service
      |
      v
Docker Container
      |
      v
Streamlit Application
      |
      v
Historical CSV + Forecasting Logic
```

The free hosting plan may put the service to sleep after inactivity, so the first request after a period of inactivity may take longer.

## 🧪 Testing

Test the following features after local execution and cloud deployment:

- Dashboard loads successfully.
- Historical charts display correctly.
- Date filters update the historical view.
- Forecasts display future dates and estimated prices.
- Sell Now vs Hold calculations return results.
- Market comparison displays selected markets.
- CSV downloads work.
- Dataset statistics and baseline metrics are displayed.
- The deployed application remains accessible after refreshing the page.

Functional testing does not establish the predictive accuracy of the forecasting model.

## ⚠️ Limitations

- Forecast accuracy depends on the amount and quality of historical data.
- The available dataset may not reflect current market conditions.
- Weather, supply, demand, transportation, and other external factors are not directly modeled.
- The trend-based fallback is a simple estimation method.
- Forecast uncertainty can increase for longer prediction horizons.
- The baseline evaluation measures a naive previous-price method, not Random Forest performance.
- The selling calculator depends on the user's assumptions and entered costs.
- Price-threshold alerts are dashboard-based and do not automatically send SMS or email notifications.
- The application does not currently provide automatic live market-price updates.

## 👨‍🌾 Applications

- Historical agricultural price analysis.
- Market price comparison.
- Crop selling decision support.
- Educational demonstration of Machine Learning and cloud deployment.
- Agricultural data visualization.

## 📝 Conclusion

Crop Price Advisor demonstrates how data analysis, Machine Learning, interactive visualization, and cloud hosting can be combined into an agricultural decision-support application.

It provides historical tomato price analysis, experimental future price estimates, market comparison, and a revenue calculator to help users explore possible selling decisions.

The application is intended to support informed decision-making rather than guarantee market prices or profits.

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

st.set_page_config(page_title="Time Series Forecasting App", layout="wide")
st.title("📈 Time Series Forecasting Web App")

# --- Upload File ---
uploaded_file = st.sidebar.file_uploader("Upload Time Series CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Prepare columns
    date_cols = df.select_dtypes(include=['object', 'datetime']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    if not date_cols or not numeric_cols:
        st.error("The CSV must contain at least one date and one numeric column.")
    else:
        date_col = st.sidebar.selectbox("Select Date Column", date_cols)
        value_col = st.sidebar.selectbox("Select Value Column", numeric_cols)

        # Prepare time series data
        df[date_col] = pd.to_datetime(df[date_col])
        ts = df.set_index(date_col)[value_col].dropna()

        # --- Forecasting ---
        model_choice = st.sidebar.selectbox("Select Forecasting Model", ["Holt-Winters", "Prophet"])
        forecast_period = st.sidebar.slider("Forecast Period (Months)", 1, 24, 12)

        if model_choice == "Holt-Winters":
            model = ExponentialSmoothing(ts, trend="add", seasonal="add", seasonal_periods=12).fit()
            forecast = model.forecast(forecast_period)
            st.line_chart(ts.append(forecast))

        elif model_choice == "Prophet":
            prophet_df = df.rename(columns={date_col: 'ds', value_col: 'y'})
            model = Prophet().fit(prophet_df)
            future = model.make_future_dataframe(periods=forecast_period, freq='M')
            forecast = model.predict(future)
            st.line_chart(forecast.set_index('ds')['yhat'])
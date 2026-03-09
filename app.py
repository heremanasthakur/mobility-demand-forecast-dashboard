import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(page_title="Mobility Demand Forecast Dashboard", layout="wide")


# ---------------------------
# Utility functions
# ---------------------------
def poisson_pmf(k, lam):
    if lam < 0:
        return 0
    return (math.exp(-lam) * (lam ** k)) / math.factorial(k)


def poisson_cdf(k, lam):
    total = 0
    for i in range(k + 1):
        total += poisson_pmf(i, lam)
    return total


def overload_probability(lam, threshold):
    # P(X > threshold) = 1 - P(X <= threshold)
    return 1 - poisson_cdf(threshold, lam)


@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


def preprocess_data(df):
    df = df.copy()

    # Standardize column names
    df.columns = [col.strip().lower() for col in df.columns]

    if "timestamp" not in df.columns or "booking_count" not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'booking_count' columns.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["booking_count"] = pd.to_numeric(df["booking_count"], errors="coerce")
    df = df.dropna(subset=["booking_count"])

    df["booking_count"] = df["booking_count"].astype(int)
    df["hour"] = df["timestamp"].dt.hour
    df["day_name"] = df["timestamp"].dt.day_name()
    df["date"] = df["timestamp"].dt.date
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5

    return df


def calculate_hourly_lambda(df):
    hourly_mean = df.groupby("hour")["booking_count"].mean().reset_index()
    hourly_mean.columns = ["hour", "lambda"]
    return hourly_mean


def build_hour_day_heatmap(df):
    heatmap_df = df.groupby(["day_name", "hour"])["booking_count"].mean().reset_index()

    day_order = [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday"
    ]
    heatmap_df["day_name"] = pd.Categorical(heatmap_df["day_name"], categories=day_order, ordered=True)
    heatmap_df = heatmap_df.sort_values(["day_name", "hour"])

    pivot_df = heatmap_df.pivot(index="day_name", columns="hour", values="booking_count")
    return pivot_df


def simulate_next_month(hourly_lambda, days=30):
    rows = []
    for day in range(1, days + 1):
        for _, row in hourly_lambda.iterrows():
            hour = int(row["hour"])
            lam = float(row["lambda"])
            simulated_bookings = np.random.poisson(lam)
            rows.append({
                "day": day,
                "hour": hour,
                "predicted_bookings": simulated_bookings
            })
    return pd.DataFrame(rows)


def recommend_fleet(lam, buffer_percent):
    vehicles = math.ceil(lam * (1 + buffer_percent / 100))
    return vehicles


# ---------------------------
# UI
# ---------------------------
st.title("Mobility Demand Forecast Dashboard")
st.markdown("Upload historical mobility booking data and forecast hourly passenger demand using Poisson distribution.")

st.sidebar.header("Controls")
buffer_percent = st.sidebar.slider("Fleet safety buffer (%)", 0, 100, 20)
selected_threshold = st.sidebar.slider("Overload threshold (bookings)", 0, 100, 20)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

st.info("CSV must contain 2 columns: `timestamp` and `booking_count`")

if uploaded_file is None:
    st.markdown("### Sample CSV format")
    sample_df = pd.DataFrame({
        "timestamp": [
            "2025-01-01 08:00:00",
            "2025-01-01 09:00:00",
            "2025-01-01 10:00:00",
            "2025-01-01 18:00:00",
            "2025-01-02 08:00:00",
            "2025-01-02 09:00:00",
            "2025-01-02 18:00:00"
        ],
        "booking_count": [12, 9, 7, 18, 14, 10, 21]
    })
    st.dataframe(sample_df, use_container_width=True)
    st.stop()

try:
    raw_df = load_data(uploaded_file)
    df = preprocess_data(raw_df)
except Exception as e:
    st.error(f"Error in file: {e}")
    st.stop()

if df.empty:
    st.warning("No valid rows found in the uploaded file.")
    st.stop()

hourly_lambda = calculate_hourly_lambda(df)

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview",
    "Hourly Forecast",
    "Poisson Analysis",
    "Monthly Simulation"
])

# ---------------------------
# Tab 1: Data Overview
# ---------------------------
with tab1:
    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(df))
    col2.metric("Average Bookings", round(df["booking_count"].mean(), 2))
    col3.metric("Peak Booking", int(df["booking_count"].max()))

    st.subheader("Average Bookings by Hour")
    fig_hourly = px.line(
        hourly_lambda,
        x="hour",
        y="lambda",
        markers=True,
        title="Average Hourly Booking Demand"
    )
    fig_hourly.update_layout(xaxis_title="Hour of Day", yaxis_title="Average Bookings (λ)")
    st.plotly_chart(fig_hourly, use_container_width=True)

    st.subheader("Hourly Demand Table")
    hourly_display = hourly_lambda.copy()
    hourly_display["recommended_vehicles"] = hourly_display["lambda"].apply(
        lambda x: recommend_fleet(x, buffer_percent)
    )
    st.dataframe(hourly_display, use_container_width=True)

    st.subheader("Day vs Hour Heatmap")
    heatmap_pivot = build_hour_day_heatmap(df)

    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Hour", y="Day", color="Avg Booking"),
        aspect="auto",
        title="Average Bookings Heatmap"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ---------------------------
# Tab 2: Hourly Forecast
# ---------------------------
with tab2:
    st.subheader("Fleet Planning by Hour")

    selected_hour = st.selectbox("Select hour", list(range(24)), index=8 if 8 in list(range(24)) else 0)

    selected_row = hourly_lambda[hourly_lambda["hour"] == selected_hour]

    if selected_row.empty:
        st.warning("No historical data available for this hour.")
    else:
        lam = float(selected_row["lambda"].values[0])
        rec_vehicles = recommend_fleet(lam, buffer_percent)
        overload_prob = overload_probability(lam, selected_threshold)

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected Bookings (λ)", round(lam, 2))
        col2.metric("Recommended Vehicles", rec_vehicles)
        col3.metric("P(Bookings > Threshold)", f"{overload_prob:.2%}")

        st.markdown("### Hourly Business Insight")
        if overload_prob > 0.5:
            st.error("High overload risk. Keep more vehicles available in this hour.")
        elif overload_prob > 0.2:
            st.warning("Moderate overload risk. Add a small safety buffer.")
        else:
            st.success("Low overload risk. Current planning looks safe.")

        st.markdown("### Forecast for All Hours")
        forecast_df = hourly_lambda.copy()
        forecast_df["recommended_vehicles"] = forecast_df["lambda"].apply(
            lambda x: recommend_fleet(x, buffer_percent)
        )
        forecast_df["overload_probability"] = forecast_df["lambda"].apply(
            lambda x: overload_probability(x, selected_threshold)
        )

        st.dataframe(forecast_df, use_container_width=True)

# ---------------------------
# Tab 3: Poisson Analysis
# ---------------------------
with tab3:
    st.subheader("Poisson Distribution for Selected Hour")

    selected_hour_poisson = st.selectbox("Choose hour for Poisson probability chart", list(range(24)), index=18 if 18 in list(range(24)) else 0)

    selected_row_poisson = hourly_lambda[hourly_lambda["hour"] == selected_hour_poisson]

    if selected_row_poisson.empty:
        st.warning("No data available for this hour.")
    else:
        lam = float(selected_row_poisson["lambda"].values[0])

        max_k = max(20, int(lam * 2 + 10))
        k_values = list(range(0, max_k + 1))
        probs = [poisson_pmf(k, lam) for k in k_values]

        prob_df = pd.DataFrame({
            "bookings": k_values,
            "probability": probs
        })

        fig_prob = px.bar(
            prob_df,
            x="bookings",
            y="probability",
            title=f"Poisson Probability Distribution for Hour {selected_hour_poisson}:00 (λ={lam:.2f})"
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("### Key Probabilities")
        p_zero = poisson_pmf(0, lam)
        p_threshold = overload_probability(lam, selected_threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("P(0 bookings)", f"{p_zero:.2%}")
        c2.metric(f"P(bookings > {selected_threshold})", f"{p_threshold:.2%}")
        c3.metric("Expected bookings", f"{lam:.2f}")

# ---------------------------
# Tab 4: Monthly Simulation
# ---------------------------
with tab4:
    st.subheader("Next 30-Day Simulation")

    sim_days = st.slider("Simulation days", 7, 60, 30)
    sim_df = simulate_next_month(hourly_lambda, days=sim_days)

    daily_totals = sim_df.groupby("day")["predicted_bookings"].sum().reset_index()
    hourly_totals = sim_df.groupby("hour")["predicted_bookings"].mean().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        fig_daily = px.line(
            daily_totals,
            x="day",
            y="predicted_bookings",
            markers=True,
            title="Predicted Daily Total Bookings"
        )
        st.plotly_chart(fig_daily, use_container_width=True)

    with col2:
        fig_sim_hour = px.bar(
            hourly_totals,
            x="hour",
            y="predicted_bookings",
            title="Average Simulated Bookings by Hour"
        )
        st.plotly_chart(fig_sim_hour, use_container_width=True)

    st.markdown("### Simulated Data Preview")
    st.dataframe(sim_df.head(50), use_container_width=True)

    csv = sim_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Simulation CSV",
        data=csv,
        file_name="simulated_next_month_bookings.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Built with Streamlit, Pandas, NumPy, Plotly, and Poisson-based probabilistic forecasting.")
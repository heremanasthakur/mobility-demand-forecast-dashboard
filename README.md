# Mobility Demand Forecast Dashboard

## Overview

This project is an AI/ML-based web application that forecasts hourly passenger demand for mobility services.
It uses the **Poisson distribution** to estimate the expected number of ride bookings in each hourly interval based on historical booking data.

The application helps analyze ride demand patterns, estimate the probability of ride volumes, and recommend fleet size for better operational planning.

---

## Features

* Upload historical mobility booking data (CSV)
* Analyze **hourly ride demand patterns**
* Forecast demand using **Poisson probabilistic modeling**
* Calculate **probability of ride volumes for each hour**
* Detect **high-demand (overload) hours**
* Provide **fleet size recommendations**
* Visualize **demand heatmaps**
* Run **future demand simulations (30-day forecast)**
* Interactive web dashboard built with **Streamlit**

---

## How the Model Works

1. Historical booking data is uploaded to the system.
2. The system calculates the **average ride requests per hour (λ)**.
3. The **Poisson distribution** is applied to model ride demand probability for each hour.
4. The dashboard estimates:

   * Expected number of bookings
   * Probability of exceeding a demand threshold
   * Recommended fleet size based on demand patterns

The Poisson distribution is suitable because ride bookings are **count events occurring within a fixed time interval**.

---

## Technology Stack

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Plotly**

---

## Project Structure

```
mobility_forecast_app/
│
├── app.py              # Streamlit web application
├── requirements.txt    # Python dependencies
├── sample_data.csv     # Example dataset for testing
└── README.md           # Project documentation
```

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/mobility-demand-forecast-dashboard.git
cd mobility-demand-forecast-dashboard
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the application:

```
streamlit run app.py
```

The dashboard will open in your browser.

---

## Dataset Format

The dataset must contain the following columns:

```
timestamp,booking_count
2025-01-01 08:00:00,12
2025-01-01 09:00:00,9
2025-01-01 10:00:00,7
```

* **timestamp** → Date and time of booking interval
* **booking_count** → Number of ride bookings in that hour

---

## Example Use Cases

* Mobility fleet planning
* Demand forecasting for ride-sharing services
* Urban transportation analytics
* EV fleet utilization optimization

---

## Future Improvements

* Machine learning demand forecasting (Random Forest / XGBoost)
* Zone-wise demand prediction
* Weather-based demand analysis
* Real-time booking data integration
* Deployment as a production web service

---

## Author

**Manas Thakur**

AI / Machine Learning Enthusiast


---

## License

This project is open-source and available for learning and research purposes.

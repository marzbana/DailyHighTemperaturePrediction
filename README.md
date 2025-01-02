# Miami Daily High Temperature Prediction and Automated Kalshi Betting

This repository contains a project focused on predicting daily high temperatures in Miami, Florida using a variety of data sources (NWS, NOAA, Open-Meteo, EPA, etc.) and automatically placing trades on [Kalshi](https://kalshi.com/) based on those predictions.

## Overview
The main objective of this project is to predict daily high temperatures for Miami, Florida, and place conditional bets on Kalshi’s weather markets. The workflow includes:
1. Gathering raw weather data from multiple APIs.
2. Cleaning, merging, and exploring the data.
3. Training neural network models (LSTM and Bayesian neural networks) to predict future daily high temperatures.
4. Estimating confidence intervals around the predictions.
5. Using Kalshi's API to place trades automatically based on the model output.

This project aims to showcase:
- **API integration** (e.g., NOAA, Open-Meteo, EPA, National Weather Service)
- **Data cleaning and preprocessing** on real-world datasets.
- **Deep learning for time series** (LSTM).
- **Bayesian modeling** to quantify predictive uncertainty.
- **Automated trading** via the Kalshi API.

---

## Data Sources
1. **National Weather Service (NWS)** – Hourly weather forecasts for Miami.
2. **Open-Meteo (Historical & Forecast)** – Historical weather variables (temperature, humidity, precipitation, etc.) and short-term forecasts.
3. **NOAA** – Ocean temperature and tide height data.
4. **EPA** – Air quality data (PM2.5, Ozone) for Miami.
5. **Kalshi** – Weather markets for placing trades on daily high temperature outcomes.

Each data source provides a specific piece of the weather or environmental puzzle. Together, they yield a comprehensive dataset for training and predicting.

---

## Data Collection and Processing
- **APIs**: Each source is queried via REST APIs using Python’s `requests` module.  
- **Data Cleaning**: 
  - Merging multiple JSON/CSV responses.
  - Dealing with missing values, data outliers, and inconsistent time formats.
  - Interpolating data gaps in time series.
- **Feature Engineering**:
  - Creation of time-based features (`month_sin`, `month_cos`, etc.).
  - Rolling averages for smoothing temperature or pressure data.
  - Standardization and scaling of numerical features.

The processed dataset ends up in CSV files for quick ingestion into the modeling pipeline.

---

## Modeling Approach

### LSTM Model
A multi-layered LSTM (Long Short-Term Memory) network is employed to capture the temporal dependencies in the weather data.  
- **Architecture**:
  - Two LSTM layers with dropout regularization.
  - Fully connected (dense) output layer predicting temperature.
- **Training**:
  - Loss function: Mean Absolute Error (MAE) or Mean Squared Error (MSE).
  - Optimizer: `Adam`.
  - Train/test splits using `TimeSeriesSplit` for robust validation.

### Bayesian Layers for Confidence Intervals
To quantify predictive uncertainty, additional Bayesian layers (via [TensorFlow Probability](https://www.tensorflow.org/probability)) are used:
- A **variational inference** approach models the distribution of weights.
- By sampling multiple times from the posterior weight distribution, the model produces **confidence intervals** around the forecasted daily high temperature.

---

## Kalshi API Integration
Once the predictions (and confidence intervals) for a given day are generated:
1. The script interfaces with the **Kalshi Demo API** (or real environment if you have appropriate access).
2. It determines the relevant Kalshi weather market ticker for Miami’s daily high temperature.
3. Based on the model’s predicted distribution, the script decides how to place trades (e.g., buy “Yes” contracts for the range matching the forecast or “No” contracts if the forecast is below/above certain thresholds).

This automated approach allows for a fully end-to-end system: data ingestion, model prediction, and execution on the trading platform.

---

## Requirements
- **Python 3.8+**
- Core libraries:
  - `pandas`, `numpy`, `requests`, `matplotlib`, `seaborn`
  - `scikit-learn`, `tensorflow`, `tensorflow-probability`
  - `kalshi-python` 
- A **Kalshi** account (demo or real) to retrieve API credentials for placing trades.


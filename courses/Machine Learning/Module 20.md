# Module 20: Time Series Analysis - ARIMA Models

## What is ARIMA?

**ARIMA** stands for **AutoRegressive Integrated Moving Average**, and it is one of the most popular models used in time series forecasting. ARIMA models are effective for modeling and forecasting data that exhibit a clear trend and seasonality (though seasonal ARIMA variants like SARIMA are used for more complex seasonal data).

### Components of ARIMA:

ARIMA is made up of three components:

- **AR (AutoRegressive)**: This component captures the relationship between an observation and a number of lagged observations (previous data points).
- **I (Integrated)**: This component is used to make the data stationary by differencing the series, i.e., subtracting the current value from the previous value.
- **MA (Moving Average)**: This component captures the relationship between an observation and the residual errors from a moving average model applied to lagged observations.

### ARIMA Model Parameters:

- **p**: The number of lag observations in the AR model.
- **d**: The number of times the data needs to be differenced to make it stationary.
- **q**: The size of the moving average window.

A typical ARIMA model is denoted as **ARIMA(p, d, q)**.

---

## Section 20.1: Preparing the Data for ARIMA

### Stationarity in Time Series

Before fitting an ARIMA model, the time series data must be **stationary**. A stationary series is one where the statistical properties (such as mean, variance, and autocorrelation) do not change over time.

To check for stationarity:

- **Visualization**: Plot the data and look for trends, seasonality, and constant variance.
- **Augmented Dickey-Fuller Test**: A statistical test to check if the series is stationary.

If the data is not stationary, you may need to apply **differencing** (subtracting previous data points) to remove trends and make it stationary.

### Code Example 1: Checking for Stationarity

**Importing Libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

```

**Explanation**

We will use **Matplotlib** to plot the data and **statsmodels** to perform the Augmented Dickey-Fuller test for stationarity.

---

**Creating a Time Series Data with Trend**

```python
# Create a time series with an upward trend
date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
data_with_trend = [i + np.random.randn() * 5 for i in range(100)]

df_trend = pd.DataFrame({'Date': date_range, 'Value': data_with_trend})
df_trend.set_index('Date', inplace=True)

# Plot the time series data
plt.figure(figsize=(10,6))
plt.plot(df_trend.index, df_trend['Value'], label='Time Series with Trend', color='blue')
plt.title('Time Series with Trend')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Perform Augmented Dickey-Fuller Test
adf_result = adfuller(df_trend['Value'])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

```

**Explanation**

- We create a time series with a trend and plot it.
- The **Augmented Dickey-Fuller** test checks if the time series is stationary. If the p-value is greater than 0.05, the series is non-stationary, and we need to apply differencing.

---

### Output for the Example

**Output**

- The plot will show an increasing trend in the data.
- The p-value from the Augmented Dickey-Fuller test will indicate whether the series is stationary or not.

---

## Section 20.2: Fitting an ARIMA Model

### Fitting an ARIMA Model

Once the data is stationary, we can fit an ARIMA model. In practice, we often use **auto_arima** (from the `pmdarima` library) or **grid search** to select the optimal values for p, d, and q.

### Code Example 2: Fitting ARIMA Model

**Importing the Required Library**

```python
from statsmodels.tsa.arima.model import ARIMA

```

**Explanation**

We will use **ARIMA** from **statsmodels** to fit the model.

---

**Fitting the ARIMA Model**

```python
# Differencing to make the series stationary
df_trend_diff = df_trend['Value'].diff().dropna()

# Fit ARIMA model with p=1, d=1, q=1 (as an example)
model = ARIMA(df_trend_diff, order=(1, 1, 1))
model_fitted = model.fit()

# Display the model summary
print(model_fitted.summary())

# Plot the fitted values vs original data
plt.figure(figsize=(10,6))
plt.plot(df_trend.index[1:], df_trend_diff, label='Differenced Data', color='blue')
plt.plot(df_trend.index[1:], model_fitted.fittedvalues, label='Fitted ARIMA Model', color='red')
plt.title('ARIMA Model Fit')
plt.xlabel('Date')
plt.ylabel('Differenced Value')
plt.legend()
plt.show()

```

**Explanation**

- We first apply differencing to make the series stationary.
- Then, we fit an **ARIMA(1, 1, 1)** model to the stationary data.
- The `ARIMA` model is fitted to the differenced data, and we plot the fitted values against the original differenced data to visualize how well the model fits.

---

### Output for the Example

**Output**

The plot will show the original differenced data and the fitted ARIMA model. The red line represents the ARIMA model's predicted values, while the blue line represents the original differenced data.

---

## Section 20.3: Forecasting with ARIMA

### Forecasting Future Values

Once the ARIMA model is fitted, we can use it to forecast future values. The `forecast()` method in **statsmodels** allows us to predict the next values based on the fitted model.

### Code Example 3: Forecasting Future Values with ARIMA

```python
# Forecast the next 10 values
forecast_steps = 10
forecast_values = model_fitted.forecast(steps=forecast_steps)

# Create a future date range for the forecast
future_dates = pd.date_range(df_trend.index[-1], periods=forecast_steps + 1, freq='D')[1:]

# Plot the forecasted values
plt.figure(figsize=(10,6))
plt.plot(df_trend.index, df_trend['Value'], label='Original Data', color='blue')
plt.plot(future_dates, forecast_values, label='Forecasted Values', color='green')
plt.title('ARIMA Forecasting')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

```

**Explanation**

- After fitting the model, we use the `forecast()` method to predict the next 10 values.
- We generate future dates and plot the forecasted values along with the original data.

---

### Output for the Example

**Output**

The plot will display the original time series data and the forecasted values. The green line represents the predicted values based on the ARIMA model.
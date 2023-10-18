import streamlit as st
import pandas as pd
import numpy as np
from math import sqrt

from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from pandas import concat

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import boxcox

import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
from matplotlib import colors

import pmdarima as pm
from pmdarima.model_selection import train_test_split
from pmdarima.arima import auto_arima

st.set_page_config(page_title='Page 1', layout="wide")
st.title("Results")

df = pd.read_csv("D:/Lab_Main/Sem_5/ML/Final_LabTest/Electric_Production.csv")
df.rename(columns={"IPG2211A2N":"Energy","DATE":"Date"},inplace=True)

df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
df = df.set_index(['Date'])

df_log_scaled = df
df_log_scaled['Energy'] = boxcox(df_log_scaled['Energy'], lmbda=0.0)

moving_avg = df_log_scaled.rolling(window=12).mean()
df_log_scaled_ma = df_log_scaled - moving_avg
df_log_scaled_ma.dropna(inplace=True)

auto_c_f = acf(df_log_scaled_ma, nlags=20)
partial_auto_c_f = pacf(df_log_scaled_ma, nlags=20, method='ols')

ar_values = df_log_scaled_ma.values
train = ar_values[1:len(ar_values)-10]
test = ar_values[len(ar_values)-10:]

#-------------------------------------------------------------------------------
st.markdown('---')
st.subheader('Autoregression Model')
ar_values = df_log_scaled_ma.values
train = ar_values[1:len(ar_values)-10]
test = ar_values[len(ar_values)-10:]

model = ARIMA(train, order=(2,1,0))
AR_model = model.fit()

predictions = AR_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
ar_mse = mean_squared_error(test, predictions)
ar_mae = mean_absolute_error(test, predictions)
ar_rmse = sqrt(mean_squared_error(test, predictions))

predictions = AR_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
ar_score = mean_squared_error(test, predictions)
print('AR MSE: ',(round(ar_score,4)))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test, label = "true values", color = "cornflowerblue")
ax.plot(predictions,label = "forecasts", color='darkorange')
ax.set_title("AR Model", size = 14)
ax.legend(loc = 'upper left')
st.pyplot(fig)

#-------------------------------------------------------------------------------
st.markdown('---')
st.subheader('Moving Average Model')
model = ARIMA(train, order=(0,1,2))
MA_model = model.fit()

predictions = MA_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
ma_mse = mean_squared_error(test, predictions)
ma_mae = mean_absolute_error(test, predictions)
ma_rmse = sqrt(mean_squared_error(test, predictions))
predictions = MA_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
ma_score = mean_squared_error(test, predictions)
print('MA MSE: {}'.format(round(ma_score,4)))
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test, label = "true values", color = "cornflowerblue")
ax.plot(predictions,label = "forecasts", color='darkorange')
ax.set_title("MA Model", size = 14)
ax.legend(loc = 'upper left')
st.pyplot(fig)
#-------------------------------------------------------------------------------
st.markdown('---')
st.subheader('ARIMA Model')
model = ARIMA(train, order=(2,1,2))
ARIMA_model = model.fit()

predictions = ARIMA_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
arima_mse = mean_squared_error(test, predictions)
arima_mae = mean_absolute_error(test, predictions)
arima_rmse = sqrt(mean_squared_error(test, predictions))

predictions = ARIMA_model.predict(start=len(train), end=len(train)+len(test)-1, dynamic=False)
arima_score = mean_squared_error(test, predictions)
print('ARIMA MSE: {}'.format(round(arima_score,4)))
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test, label = "true values", color = "cornflowerblue")
ax.plot(predictions,label = "forecasts", color='darkorange')
ax.set_title("ARIMA Model", size = 14)
ax.legend(loc = 'upper left')
st.pyplot(fig)
#-------------------------------------------------------------------------------
st.markdown('---')
st.subheader('Mean Squared Errors')

errors = pd.DataFrame()
errors["Model"] = [ "Autoregression", "Moving Average", "ARIMA"]
errors["MSE"] = [ar_score, ma_score, arima_score]
errors = errors.sort_values("MSE", ascending = True, ignore_index = True)
errors.index = errors.Model
del errors["Model"]

def coloring_bg(s, min_, max_, cmap='Reds', low=0, high=0):
    color_range = max_ - min_
    norm = colors.Normalize(min_ - (color_range * low), max_ + (color_range * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]

st.write("MSE for different models:")

# Apply the style to the DataFrame
styled_errors = errors.style.apply(coloring_bg,
                                    min_=errors.min().min(),
                                    max_=errors.max().max(),
                                    low=0.1, high=0.85)

# Display the styled DataFrame
st.dataframe(styled_errors)

st.markdown('---')
st.write("MAE for different models:")
errors = pd.DataFrame()
errors["Model"] = [ "Auto Regression", "Moving Average", "ARIMA"]
errors["MAE"] = [ar_mae, ma_mae, arima_mae]
errors = errors.sort_values("MAE", ascending = True, ignore_index = True)
errors.index = errors.Model
del errors["Model"]

styled_errors = errors.style.apply(coloring_bg,min_ =errors.min().min(),
               max_ = errors.max().max(),cmap="Blues", low = 0.1, high = 0.85)
st.dataframe(styled_errors)

st.markdown('---')
st.write("RMSE for different models:")
errors = pd.DataFrame()
errors["Model"] = [ "Auto Regression", "Moving Average", "ARIMA"]
errors["RMSE"] = [ar_rmse, ma_rmse, arima_rmse]
errors = errors.sort_values("RMSE", ascending = True, ignore_index = True)
errors.index = errors.Model
del errors["Model"]

styled_errors = errors.style.apply(coloring_bg,min_ =errors.min().min(),
               max_ = errors.max().max(),cmap="Oranges", low = 0.1, high = 0.85)
st.dataframe(styled_errors)
#-------------------------------------------------------------------------------
st.markdown('---')
st.subheader("ARIMA Model Summary")

# Load your data (assuming 'df' is your DataFrame with the 'Energy' column)

# Split the data into training and testing sets
train, test = train_test_split(df['Energy'], test_size=0.2)

# Use auto_arima to automatically select the best ARIMA parameters
autoarima_model = auto_arima(train,
                             start_p=0, max_p=5,
                             start_q=0, max_q=5,
                             d=None, seasonal=False,
                             stepwise=True, suppress_warnings=True,
                             trace=True, error_action="ignore",
                             max_order=None, out_of_sample_size=int(len(test)))

# Print the best parameters selected by auto_arima
p, d, q = autoarima_model.order
st.write("Optimal values of parameters using AIC:")
st.write(f"p = {p}, d = {d}, q = {q}")

# Fit the ARIMA model with the selected parameters to the entire training data
arima_model = ARIMA(train, order=(p, d, q))
arima_fit = arima_model.fit()

# Display the ARIMA model summary
st.write("ARIMA Model Summary:")
st.write(arima_fit.summary())

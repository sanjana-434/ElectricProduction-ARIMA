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

st.set_page_config(page_title='Page 2', layout="wide")
st.title("Predictions")

df = pd.read_csv("Electric_Production.csv")
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
st.subheader("Predicting next 12 month's Electric Production")
model = ARIMA(train, order=(2, 1, 2))
ARIMA_model = model.fit()

model = ARIMA(train, order=(0,1,2))
MA_model = model.fit()

model = ARIMA(train, order=(2,1,0))
AR_model = model.fit()

predictions1 = ARIMA_model.predict(start=len(train), end=len(train) + 11, dynamic=False)

predictions2 = MA_model.predict(start=len(train), end=len(train) + 11, dynamic=False)

predictions3= AR_model.predict(start=len(train), end=len(train) + 11, dynamic=False)

next_12_dates = pd.date_range(start=df.index[-1], periods=12, freq=df.index.freq)


# fig, ax = plt.subplots(figsize=(10, 6))
# ax.plot(next_12_dates, predictions1, label='Predicted Data by ARIMA', marker='o', color = "cornflowerblue")
# ax.plot(next_12_dates, predictions2, label='Predicted Data by MA', marker='o', color = "orange")
# ax.plot(next_12_dates, predictions3, label='Predicted Data by AR', marker='o', color = "red")
# ax.set_title('Next 12 Data Points Prediction')
# ax.set_title("ARIMA Model", size = 14)
# ax.legend(loc = 'upper left')
# st.pyplot(fig)
predictions1=pd.DataFrame(predictions1)
predictions2=pd.DataFrame(predictions2)
predictions3=pd.DataFrame(predictions3)

predictions1.rename(columns={0:"Energy"},inplace=True)
predictions2.rename(columns={0:"Energy"},inplace=True)
predictions3.rename(columns={0:"Energy"},inplace=True)

t=df_log_scaled_ma.tail(12)

prediction_df1 = t.append(predictions1,ignore_index=True).tail(24)
prediction_df2 = t.append(predictions2,ignore_index=True).tail(24)
prediction_df3 = t.append(predictions3,ignore_index=True).tail(24)

inv_ma_predictions = prediction_df1.rolling(window=12).mean()
prediction_df1 = prediction_df1 + inv_ma_predictions
original_predictions1 = np.exp(prediction_df1.tail(12))

inv_ma_predictions = prediction_df2.rolling(window=12).mean()
prediction_df2 = prediction_df2 + inv_ma_predictions
original_predictions2 = np.exp(prediction_df2.tail(12))

inv_ma_predictions = prediction_df3.rolling(window=12).mean()
prediction_df3 = prediction_df3 + inv_ma_predictions
original_predictions3 = np.exp(prediction_df3.tail(12))


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(next_12_dates, original_predictions1, label='Predicted Data by ARIMA', marker='o', color = "cornflowerblue")
ax.plot(next_12_dates, original_predictions2, label='Predicted Data by MA', marker='o', color = "orange")
ax.plot(next_12_dates, original_predictions3, label='Predicted Data by AR', marker='o', color = "red")
ax.set_title('Next 12 Data Points Prediction')
ax.set_title("Model", size = 14)
ax.legend(loc = 'upper left')
st.pyplot(fig)

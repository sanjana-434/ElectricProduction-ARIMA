import streamlit as st
st.set_page_config(page_title='digit recognition', layout="wide")

st.title('Electricity Production Forecasting (ARIMA) ⚡')
st.markdown('---')

col1, col2 = st.columns(2)

with col1:
    st.image('https://www.scaler.com/topics/images/arima-examples.webp')

with col2:
    st.markdown('Electric Production dataset consists of Date and the  respective Electric Production.')
    st.markdown('It is a univariate time series dataset.')
    st.markdown('It contains 387 training images and 10 testing images.')


st.sidebar.write('Developed by ')
st.sidebar.write('Harini K V - 21PD10')
st.sidebar.write('Sanjana R - 21PD31')
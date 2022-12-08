from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from datetime import date

# st.title('Stock Forecast App')
# App title
st.markdown('''
# Stock Price App
''')
st.write('---')

# Sidebar
st.sidebar.header('FETCH DATA TILL..')
start = "2012-01-01"
end_d = st.sidebar.date_input("CHOOSE END DATE", date.today())

# start = "2012-01-01"
# end = st.date_input("Choose End date", date.today())


stocks = ('RELIANCE.NS', 'SBIN.NS', 'AAPL', 'MSFT', 'AMZN')
selected_stock = st.sidebar.selectbox('SELECT COMPANY', stocks)
df = data.DataReader(selected_stock, 'yahoo', start, end_d)

data_load_state = st.sidebar.text('LOADING DATA...')

# Describing Data
st.subheader(selected_stock)
st.write(df.reset_index().drop(['Date', 'Adj Close'], axis=1))
st.write('---')

data_load_state.text('LOADING DATA... DONE!')

# Visualizations
st.subheader('**CLOSING PRICE**')
# st.subheader("Closing price vs Time Chart")
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)
st.write('---')

st.subheader('**CLOSING PRICE WITH 100 MA**')
# st.subheader("Closing price vs Time Chart with 100 Moving Avg")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)
st.write('---')

st.subheader('**CLOSING PRICE WITH 100 MA  &  200 MA**')
# st.subheader("Closing price vs Time Chart with 100 MA and 200 MA")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)
st.write('---')

# splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range=(0, 1))

data_training_array = scaler.fit_transform(data_training)

# Load my models
model = load_model('keras_model1.h5')

# testing part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


# Making predictions
y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('**PREDICTED PRICE  vs  ORIGINAL PRICE**')
# st.subheader("Predicted Prices vs Original Prices")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
st.write('---')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import load_model


# Define the start and end dates
start = '2010-01-01'
end = '2024-05-16'

st.title("Stock Trend Prediction")

# User input for stock ticker
user_input = st.text_input("Enter Stock Ticker", 'AAPL')

try:
    # Retrieve the data based on user input
    df = yf.download(user_input, start=start, end=end)
    
    if df.empty:
        st.error("No data available for the specified stock ticker symbol.")
    else:
        # Display the first few rows of the dataframe
        st.subheader('Data from 2010-2024')
        st.write(df.describe())
except Exception as e:
    st.error(f"An error occurred: {e}")


st.subheader('Closing Price Chart')
# Plot the closing prices
fig = plt.figure(figsize=(12, 6), facecolor='black')  # Set facecolor to black
plt.plot(df.index, df['Close'], label='Close', color='green')
# Add labels and title
plt.title('Closing Prices', color='white')  # Set title color to white
plt.xlabel('Date', color='white')  # Set x-label color to white
plt.ylabel('Price (USD)', color='white')  # Set y-label color to white
plt.xticks(color='white')  # Set x-axis ticks color to white
plt.yticks(color='white')  # Set y-axis ticks color to white
# Rotate x-axis labels for better readability
plt.xticks(rotation=45)
# Add legend
plt.legend()
# Show plot
plt.tight_layout()
plt.show()

st.pyplot(fig)





# Calculate Moving Averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

# Plot Closing Price with Moving Averages
st.subheader('Closing Price vs Time Chart with 100MA')
fig = plt.figure(figsize=(12, 6), facecolor='black')  # Set facecolor to black
plt.plot(df.index, df['Close'], label='Close', color='green')
plt.plot(df.index, ma100, label='100-day Moving Average', color='blue')
plt.title('Closing Prices with 100-day Moving Average', color='white')  # Set title color to white
plt.xlabel('Date', color='white')  # Set x-label color to white
plt.ylabel('Price (USD)', color='white')  # Set y-label color to white
plt.xticks(color='white')  # Set x-axis ticks color to white
plt.yticks(color='white')  # Set y-axis ticks color to white
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
fig = plt.figure(figsize=(12, 6), facecolor='black')  # Set facecolor to black
plt.plot(df.index, df['Close'], label='Close', color='green')
plt.plot(df.index, ma100, label='100-day Moving Average', color='blue')
plt.plot(df.index, ma200, label='200-day Moving Average', color='red')
plt.title('Closing Prices with 100-day and 200-day Moving Averages', color='white')  # Set title color to white
plt.xlabel('Date', color='white')  # Set x-label color to white
plt.ylabel('Price (USD)', color='white')  # Set y-label color to white
plt.xticks(color='white')  # Set x-axis ticks color to white
plt.yticks(color='white')  # Set y-axis ticks color to white
plt.xticks(rotation=45)
plt.legend()
st.pyplot(fig)




# Splitting data into trainin and testin

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing =  pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)



# Load Model
model = load_model('LSTM_Stock_Model1.keras')

# Predictions

# Testing part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data= scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)


y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor= 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Final Graph
st.subheader('Predictions vs Original')
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig2) 
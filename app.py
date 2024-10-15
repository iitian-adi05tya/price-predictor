import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

#start & today's date
start = "2010-01-01"
today = date.today().strftime("%Y-%m-%d")

st.title('Price Predictor')

# Collection of stocks
stocks = (
    'AAPL', 'AMZN', 'DJI', 'GOOG', 'IXIC', 'META', 
    'MSFT', 'NFLX', 'NIFTY 50', 'NIFTY_FIN_SERVICE.NS', 
    'NSEBANK', 'RELIANCE.NS', 'SBIN.NS', 'TATASTEEL.NS', 
    'TSLA', 'UBER'
)
selected_stock = st.selectbox('Select dataset for prediction', stocks)

# No. of years slider
n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    """Load stock data from Yahoo Finance."""
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

# Loader status
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Done!')

# Display real raw data
st.subheader('Raw data')
st.write(data.tail())

# Plot 
def plot_raw_data(data):
    """Plot raw stock data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data)

# Forecasting by Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

# Model fitting
m = Prophet()
m.fit(df_train)

# Future predictions
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Plot forecast data
st.subheader('Forecast data')
st.write(forecast.tail())

st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)  

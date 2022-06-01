import pydaisi as pyd
import streamlit as st
from bokeh.plotting import figure
import yfinance as yf
import bokeh
import pandas as pd
from datetime import date
from datetime import timedelta

def plot_price(symbol, period):
    title = f"{symbol} stock price evolution for the last {period} days"
    p = figure(title=title, x_axis_label="Date", y_axis_label="Opening Price (USD)",
               x_axis_type='datetime', sizing_mode='stretch_both',background_fill_alpha = 0)
    today = date.today()

    # tickerSymbol = [str(Symbol1), str(Symbol2)]

    full_tickerDf = pd.DataFrame()
    tickerData = yf.Ticker(symbol)

    tickerDf = tickerData.history(period='1d', start=today - timedelta(days = int(period)), end=today)
    # tickerDf['Open'] /= np.max(tickerDf['Open'])
    tickerDf["Symbol"] = symbol
    tickerDf["Date"] = tickerDf.index

    p.line(tickerDf["Date"].tolist(), tickerDf["Open"].tolist(), legend_label=symbol, line_width=2, line_color = 'orange')

    full_tickerDf = pd.concat([full_tickerDf, tickerDf], ignore_index=True)

    return p, full_tickerDf

def get_price(symbol, period):
    
    today = date.today()

    # tickerSymbol = [str(Symbol1), str(Symbol2)]

    full_tickerDf = pd.DataFrame()
    tickerData = yf.Ticker(symbol)

    tickerDf = tickerData.history(period='1d', start=today - timedelta(days = int(period)), end=today)
    # tickerDf['Open'] /= np.max(tickerDf['Open'])
    tickerDf["Symbol"] = symbol
    tickerDf["Date"] = tickerDf.index

    full_tickerDf = pd.concat([full_tickerDf, tickerDf], ignore_index=True)

    return full_tickerDf

def st_ui():
    st.set_page_config(layout = "wide")
    symbol = st.sidebar.text_input("Enter a symbol", "AAPL")
    period = st.sidebar.slider("Time period for stock price plot", 10, 900, 365)
    st.title(f"Stock price for {symbol} over the last {period} days")

    p, full_tickerDf = plot_price(symbol, period)
    st.bokeh_chart(p, use_container_width=True)

    st.dataframe(full_tickerDf)

if __name__ == "__main__":
    st_ui()
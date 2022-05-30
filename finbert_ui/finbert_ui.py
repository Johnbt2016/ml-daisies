import streamlit as st 
import pydaisi as pyd
import pandas as pd
import yfinance as yf
from datetime import date
from datetime import timedelta
from bokeh.plotting import figure
import numpy as np

finbert = pyd.Daisi("laiglejm/Finbert")
google_news = pyd.Daisi("laiglejm/GoogleNews")

def plot_price(symbol, period):
    title = f"{symbol} stock price evolution for the last {period} days"
    p = figure(title=title, x_axis_label="Date", y_axis_label="Opening Price (USD)",
               x_axis_type='datetime', sizing_mode='stretch_both')
    today = date.today()

    # tickerSymbol = [str(Symbol1), str(Symbol2)]

    full_tickerDf = pd.DataFrame()
    tickerData = yf.Ticker(symbol)

    tickerDf = tickerData.history(period='1d', start=today - timedelta(days = int(period)), end=today)
    # tickerDf['Open'] /= np.max(tickerDf['Open'])
    tickerDf["Symbol"] = symbol
    tickerDf["Date"] = tickerDf.index

    p.line(tickerDf["Date"].tolist(), tickerDf["Open"].tolist(), legend_label=symbol, line_width=2)

    full_tickerDf = pd.concat([full_tickerDf, tickerDf], ignore_index=True)

    return p

def compute_sentiments(symbol, nb):
    news = google_news.get_news(symbol, num = nb).value

    try:
        news = news.drop(columns=['desc', 'img', 'media'])
    except:
        pass

    headlines = news['title'].to_list()

    answer = finbert.give_sentiment(headlines).value
    sentiments = [d['label'] for d in answer]
    news['Sentiment'] = sentiments
    results = {'positive':sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')}

    overall_sentiment = max(results, key=results.get)
    overall_score = int(100*results[overall_sentiment]/len(sentiments))

    return news, overall_sentiment, overall_score


def st_ui():
    st.set_page_config(layout = "wide")
    st.title("Stock Sentiment from Google News headlines")

    symbol = st.sidebar.text_input("Enter a symbol", "AAPL")
    nb = st.sidebar.slider("Nb of results to process", 1, 100, 10)
    period = st.sidebar.slider("Time period for stock price plot", 10, 700, 365)

    with st.spinner('Getting news headlines and analyzing sentiment'):
        news, overall_sentiment, overall_score = compute_sentiments(symbol, nb)
    to_print = f"General sentiment for {symbol} is: {overall_sentiment} ({overall_score}%)"

    p = plot_price(symbol, period)
    col1, col2 = st.columns(2)
    with col1:
        st.header(to_print)
        st.dataframe(news)
        st.bokeh_chart(p, use_container_width=True)

    with col2:
        filename = f"assets/{overall_sentiment}.png"
        st.image(filename, width=400)


if __name__ == "__main__":
    st_ui()
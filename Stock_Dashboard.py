#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:21:52 2023
Todos : 
    - Add additional information on Pricing Data
    - Improve readability of Fundamental Data
    - Add a DCF

@author: zishan-lab
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

#initialize streamlit webapp
st.title("Stock Dashboard")
ticker = st.sidebar.text_input('Ticker',value='MSFT')
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')

#download yfinance data
data = yf.download(ticker,start=start_date,end=end_date)

#plotting the data with values below the mean in a lighter color than the ones above the mean
mean_value = data['Adj Close'].mean()
fig = px.line(data, x = data.index, y = data['Adj Close'], title = ticker)

# Add a horizontal line for the mean value
fig.add_hline(y=mean_value, line=dict(color='red', dash='dash'))

st.plotly_chart(fig)

pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    data2 =data
    data2['% change'] = data['Adj Close'] / data['Adj Close'].shift(1) -1
    data2.dropna(inplace = True)
    annual_return = data2['% change'].mean()*252*100
    stdev = np.std(data2["% change"]*np.sqrt(252))
    
    st.write('Annual return is ', round(annual_return,3), '%')
    st.write('Standard deviation is ', round(stdev*100,3), '%')
    st.write('Risk Adj. return is ', round(annual_return/(stdev*100),3), '%')

from alpha_vantage.fundamentaldata import FundamentalData

with fundamental_data:
    
    key = "9G0813X19QOSFDUY"
    fd = FundamentalData(key,output_format="pandas")
    
    st.subheader(f'Balance Sheet of {ticker}')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    st.write(bs)
    
    st.subheader(f'Income Statement of {ticker}')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    st.write(is1)
    
    st.subheader(f'Cash Flow Statement of {ticker}')
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    st.write(cf)
    

from stocknews import StockNews

with news:
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news = sn.read_rss()
    for i in range (10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')
        
    
    
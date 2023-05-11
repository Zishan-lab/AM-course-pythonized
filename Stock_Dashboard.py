#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
link :  https://zishan-lab-am-course-pythonized-stock-dashboard-6ozc6k.streamlit.app/
Created on Mon May  1 18:21:52 2023
Todos : 
    - Add additional information on Pricing Data
    - Improve readability of Fundamental Data : done for numbers, missing spaces added to the indexes
    - Add a DCF
    

@author: zishan-lab
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime

#testing function for formatting numbers of Fundamental data

def format_dataframe(df):
    
    df.index = df.index.str.replace(r'(?<=[a-z])(?=[A-Z])', ' ')
    # Iterate over columns (except index and first row)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: int(x) if x != 'None' else None)
        if df.dtypes[col] == 'float64' or df.dtypes[col] == 'int64':
            # Apply formatting to numerical columns
            df[col] = df[col].apply(lambda x: '${:,.0f}'.format(x / 1000))

#initialize streamlit webapp
default_date = datetime.date(2023,1,3)
st.title("Stock Dashboard")
ticker = st.sidebar.text_input('Ticker',value='MSFT')
start_date = st.sidebar.date_input('Start Date',value=default_date)
end_date = st.sidebar.date_input('End Date')

#download yfinance data
data = yf.download(ticker,start=start_date,end=end_date)

#plotting the data with values below the mean in a lighter color than the ones above the mean
fig = go.Figure() #testing projections
mean_value = data['Adj Close'].mean()

fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Adj Close'],
    mode='markers',
    name='Actuals'
))
#fig = px.line(data, x = data.index, y = data['Adj Close'], title = ticker)

# Add a horizontal line for the mean value
fig.add_hline(y=mean_value, line=dict(color='red', dash='dash'))

last_adj_close = data['Adj Close'].iloc[-1]
next_day = pd.to_datetime(data.index[-1]) + pd.DateOffset(days=1)
fig.add_trace(go.Scatter(
    x=[next_day],
    y=[last_adj_close],
    mode='lines',
    name='Projected Value'
))
# Update x-axis range to include projected value
fig.update_xaxes(range=[data.index[0], next_day])

# Set plot title and axis labels
fig.update_layout(
    title='Adjusted Close Price',
    xaxis_title='Date',
    yaxis_title='Adj Close'
)

# Add annotation with the mean value
mean_value_adj = mean_value + 15
fig.add_annotation(
    x=data.index[0],  # Adjust the x-coordinate for annotation placement
    y= mean_value_adj
    text=f"Mean Value: {mean_value}",
    showarrow=False,
    font=dict(color='red')
)

st.plotly_chart(fig)

pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundamental Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    data2 =data
    data2['% change'] = data['Adj Close'] / data['Adj Close'].shift(1) -1
    data2.dropna(inplace = True)
    annual_return = data2['% change'].mean()*252*100
    stdev = np.std(data2["% change"]*np.sqrt(252))
    vol = np.sqrt((annual_return / 100) * (annual_return/(stdev*100)) * (252) * (mean_value - standard_deviation) ** 2)
    
    st.write('Annual return is ', round(annual_return,3), '%')
    st.write('Standard deviation is ', round(stdev*100,3), '%')
    st.write('Risk Adj. return is ', round(annual_return/(stdev*100),3), '%')
    st.write('Volatility for the period is ', round(vol,3), '%')

from alpha_vantage.fundamentaldata import FundamentalData

with fundamental_data:
    
    key = "9G0813X19QOSFDUY"
    fd = FundamentalData(key,output_format="pandas")
    
    st.subheader(f'Balance Sheet of {ticker} in 000\'s')
    balance_sheet = fd.get_balance_sheet_annual(ticker)[0]
    bs = balance_sheet.T[2:]
    bs.columns = list(balance_sheet.T.iloc[0])
    format_dataframe(bs)
    st.write(bs)
    
    st.subheader(f'Income Statement of {ticker} in 000\'s')
    income_statement = fd.get_income_statement_annual(ticker)[0]
    is1 = income_statement.T[2:]
    is1.columns = list(income_statement.T.iloc[0])
    format_dataframe(is1)
    st.write(is1)
    
    st.subheader(f'Cash Flow Statement of {ticker} in 000\'s')
    cash_flow = fd.get_cash_flow_annual(ticker)[0]
    cf = cash_flow.T[2:]
    cf.columns = list(cash_flow.T.iloc[0])
    format_dataframe(cf)
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
        
    
    

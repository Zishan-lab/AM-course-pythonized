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
from sklearn.linear_model import LinearRegression
from scipy.stats import t

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

#add a projection for the next day using linear regression

df = pd.DataFrame()
df['Adj Close'] = data['Adj Close']
df['Next Day'] = df['Adj Close'].shift(-1)

# Drop the last row since it will have NaN for the "Next Day" value
df = df.dropna()

# Split the data into features (X) and target (y)
X = df[['Adj Close']]
y = df['Next Day']

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the last observed "Adj Close" value to predict the next day's value
last_adj_close = df['Adj Close'].iloc[-1]

# Predict the value for the next day
next_day_prediction = model.predict([[last_adj_close]])

# Create a scatter plot of the 'Adj Close' data
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Adj Close'],
    mode='markers',
    name='Actuals'
))

# Calculate the confidence interval
confidence_interval = 0.95  # Set the desired confidence interval
X_array = np.array(X)
y_pred = model.predict(X_array)
residuals = y - y_pred
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)
t_score = np.abs(t.ppf((1 - confidence_interval) / 2, df=len(X_array) - 1))
margin_error = t_score * std_residual
lower_bound = next_day_prediction - margin_error
upper_bound = next_day_prediction + margin_error

# Add the predicted value for the next day as a scatter point
next_day = df.index[-1] + pd.DateOffset(days=1)
fig.add_trace(go.Scatter(
    x=[next_day],
    y=next_day_prediction,
    mode='markers',
    name='Next Day Prediction'
))

mean_value = data['Adj Close'].mean()

# Add a horizontal line for the mean value
fig.add_hline(y=mean_value, line=dict(color='red', dash='dash'))

# Set plot title and axis labels
fig.update_layout(
    title='Price evolution of ' + ticker,
    xaxis_title='Date',
    yaxis_title='Adj Close'
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
    vol = volatility = np.sqrt((annual_return / 100) * (annual_return/stdev) * 252 * stdev ** 2)
    
    st.write('Mean price of the stock for the period is ', round(mean_value,3), " $")
    st.write('Annual return is ', round(annual_return,3), '%')
    st.write('Standard deviation is ', round(stdev*100,3), '%')
    st.write('Risk Adj. return is ', round(annual_return/(stdev*100),3), '%')
    st.write('Volatility for the period is ', round(vol,3), '%')
    st.write("Next Day Price Prediction:", round(next_day_prediction[0],3), " $")
    st.write("Lower Bound ({}% confidence):".format(confidence_interval * 100), round(lower_bound[0],3), " $")
    st.write("Upper Bound ({}% confidence):".format(confidence_interval * 100), round(upper_bound[0],3, " $")

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
        
    
    

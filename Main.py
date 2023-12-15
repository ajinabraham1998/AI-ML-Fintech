#For this code we will be showing the greaphs in streamlit so please pip install these below libraries if not already present
#Thanks
import streamlit as st
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import math
import scipy.stats as ss
import numpy as np
import pandas as pd


def tesla_stock():
    # Download stock data
    st.set_page_config(page_title="Tesla Stock Analysis", layout="wide")
    st.title(" Tesla Stock Analysis Dashboard")

    STOCK = yf.download('TSLA')

    st.header("Tesla Inc. (TSLA) Stock Trends")
    st.image('https://source.unsplash.com/featured/?tesla,stock', use_column_width=True)


    # Plot 1: Kernel Density Estimation

    st.subheader('Kernel Density Estimation of Adjusted Close Prices')
    plt.figure(1)
    sns.kdeplot(data=STOCK['Adj Close'].values, linewidth=4)
    plt.title('Kernel Density Estimation')
    st.pyplot(plt)

    # Display summary statistics for 'Adj Close'
    st.subheader('Summary Statistics for Adj Close')
    st.write(STOCK['Adj Close'].describe())

    # Plot 2: Kernel Density Estimation (Monthly)
    st.subheader('Monthly Kernel Density Estimation of Adjusted Close Prices')
    plt.figure(2)
    MONTH = STOCK.resample('M').mean()
    sns.kdeplot(data=MONTH['Adj Close'], linewidth=4)
    plt.title('Kernel Density Estimation (Monthly)')
    st.pyplot(plt)

    # Display summary statistics for 'Adj Close' (Monthly)
    st.subheader('Summary Statistics for Adj Close (Monthly)')
    st.write(MONTH['Adj Close'].describe())

    # Generate random number
    u = np.random.normal(0, 1, 1)

    # Parameters for distribution 1
    mu_1 = 175
    Sigma1 = 625
    r1 = mu_1 + math.sqrt(Sigma1) * u
    sigma_1 = math.sqrt(Sigma1)
    x_1 = np.linspace(50, 400, 350)
    y_1 = ss.norm.pdf(x_1, mu_1, sigma_1)

    # Parameters for distribution 2
    mu_2 = 325
    Sigma2 = 625
    r2 = mu_2 + math.sqrt(Sigma2) * u
    sigma_2 = math.sqrt(Sigma2)
    x_2 = np.linspace(50, 400, 350)
    y_2 = ss.norm.pdf(x_2, mu_2, sigma_2)

    # Plot 1: Distribution 1
    plt.figure(2)
    plt.plot(x_1, y_1)
    plt.title('Distribution 1')
    st.pyplot(plt)

    # Plot 2: Distribution 2
    plt.figure(3)
    plt.plot(x_2, y_1)
    plt.title('Distribution 2')
    st.pyplot(plt)

    # Parameters for mixture distribution
    p = 0.8
    S = 5000
    mu_1 = 175
    Sigma1 = 625
    mu_2 = 325
    Sigma2 = 625

    # Generate mixture distribution
    st.header("Mixture Distribution")
    r = np.zeros(S)
    for s in range(1, S):
        eps = np.random.normal(0, 1, 1)
        r1 = mu_1 + math.sqrt(Sigma1) * eps
        r2 = mu_2 + math.sqrt(Sigma2) * eps
        u = np.random.uniform(0, 1, 1)
        r[s] = r1 * (u < p) + r2 * (u >= p)

    # Plot: Histogram
    st.header("Custom Stock Analysis")
    plt.figure(5)
    plt.hist(r, bins=np.linspace(50, 400, 350), density=True)
    plt.title('Mixture Distribution')
    st.pyplot(plt)

    # Set Streamlit app title
    st.title("TESLA Stock: Monthly Mean with Standard Deviation")

    # Get user input for stock symbol
    symbol = st.text_input("Enter stock symbol (e.g., VWAGY)", value="TSLA", max_chars=5)

    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(symbol)

    # Check if stock data is available
    if stock_data.empty:
        st.error("No data found for the entered stock symbol. Please try again.")
    else:
        # Resample the data to monthly frequency
        monthly_data = stock_data.resample('M').mean()

        # Calculate mean and standard deviation for each month
        monthly_mean = monthly_data['Close']
        monthly_std = stock_data['Close'].resample('M').std()

        # Create a DataFrame to store the results
        result_df = pd.DataFrame({'Month': monthly_data.index.strftime('%Y-%m'),
                                  'Mean': monthly_mean,
                                  'Standard Deviation': monthly_std})

        # Display the results in a table
        st.write(result_df)

        # Plot monthly mean and standard deviation
        fig1, ax = plt.subplots(figsize=(7, 6))

        # Line plot for monthly mean
        ax.plot(monthly_data.index, monthly_mean, color='b', label='Mean')

        # Set plot title and axis labels
        ax.set_title(f"{symbol} Monthly Mean")
        ax.set_xlabel("Month")
        ax.set_ylabel("Price")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Display legend
        ax.legend()

        # Render the plot using Matplotlib
        st.pyplot(fig1)

        # Plot monthly mean and standard deviation
        fig2, ax = plt.subplots(figsize=(10, 6))

        # Line plot for standard deviation
        ax.plot(monthly_data.index, monthly_std, color='r', label='Standard Deviation')

        # Set plot title and axis labels
        ax.set_title(f"{symbol} Monthly Standard Deviation")
        ax.set_xlabel("Month")
        ax.set_ylabel("Price")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Display legend
        ax.legend()

        # Render the plot using Matplotlib
        st.pyplot(fig2)

    # Set Streamlit app title
    st.title("Tesla Stock: Moving Averages")

    # Check if stock data is available
    if stock_data.empty:
        st.error("No data found for the entered stock symbol. Please try again.")
    else:
        # Calculate the moving averages for the last 6 months
        last_6_months_data = stock_data.tail(6 * 30)
        moving_averages = last_6_months_data['Close'].rolling(window=30).mean()

        # Plot moving averages
        fig, ax = plt.subplots(figsize=(10, 6))

        # Line plot for stock closing prices
        ax.plot(last_6_months_data.index, last_6_months_data['Close'], color='b', label='Closing Price')

        # Line plot for moving averages
        ax.plot(last_6_months_data.index, moving_averages, color='r', label='30-day Moving Average')

        # Set plot title and axis labels
        ax.set_title(f"{symbol} Moving Averages (Last 6 Months)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)

        # Display legend
        ax.legend()

        # Render the plot using Matplotlib
        st.pyplot(fig)


# Run the function
tesla_stock()
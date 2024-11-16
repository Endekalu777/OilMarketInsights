import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
import ruptures as rpt
from IPython.display import display
from statsmodels.tsa.seasonal import seasonal_decompose

# Create logs directory if it doesn't exist
if not os.path.exists('../logs'):
    os.makedirs('../logs')
logging.basicConfig(level=logging.INFO,
                    handlers = [logging.FileHandler('../logs/Oil_price_analysis.log')],
                    format='%(asctime)s:%(levelname)s:%(message)s')

class BrentOilAnalysis:
    def load_data(self, file_path):
        """
        Load and preprocess the Brent oil price data.
        """
        try:
            logging.info(f"Loading data from {file_path}.")
            data = pd.read_csv(file_path)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data.sort_index(inplace=True)
            logging.info("Data loaded successfully.")
            return data
        except Exception as e:
            self.logging.error(f"Failed to load data: {e}")
            raise

    def clean_data(self, df):
        """
        Handle missing values and ensure data consistency.
        """
        try:
            logging.info("Cleaning data and handling missing values.")
            missing_summary = df.isnull().sum()
            df.interpolate(method='time', inplace=True)
            df.ffill(inplace=True)
            display(df.describe())
            logging.info("Data cleaning completed.")
            return df, missing_summary
        except Exception as e:
            logging.error(f"Failed to clean data: {e}")
            raise

    def calculate_technical_indicators(self, df):
        """
        Add technical indicators such as returns, volatility, and moving averages.
        """
        try:
            logging.info("Calculating technical indicators.")
            df['Returns'] = df['Price'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=30).std()
            df['MA_50'] = df['Price'].rolling(window=50).mean()
            df['MA_200'] = df['Price'].rolling(window=200).mean()
            df['Momentum'] = df['Price'].pct_change(periods=20)
            df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1))
            df.dropna(inplace=True)
            logging.info("Technical indicators added successfully.")
            return df
        except Exception as e:
            self.logging.error(f"Failed to calculate technical indicators: {e}")
            raise

    def test_stationarity(self, series):
        """
        Perform the Augmented Dickey-Fuller test for stationarity.
        """
        try:
            logging.info("Performing stationarity test (ADF).")
            adf_result = adfuller(series.dropna())
            result = {
                'ADF Statistic': adf_result[0],
                'p-value': adf_result[1],
                'Critical Values': adf_result[4]
            }
            logging.info(f"Stationarity test completed: {result}")
            return result
        except Exception as e:
            logging.error(f"Failed to perform stationarity test: {e}")
            raise

    def detect_structural_breaks(self, series):
        """
        Identify structural breaks using the PELT algorithm.
        """
        try:
            logging.info("Detecting structural breaks.")
            algo = rpt.Pelt(model='rbf').fit(series.values)
            change_points = algo.predict(pen=10)
            result = {
                'Change Points': change_points,
                'Number of Changes': len(change_points)
            }
            logging.info(f"Structural breaks detected: {result}")
            return result
        except Exception as e:
            logging.error(f"Failed to detect structural breaks: {e}")
            raise

    def perform_seasonality_analysis(self, series, period=300):
        """
        Decompose the time series into trend, seasonal, and residual components.
        """
        try:
            logging.info("Performing seasonality analysis.")
            decomposition = seasonal_decompose(series, model='additive', period=period)
            
            # Drop NaN values from each component
            result = {
                'Trend': decomposition.trend.dropna(),
                'Seasonal': decomposition.seasonal.dropna(),
                'Residual': decomposition.resid.dropna()
            }
            
            logging.info("Seasonality analysis completed.")
            return result
        except Exception as e:
            logging.error(f"Failed to perform seasonality analysis: {e}")
            raise

    def plot_time_series_with_indicators(self, df):
        """
        Plot price data alongside moving averages.
        """
        try:
            logging.info("Plotting time series with indicators.")
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Price'], label='Price', color='blue')
            plt.plot(df.index, df['MA_50'], label='50-Day MA', color='orange')
            plt.plot(df.index, df['MA_200'], label='200-Day MA', color='green')
            plt.title('Brent Oil Price with Moving Averages')
            plt.xlabel('Date')
            plt.ylabel('Price (USD)')
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot time series: {e}")
            raise

    def plot_returns_distribution(self, df):
        """
        Visualize the distribution of daily returns.
        """
        try:
            logging.info("Plotting returns distribution.")
            plt.figure(figsize=(10, 5))
            sns.histplot(df['Returns'].dropna(), bins=50, kde=True, color='purple')
            plt.title('Distribution of Daily Returns')
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot returns distribution: {e}")
            raise

    def plot_volatility_over_time(self, df):
        """
        Plot rolling volatility over time.
        """
        try:
            logging.info("Plotting rolling volatility over time.")
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Volatility'], color='red')
            plt.title('30-Day Rolling Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot volatility: {e}")
            raise

    def plot_outliers_with_boxplot(self, series, title="Boxplot with Outliers"):
        """
        Plot a boxplot of the data and highlight outliers.
        
        Parameters:
            series (pd.Series): Input series (e.g., Price, Returns, etc.)
            title (str): Title of the plot.
        """
        try:
            logging.info("Plotting boxplot with outliers.")
            plt.figure(figsize=(8, 6))
            sns.boxplot(x=series, color="skyblue", showmeans=True)
            plt.title(title)
            plt.xlabel(series.name)
            plt.grid(True)
            plt.show()
        except Exception as e:
            logging.error(f"Failed to plot boxplot: {e}")
            raise
    



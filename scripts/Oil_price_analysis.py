import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from datetime import datetime
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from IPython.display import display

class OilPriceAnalysis:
    def __init__(self, oil_prices_file, inflation_file, gdp_file, unemployment_file):
        # Load the datasets
        self.oil_prices = pd.read_csv(oil_prices_file, parse_dates=['Date'], dayfirst=True)
        self.inflation = pd.read_csv(inflation_file, parse_dates=['Date'], dayfirst=True)
        self.gdp = pd.read_csv(gdp_file, parse_dates=['DATE'])
        self.unemployment = pd.read_csv(unemployment_file, parse_dates=['Date'], dayfirst=True)
        self.data = None

    def preprocess_data(self):
        # Process oil prices, resampling to annual frequency
        self.oil_prices['Year'] = self.oil_prices['Date'].dt.year
        self.oil_prices['Price'] = pd.to_numeric(self.oil_prices['Price'], errors='coerce')
        annual_oil = self.oil_prices.groupby('Year', as_index=False)['Price'].mean()

        # Align dates for other data sources
        self.inflation['Year'] = self.inflation['Date'].dt.year
        self.unemployment['Year'] = self.unemployment['Date'].dt.year
        self.gdp['Year'] = self.gdp['DATE'].dt.year

        # Merge datasets on 'Year'
        self.data = pd.merge(annual_oil[['Year', 'Price']], self.inflation[['Year', 'Inflation Rate']], on='Year')
        self.data = pd.merge(self.data, self.gdp[['Year', 'GDP']], on='Year')
        self.data = pd.merge(self.data, self.unemployment[['Year', 'Unemployment Rate']], on='Year')

        # Handle missing data
        self.data.dropna(inplace=True)

    def plot_data(self):
        # Plot multiple economic indicators alongside oil prices
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.data['Year'], self.data['Price'], label='Brent Oil Price', color='blue')
        plt.title('Brent Oil Price')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.plot(self.data['Year'], self.data['Inflation Rate'], label='Inflation Rate', color='orange')
        plt.title('Inflation Rate')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.plot(self.data['Year'], self.data['GDP'], label='GDP', color='green')
        plt.title('GDP')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot(self.data['Year'], self.data['Unemployment Rate'], label='Unemployment Rate', color='red')
        plt.title('Unemployment Rate')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def fit_var_model(self):
        # Fit a VAR model
        model_data = self.data[['Price', 'Inflation Rate', 'GDP', 'Unemployment Rate']]
        model_data_diff = model_data.diff().dropna()
        model = VAR(model_data_diff)
        fitted_model = model.fit(2)
        print(fitted_model.summary())

        # Forecast future values
        forecast = fitted_model.forecast(model_data_diff.values[-2:], steps=5)
        forecast_df = pd.DataFrame(forecast, columns=['Price', 'Inflation Rate', 'GDP', 'Unemployment Rate'])
        display(forecast_df.head())

    def fit_markov_switching_model(self):
        oil_prices_diff = self.data['Price'].diff().dropna()
        
        # Using MarkovRegression for regime changes in mean with a constant term
        model = MarkovRegression(oil_prices_diff, k_regimes=2, trend='c', switching_variance=True)
        fitted_model = model.fit()
        print(fitted_model.summary())

    
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

    def fit_lstm_model(self):
        try:
            # Normalize data
            data = self.data[['Price', 'Inflation Rate', 'GDP', 'Unemployment Rate']].values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Reduce lookback period based on data size
            lookback = min(12, len(scaled_data) // 4)  # Adjust lookback period dynamically
            
            # Check if we have enough data
            if len(scaled_data) < lookback + 1:
                print(f"Not enough data points. Need at least {lookback + 1} points, but got {len(scaled_data)}")
                return

            # Prepare training data for LSTM
            X, y = [], []
            for i in range(lookback, len(scaled_data)):
                X.append(scaled_data[i-lookback:i])
                y.append(scaled_data[i, 0])  # Predicting oil price (Price)

            X, y = np.array(X), np.array(y)
            
            # Check if we have any samples
            if len(X) == 0 or len(y) == 0:
                print("No samples could be created with the current lookback period")
                return

            # Split data into training and test sets
            split_idx = max(1, int(0.8 * len(X)))  # Ensure at least 1 training sample
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Check if we have enough data for training and testing
            if len(X_train) < 1 or len(X_test) < 1:
                print("Not enough data for training and testing split")
                return

            # Build LSTM model
            model = Sequential()
            model.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(LSTM(units=32))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Adjust batch size based on data size
            batch_size = min(32, len(X_train))
            model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=1)

            # Predict on test set
            predicted_oil_price = model.predict(X_test)
            
            # Reshape predictions and actual values for inverse transformation
            predicted_oil_price_padded = np.concatenate((predicted_oil_price, np.zeros((predicted_oil_price.shape[0], 3))), axis=1)
            predicted_oil_price = scaler.inverse_transform(predicted_oil_price_padded)[:, 0]

            y_test_padded = np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 3))), axis=1)
            y_test_rescaled = scaler.inverse_transform(y_test_padded)[:, 0]

            # Calculate Mean Squared Error (MSE)
            mse = mean_squared_error(y_test_rescaled, predicted_oil_price)
            print(f"Mean Squared Error (MSE): {mse}")

            # Plot actual vs predicted prices
            plt.figure(figsize=(10, 6))
            plt.plot(y_test_rescaled, label='Actual Oil Price')
            plt.plot(predicted_oil_price, label='Predicted Oil Price')
            plt.title('LSTM Model - Actual vs Predicted Oil Prices')
            plt.xlabel('Time Steps')
            plt.ylabel('Oil Price')
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print(f"Data shape: {data.shape}")
            print(f"Scaled data shape: {scaled_data.shape}")

    
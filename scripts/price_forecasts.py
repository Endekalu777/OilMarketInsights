import os
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from datetime import timedelta

# Create a logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler('logs/price_forecasts.log')],
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class OilPriceAnalysis:
    """
    A class for analyzing and forecasting oil prices using ARIMA, GARCH, VAR, and LSTM models.
    """

    def __init__(self, data):
        """
        Initialize the OilPriceAnalysis object.

        Parameters:
        data (pd.DataFrame): DataFrame containing oil price data with a DateTimeIndex.
        """
        # Ensure Date is in datetime format
        if 'Date' not in data.columns:
            raise ValueError("The dataset must have a 'Date' column.")
        
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Index must be a DateTimeIndex.")
        
        self.data = data
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        logging.info("OilPriceAnalysis initialized with data of shape %s", data.shape)

    def prepare_data(self, test_size=0.2):
        """
        Prepare the data for training and testing, including feature scaling.

        Parameters:
        test_size (float): The proportion of data to be used as the test set.
        """
        try:
            train_size = int(len(self.data) * (1 - test_size))
            self.train = self.data[:train_size]
            self.test = self.data[train_size:]

            # Scale the data
            self.scaler = MinMaxScaler()
            self.scaled_data = self.scaler.fit_transform(self.data[['Price']])
            self.scaled_train = self.scaled_data[:train_size]
            self.scaled_test = self.scaled_data[train_size:]

            logging.info("Data prepared: train size = %d, test size = %d", len(self.train), len(self.test))
        except Exception as e:
            logging.error("Error in prepare_data: %s", str(e))

    def create_sequences(self, data, seq_length):
        """
        Create sequences of data for LSTM models.

        Parameters:
        data (np.ndarray): Scaled data array.
        seq_length (int): The sequence length for each input sequence.

        Returns:
        Tuple (np.ndarray, np.ndarray): Input (X) and output (y) sequences.
        """
        try:
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:(i + seq_length), 0])
                y.append(data[i + seq_length, 0])
            logging.info("Sequences created successfully.")
            return np.array(X), np.array(y)
        except Exception as e:
            logging.error("Error creating sequences: %s", str(e))
            return None, None

    def fit_arima(self, order=(5, 1, 0)):
        """
        Fit an ARIMA model to the training data.

        Parameters:
        order (tuple): The (p, d, q) order of the ARIMA model.
        """
        try:
            model = ARIMA(self.train['Price'], order=order)
            self.models['arima'] = model.fit()
            logging.info("ARIMA model fitted with order: %s", order)

            self.predictions['arima'] = self.models['arima'].forecast(steps=len(self.test))
            self.calculate_metrics('arima')
        except Exception as e:
            logging.error("Error fitting ARIMA: %s", str(e))

    def fit_garch(self, p=1, q=1):
        """
        Fit a GARCH model to the training data.

        Parameters:
        p (int): Order of the ARCH component.
        q (int): Order of the GARCH component.
        """
        try:
            returns = self.train['Price'].pct_change().dropna()
            model = arch_model(returns, vol='Garch', p=p, q=q)
            self.models['garch'] = model.fit(disp="off")
            logging.info("GARCH model fitted with p=%d, q=%d", p, q)

            garch_forecast = self.models['garch'].forecast(horizon=len(self.test))
            self.predictions['garch'] = garch_forecast.mean.values[-1]
            self.calculate_metrics('garch')
        except Exception as e:
            logging.error("Error fitting GARCH: %s", str(e))

    def fit_auto_arima(self, seasonal=False, m=1):
        """
        Fit an auto ARIMA model to the training data.

        Parameters:
        seasonal (bool): Whether to fit a seasonal ARIMA model.
        m (int): Number of periods in a seasonal cycle (e.g., 12 for monthly data).
        """
        try:
            self.models['auto_arima'] = auto_arima(
                self.train['Price'], seasonal=seasonal, m=m, trace=True,
                suppress_warnings=True, stepwise=True, error_action='ignore'
            )
            logging.info("Auto ARIMA fitted with order: %s", self.models['auto_arima'].order)

            self.predictions['auto_arima'] = self.models['auto_arima'].predict(n_periods=len(self.test))
            self.calculate_metrics('auto_arima')
        except Exception as e:
            logging.error("Error fitting Auto ARIMA: %s", str(e))

    def fit_var(self, maxlags=5):
        """
        Fit a VAR model to the training data.

        Parameters:
        maxlags (int): Maximum number of lags to consider.
        """
        try:
            # Select features for VAR model
            features = ['Price', 'Returns', 'Volatility', 'Momentum']
            var_data = self.train[features]

            model = VAR(var_data)
            self.models['var'] = model.fit(maxlags=maxlags)
            logging.info("VAR model fitted with maxlags %d", maxlags)

            # Make predictions
            lag_order = self.models['var'].k_ar
            forecast = self.models['var'].forecast(var_data.values[-lag_order:], steps=len(self.test))
            self.predictions['var'] = pd.DataFrame(forecast, columns=features, index=self.test.index)
            logging.info("VAR predictions made for test set")

            # Calculate metrics
            self.calculate_metrics('var')
        except Exception as e:
            logging.error("Error fitting VAR model: %s", str(e))

    def fit_lstm(self, seq_length=60, epochs=50, batch_size=32, save_model_path="../models/lstm_model.h5"):
        try:
            X_train, y_train = self.create_sequences(self.scaled_train, seq_length)
            X_test, y_test = self.create_sequences(self.scaled_test, seq_length)

            # Reshape data for LSTM
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

            # Define the LSTM model
            model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

            # Store the model
            self.models['lstm'] = model

            # Save the model
            model.save(save_model_path)
            logging.info(f"LSTM model saved to {save_model_path}")
                
            # Generate predictions
            lstm_predictions = model.predict(X_test).reshape(-1)
            predictions_scaled = lstm_predictions.reshape(-1, 1)
            predictions_original = self.scaler.inverse_transform(predictions_scaled)
            
            # Store both the predictions and the indices where we have predictions
            self.predictions['lstm'] = predictions_original.flatten()
            self.lstm_test_indices = self.test.index[seq_length:]  # Store indices for plotting
            
            # Calculate metrics using the aligned data
            self.calculate_metrics('lstm')
            
            logging.info(f"LSTM model fitted successfully. Predictions shape: {predictions_original.shape}")
            
        except Exception as e:
            logging.error(f"Error fitting LSTM: {str(e)}")
            raise

    def calculate_metrics(self, model_name):
        """
        Calculate performance metrics for a given model.

        Parameters:
        model_name (str): The name of the model.
        """
        try:
            if model_name not in self.predictions:
                raise ValueError(f"No predictions available for model: {model_name}")

            y_true = self.test['Price'].values
            y_pred = self.predictions[model_name]

            self.metrics[model_name] = {
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'R2': r2_score(y_true, y_pred)
            }
            logging.info("Metrics calculated for %s: %s", model_name, self.metrics[model_name])
        except Exception as e:
            logging.error("Error calculating metrics for %s: %s", model_name, str(e))

    def plot_all_results(self):
        """
        Plot actual vs predicted values for all models in a single figure with different colors.
        """
        try:
            plt.figure(figsize=(12, 7))
            
            # Plot actual prices
            plt.plot(self.test.index, self.test['Price'], label='Actual Prices', color='#000000', linewidth=2)
            
            # Define colors for different models
            colors = {
                'arima': '#FF1E1E',          # Bright Red
                'garch': '#00FF7F',          # Spring Green
                'auto_arima': '#1E90FF',     # Dodger Blue
                'lstm': '#FFD700',           # Gold
                'var': '#FF1493',           # Deep Pink
                'actual': '#4B0082'         # Indigo (for actual prices)
                }
            
            # Plot predictions for each model
            for model_name in self.predictions:
                if model_name == 'var':
                    # Extract only 'Price' column for VAR predictions
                    if 'Price' in self.predictions['var'].columns:
                        plt.plot(
                            self.test.index, 
                            self.predictions['var']['Price'], 
                            label=f'{model_name.upper()} Predictions', 
                            color=colors.get(model_name, '#FF6347'), 
                            linewidth=1.5
                        )
                elif model_name == 'lstm':
                    # Use specific indices for LSTM predictions
                    plt.plot(
                        self.lstm_test_indices, 
                        self.predictions[model_name], 
                        label=f'{model_name.upper()} Predictions', 
                        color=colors.get(model_name, '#FFD700'), 
                        linewidth=1.5
                    )
                else:
                    # Plot other model predictions
                    plt.plot(
                        self.test.index, 
                        self.predictions[model_name], 
                        label=f'{model_name.upper()} Predictions', 
                        color=colors.get(model_name, '#1E90FF'), 
                        linewidth=1.5
                    )

            plt.title('All Models Predictions vs Actual Prices', fontsize=14, pad=20)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Price', fontsize=12)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(f"Error plotting results: {str(e)}")
            print(f"Error: {str(e)}")

    def forecast_future(self, seq_length=60, n_steps=360, save_path="../models/lstm_forecasted_prices.csv"):
        """
        Forecast future values using the trained LSTM model.

        Parameters:
        - seq_length: Length of the input sequence for prediction.
        - n_steps: Number of steps to forecast into the future.
        - save_path: File path to save the forecasted values.

        Returns:
        - A DataFrame with forecasted dates and values.
        """
        try:
            # Use the last 'seq_length' data points to predict future
            last_sequence = self.scaled_data[-seq_length:].reshape((1, seq_length, 1))
            future_predictions = []

            for _ in range(n_steps):
                # Predict the next value
                next_value = self.models['lstm'].predict(last_sequence)[0, 0]
                future_predictions.append(next_value)

                # Update the last sequence
                last_sequence = np.append(last_sequence[:, 1:, :], [[[next_value]]], axis=1)

            # Scale predictions back to the original range
            future_predictions_scaled = np.array(future_predictions).reshape(-1, 1)
            future_predictions_original = self.scaler.inverse_transform(future_predictions_scaled).flatten()

            # Create a DataFrame with forecasted dates
            last_date = self.data.index[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, n_steps + 1)]
            forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Forecasted_Price": future_predictions_original
            })

            # Save to CSV
            forecast_df.to_csv(save_path, index=False)
            logging.info(f"Forecast saved to {save_path}")
            return forecast_df

        except Exception as e:
            logging.error(f"Error during forecasting: {str(e)}")
            raise
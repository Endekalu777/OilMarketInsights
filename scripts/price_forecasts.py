import os
import logging
from sklearn.preprocessing import MinMaxScaler

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
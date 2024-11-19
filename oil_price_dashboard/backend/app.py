import os

# Suppress TensorFlow oneDNN message and other logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.losses import MeanSquaredError

app = Flask(__name__)

# Load the data
historical_prices = pd.read_csv("data/BrentOilPrices.csv")
forecast = pd.read_csv("data/lstm_forecasted_prices.csv")
brent_indicators = pd.read_csv('data/Oil_indicators.csv')

# Load LSTM model
lstm_model = load_model('data/lstm_model.h5', custom_objects={'mse': MeanSquaredError()})

# Convert DataFrames to JSON-serializable format (list of dictionaries)
historical_prices_json = historical_prices.to_dict(orient='records')
forecast_json = forecast.to_dict(orient='records')

# Selected features for training/prediction
FEATURES = ['Returns', 'Volatility', 'MA_50', 'MA_200', 'Momentum', 'Log_Returns']

@app.route('/api/historical-prices', methods=['GET'])
def get_historical_prices():
    return jsonify(historical_prices_json)

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    return jsonify(forecast_json)

@app.route('/api/indicators', methods=['GET'])
def get_indicators():
    return jsonify(brent_indicators.to_dict(orient='records'))

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Accepts a JSON payload with features for prediction:
    {
        "features": {
            "Returns": value,
            "Volatility": value,
            "MA_50": value,
            "MA_200": value,
            "Momentum": value,
            "Log_Returns": value
        }
    }
    """
    try:
        # Extract features from request
        features = request.json['features']
        feature_values = [features[feature] for feature in FEATURES]

        # Prepare data for LSTM model
        input_data = np.array(feature_values).reshape(1, -1)  

        # Predict price
        prediction = lstm_model.predict(input_data).flatten()  

        return jsonify({'predicted_price': prediction[0]})
    except KeyError as e:
        return jsonify({'error': f'Missing feature in request: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)
import pandas as pd
from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the data
historical_prices = pd.read_csv("../data/BrentOilPrices.csv")
forecast = pd.read_csv("../data/lstm_forecasted_prices.csv")
brent_indicators = pd.read_csv('data/Brent_oil_indicators.csv')

# Load LSTM model
lstm_model = load_model('data/lstm_model.h5')

# Convert DataFrames to JSON-serializable format (list of dictionaries)
historical_prices_json = historical_prices.to_dict(orient='records')
forecast_json = forecast.to_dict(orient='records')

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
    data = request.json['features']
    prediction = lstm_model.predict(np.array(data).reshape(1, -1)).tolist()
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
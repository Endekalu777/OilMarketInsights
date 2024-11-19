import pandas as pd
from flask import Flask, jsonify

app = Flask(__name__)

# Load the data
historical_prices = pd.read_csv("../data/BrentOilPrices.csv")
forecast = pd.read_csv("../data/lstm_forecasted_prices.csv")

# Convert DataFrames to JSON-serializable format (list of dictionaries)
historical_prices_json = historical_prices.to_dict(orient='records')
forecast_json = forecast.to_dict(orient='records')

@app.route('/api/historical-prices', methods=['GET'])
def get_historical_prices():
    return jsonify(historical_prices_json)

@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    return jsonify(forecast_json)

if __name__ == '__main__':
    app.run(debug=True)
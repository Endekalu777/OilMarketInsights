from flask import Flask, jsonify, request
from flask_cors import CORS
from models.OilPriceAnalysis import *
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# Initialize the OilPriceAnalysis class with your data
analysis = OilPriceAnalysis(
    'data/BrentOilPrices.csv',
    'data/country_inflation_data/United_States_Inflation_Rate.csv',
    'data/GDP/GDPUS.csv',
    'data/unemployment_rate/United_States_Unemployment_Rate.csv'
)
analysis.preprocess_data()

@app.route('/api/price-data', methods=['GET'])
def get_price_data():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    try:
        data = analysis.get_filtered_price_data("1987", "2022")
        return jsonify({
            'success': True,
            'data': data.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    try:
        # Get date range from query parameters or use defaults
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Default to 1 year of data
        
        if request.args.get('start_date'):
            start_date = datetime.strptime(request.args.get('start_date'), '%Y-%m-%d')
        if request.args.get('end_date'):
            end_date = datetime.strptime(request.args.get('end_date'), '%Y-%m-%d')

        # Get all required data
        price_data = analysis.get_filtered_price_data(start_date, end_date)
        metrics = analysis.calculate_metrics()
        events = analysis.get_significant_events()
        forecast = analysis.get_price_forecast()

        return jsonify({
            'success': True,
            'data': {
                'price_data': price_data.to_dict('records'),
                'metrics': metrics,
                'events': events,
                'forecast': forecast
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True) 
from flask import Flask, jsonify, request
from flask_cors import CORS
from models.OilPriceAnalysis import *
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
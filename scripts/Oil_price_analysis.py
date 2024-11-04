import pandas as pd

class OilPriceAnalysis:
    def __init__(self, oil_prices_file, inflation_file, gdp_file, unemployment_file):
        # Load the datasets
        self.oil_prices = pd.read_csv(oil_prices_file, parse_dates=['Date'], dayfirst=True)
        self.inflation = pd.read_csv(inflation_file, parse_dates=['Date'], dayfirst=True)
        self.gdp = pd.read_csv(gdp_file, parse_dates=['DATE'])
        self.unemployment = pd.read_csv(unemployment_file, parse_dates=['Date'], dayfirst=True)
        self.data = None

    
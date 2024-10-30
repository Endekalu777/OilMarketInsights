import pandas as pd

class EDAEventAnalysis():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df = pd.read_csv(filepath)
        # Convert 'Date' column to datetime format if not already
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        # Sort by Date to ensure correct order
        self.df = self.df.sort_values(by='Date')
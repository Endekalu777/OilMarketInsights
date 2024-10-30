import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class PreProcessor():
    # Initialize the Preprocessor with a path to the CSV data file.
    def __init__(self, filepath):
        self.data_path = filepath
        self.df = None
        

    # Load Brent oil prices data from the CSV file.
    def load_data(self):
        self.df = pd.read_csv(self.data_path, parse_dates = ['Date'], dayfirst = True)
        self.df.columns = self.df.columns.str.strip()
        self.df.sort_values('Date', inplace = True)
        logging.info("Data loaded and sorted by date")
        return self.df

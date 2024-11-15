import pandas as pd
import logging
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Create logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler('logs/pre_process.log')],
                    format='%(asctime)s:%(levelname)s:%(message)s')

class PreProcessor:
    def __init__(self, brent_oil_path, external_indicators_path):
        self.brent_oil_path = brent_oil_path  
        self.external_indicators_path = external_indicators_path  
        self.df = None
        self.brent_df = None
        self.merged_df = None
        logging.info(f"Preprocessor initialized with Brent oil file: {brent_oil_path} and External indicators file: {external_indicators_path}")
        
    def load_brent_data(self):
        try:
            self.df = pd.read_csv(self.brent_oil_path)
            self.df['Price'] = self.df['Price'].astype('float64')

            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%b-%y', errors='coerce')
                self.df['year'] = self.df['Date'].dt.year
                self.df.sort_values('Date', inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                self.df = self.df.rename(columns={'Date': 'date'})

            logging.info("Main data loaded and sorted by date")
            return self.df

        except Exception as e:
            logging.error(f"Error loading main data: {str(e)}")
            raise

    def load_external_data(self):
        try:
            self.brent_df = pd.read_csv(self.external_indicators_path)
            self.brent_df['date'] = pd.to_datetime(self.brent_df['date'])
            self.brent_df['year'] = self.brent_df['date'].dt.year

            if 'country' not in self.brent_df.columns:
                self.brent_df['country'] = 'Unknown'

            logging.info("External indicators data loaded")
            return self.brent_df
            
        except Exception as e:
            logging.error(f"Error loading external data: {str(e)}")
            raise

    def merge_data(self):
        try:
            merged_df = pd.merge(
                self.df,
                self.brent_df,
                on='year',
                how='inner'
            )
            
            self.merged_df = merged_df
            logging.info(f"Data merged successfully with {len(self.merged_df)} records")
            return self.merged_df
        except Exception as e:
            logging.error(f"Error merging data: {str(e)}")
            raise
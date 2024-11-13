import pandas as pd
import logging
from IPython.display import display
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
warnings.filterwarnings("ignore")

# Create logs directory if it doesn't exist
if not os.path.exists('../logs'):
    os.makedirs('../logs')
logging.basicConfig(level=logging.INFO,
                    handlers = [logging.FileHandler('../logs/pre_process.log')],
                    format='%(asctime)s:%(levelname)s:%(message)s')



class PreProcessor():
    # Initialize the Preprocessor with a path to the CSV data file.
    def __init__(self, filepath):
        self.data_path = filepath
        self.df = None
        logging.info("Preprocessor initialized with file: {filepath}")
        

    # Load Brent oil prices data from the CSV file.
    def load_data(self):
        try:
            self.df = pd.read_csv(self.data_path, parse_dates = ['Date'], dayfirst = True)
            self.df.columns = self.df.columns.str.strip()
            self.df.sort_values('Date', inplace = True)
            logging.info("Data loaded and sorted by date")
            return self.df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def detect_outliers(self):
        if 'Price' in self.df.columns:
            plt.figure(figsize=(12, 6))  # Increase figure size
            sns.boxplot(x=self.df['Price'])
            
            # Adjust x-axis for better readability
            plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            
            # Set x-axis ticks if necessary
            plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(nbins=10)) 
            
            plt.title("Boxplot of Price for Outlier Detection")
            plt.xlabel("Price") 
            plt.xticks(rotation=45)  
            plt.grid(True) 
            plt.show()
    
    # Handle missing values and perform initial data cleaning.
    def pre_process_data(self):
        display(self.df.describe())
        display(self.df.info())
        display(self.df.isnull().sum())

        self.detect_outliers()

    

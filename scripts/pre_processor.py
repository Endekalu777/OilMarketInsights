import pandas as pd
import logging
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from IPython.display import display
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
            # Load the main dataset (Brent oil data)
            self.df = pd.read_csv(self.brent_oil_path)
            self.df['Price'] = self.df['Price'].astype('float64')

            # Convert date column in the main data with the correct format
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%b-%y', errors='coerce')
                self.df['year'] = self.df['Date'].dt.year  # Create year column
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
            # Load the external indicators dataset
            self.brent_df = pd.read_csv(self.external_indicators_path)
            self.brent_df['date'] = pd.to_datetime(self.brent_df['date'])
            self.brent_df['year'] = self.brent_df['date'].dt.year  # Create year column

            # Ensure the external data contains the country column
            if 'country' not in self.brent_df.columns:
                self.brent_df['country'] = 'Unknown'  # or adjust according to your dataset

            logging.info("External indicators data loaded")
            return self.brent_df
            
        except Exception as e:
            logging.error(f"Error loading external data: {str(e)}")
            raise

    def merge_data(self):
        """ Merge Brent oil data with external indicators based on the year column """
        try:
            # Merge on the 'year' column
            merged_df = pd.merge(
                self.df,
                self.brent_df,
                on='year',
                how='inner'
            )
            
            # Preserve the 'country' column during merging
            self.merged_df = merged_df
            logging.info(f"Data merged successfully with {len(self.merged_df)} records")
            return self.merged_df
        except Exception as e:
            logging.error(f"Error merging data: {str(e)}")
            raise

    def detect_outliers(self):
        try:
            # Create a boxplot to detect outliers in the 'Price' column of the merged dataset
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=self.merged_df['Price'])
            
            # Customize the x-axis labels and format for better visibility
            plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
            
            # Add title and labels for better clarity
            plt.title("Boxplot of Price for Outlier Detection")
            plt.xlabel('Price')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error detecting outliers: {str(e)}")
            raise

    def visualize_data_distribution(self):
        """ Visualize distribution of numeric columns with histograms """
        try:
            numeric_cols = self.merged_df.select_dtypes(include=['float64', 'int64']).columns
            
            # Dynamically calculate the number of rows and columns needed for subplots
            num_cols = len(numeric_cols)
            num_rows = (num_cols // 3) + (1 if num_cols % 3 != 0 else 0)  # Ensure we have enough rows
            
            plt.figure(figsize=(15, 5 * num_rows))  # Adjust the figure size based on the number of rows

            # Loop through numeric columns and create subplots
            for i, col in enumerate(numeric_cols, 1):
                plt.subplot(num_rows, 3, i)
                sns.histplot(self.merged_df[col], kde=True)
                plt.title(f'Distribution of {col}')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(f"Error visualizing data distribution: {str(e)}")
            raise

    def missing_values_report(self):
        """ Provide a report on missing values percentage for each column """
        try:
            missing_values = self.merged_df.isnull().sum() / len(self.merged_df) * 100
            missing_values = missing_values[missing_values > 0].sort_values(ascending=False)
            print("\nMissing Values Report (Percentage):")
            display(missing_values)
        except Exception as e:
            logging.error(f"Error generating missing values report: {str(e)}")
            raise

    def correlation_heatmap(self):
        """ Plot a correlation heatmap for the numeric columns """
        try:
            plt.figure(figsize=(12, 8))
            corr = self.merged_df.select_dtypes(include=['float64', 'int64']).corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logging.error(f"Error generating correlation heatmap: {str(e)}")
            raise

    def pre_process_data(self):
        # Display the dataset being analyzed
        print(f"\nAnalyzing dataset: {self.brent_oil_path}")
        
        # Get a description of only numeric columns
        numeric_description = self.merged_df.select_dtypes(include=['float64', 'int64']).describe()
        
        # Display dataset description, info, missing values, and outlier detection results
        print("\nDataset Description:")
        display(numeric_description)
        
        print("\nDataset Info:")
        display(self.merged_df.info())
        
        print("\nMissing Values:")
        display(self.merged_df.isnull().sum())
        
        print("\nOutlier Detection:")
        self.detect_outliers()
        
        # Additional Checks
        self.missing_values_report()
        self.correlation_heatmap()
        self.visualize_data_distribution()


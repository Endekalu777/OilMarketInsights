import matplotlib.pyplot as plt
from matplotlib.dates import mdates
import pandas as pd
from IPyton.display import display

class EDAEventAnalysis():
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.df = pd.read_csv(filepath)
        # Convert 'Date' column to datetime format if not already
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        # Sort by Date to ensure correct order
        self.df = self.df.sort_values(by='Date')

    def perform_eda(self, moving_avg_window = 365):
        # Plot price trends over time
        plt.figure(figsize=(16, 8))
        
        # Plot the original price data
        plt.plot(self.df['Date'], self.df['Price'], label='Brent Oil Price', color='steelblue', linewidth=0.5)
        
        # Add a moving average for a smoother trend (1-year average for long-term trend)
        self.df['Price_MA'] = self.df['Price'].rolling(window=moving_avg_window).mean()
        plt.plot(self.df['Date'], self.df['Price_MA'], label=f'{moving_avg_window}-Day Moving Average', color='orange', linewidth=1.5)

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD per barrel)', fontsize=12)
        plt.title('Historical Brent Oil Prices', fontsize=14)
        plt.legend()

        # Set x-axis limits to match the date range in the data
        plt.xlim([self.df['Date'].min(), self.df['Date'].max()])

        # Set x-axis major ticks every 5 years, with minor ticks every year
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
        plt.gca().xaxis.set_minor_locator(mdates.YearLocator(1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.xticks(rotation=45)

        # Add grid for readability
        plt.grid(visible=True, linestyle='--', alpha=0.5)
        
        plt.show()

        # Display descriptive statistics
        display("Data Summary:")
        display(self.df['Price'].describe())


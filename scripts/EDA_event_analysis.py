import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from IPython.display import display

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

    def event_analysis(self):
        # Event dictionary with significant events
        events = {
            "1989-11-09": "Fall of Berlin Wall",
            "1990-08-02": "Gulf War begins",
            "1991-02-28": "Gulf War ends",
            "1991-12-24": "End of the Soviet Union",
            "1997-07-02": "Asian Financial Crisis",
            "2001-09-11": "9/11 Attack",
            "2003-03-20": "US Invasion of Iraq",
            "2006-07-12": "Israeli-Lebanese Conflict",
            "2007-12-01": "The Great Recession",
            "2010-12-17": "The Arab Uprising",
            "2011-02-15": "Libya Civil War",
            "2018-08-06": "US Sanctions on Iran",
            "2020-03-11": "COVID-19 Pandemic",
            "2022-02-24": "Russia Invasion of Ukraine",
        }
        # Initialize plot
        fig, ax = plt.subplots(figsize=(15, 7))

        # Plot the trend line with a thicker width for visibility
        ax.plot(self.df['Date'], self.df['Price'], label='Brent Oil Price', color='steelblue', linewidth=1.5)

        # Add event markers with vertical lines and labeled points
        for event_date_str, event_name in events.items():
            event_date = pd.to_datetime(event_date_str)
            
            # Find the closest date in the data to the event date
            closest_date_idx = (self.df['Date'] - event_date).abs().idxmin()
            closest_date = self.df.loc[closest_date_idx, 'Date']
            event_price = self.df.loc[closest_date_idx, 'Price']

            # Add a vertical line and marker at the closest date
            ax.axvline(closest_date, color='red', linestyle='--', linewidth=1)
            ax.plot(closest_date, event_price, 'ro')  

            # Add event label text
            ax.text(closest_date, event_price, f' {event_name}', color='black', fontsize=8, verticalalignment='bottom')

        
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import plotly.express as px
import os
import logging

# Create logs directory if it doesn't exist
if not os.path.exists('../logs'):
    os.makedirs('../logs')
logging.basicConfig(level=logging.INFO,
                    handlers = [logging.FileHandler('../logs/pre_process.log')],
                    format='%(asctime)s:%(levelname)s:%(message)s')

class EconomicOilAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.events = {
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
            "2020-03-11": "COVID-19 Pandemic"
        }
        self.load_data()

    def load_data(self):
        """Loads the dataset and parses date columns."""
        try:
            logging.info("Loading dataset...")
            self.data = pd.read_csv(self.file_path, parse_dates=['date_y'])
            self.data.rename(columns={'date_y': 'date'}, inplace=True)
            self.data['year'] = self.data['date'].dt.year 
            logging.info("Dataset loaded successfully.")
        except FileNotFoundError as e:
            logging.error("File not found. Please check the file path.")
            raise e
        except Exception as e:
            logging.error("An error occurred while loading the dataset.")
            raise e

    def inspect_data(self):
        """Displays basic information about the dataset."""
        logging.info("Inspecting dataset structure and summary.")
        print("Data Info:")
        display(self.data.info())
        print("\nStatistical Summary:")
        display(self.data.describe())
        print("\nFirst Few Rows:")
        display(self.data.head())

    def check_missing_values(self):
        """Checks and fills missing values."""
        display(self.data.isnull().sum())

    def plot_oil_price_trend(self):
        """Plots the trend of Brent oil prices over time, annotating key events."""
        logging.info("Plotting Brent oil price trend over time.")
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='date', y='Price', data=self.data, color='blue')
        plt.title('Brent Oil Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.grid(True)
        
        # Annotating key events on the plot
        for date, event in self.events.items():
            plt.axvline(pd.to_datetime(date), color='red', linestyle='--')
            plt.text(pd.to_datetime(date), self.data['Price'].min(), event, rotation=90, 
                     verticalalignment='bottom', horizontalalignment='right', fontsize=9, color='red')

        plt.show()

            
    def plot_correlation_matrix(self):
        """Plots a correlation matrix of oil prices and economic indicators."""
        logging.info("Plotting correlation matrix of economic indicators and Brent oil price.")
        corr_df = self.data[['Price', 'GDP Growth (%)', 'Inflation Rate (%)', 'Unemployment Rate (%)',
                             'Exchange Rate (Local Currency per USD)', 'Renewable Energy Consumption (%)',
                             'Environmental Tax Revenue (% of GDP)', 'Net Trade (BoP, current US$)',
                             'Natural Gas Electricity Production (%)']]
        correlation_matrix = corr_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Brent Oil Price and Economic Indicators')
        plt.show()

    def plot_rolling_correlation(self, window=12):
        """Plots rolling correlation between oil prices and key economic indicators."""
        logging.info("Calculating rolling correlations.")
        indicators = ['GDP Growth (%)', 'Inflation Rate (%)', 'Exchange Rate (Local Currency per USD)']
        
        plt.figure(figsize=(12, 8))
        for indicator in indicators:
            rolling_corr = self.data['Price'].rolling(window).corr(self.data[indicator])
            plt.plot(self.data['date'], rolling_corr, label=f'Price vs {indicator}')
        
        plt.title(f'Rolling Correlations with Brent Oil Price (Window: {window} months)')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_event_impact(self):
        """Plots oil price changes around significant events with improved handling."""
        logging.info("Analyzing event impacts.")
        event_window = 30  # Days around the event
        
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette('husl', len(self.events))  # Unique colors for each event
        for i, (date, event) in enumerate(self.events.items()):
            event_date = pd.to_datetime(date)
            # Filter data around the event
            window_data = self.data[
                (self.data['date'] >= event_date - pd.Timedelta(days=event_window)) &
                (self.data['date'] <= event_date + pd.Timedelta(days=event_window))
            ]
            
            if not window_data.empty:
                plt.plot(window_data['date'], window_data['Price'], label=event, color=colors[i])
            else:
                logging.warning(f"No data found around event: {event} on {date}")

        plt.title('Brent Oil Price Around Key Events')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot
        plt.grid(True)
        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.show()

    def add_derived_features(self):
        """Adds derived features like percentage change and volatility."""
        logging.info("Adding derived features.")
        self.data['Price Change (%)'] = self.data['Price'].pct_change() * 100
        self.data['Volatility'] = self.data['Price'].rolling(30).std()

    def interactive_price_trend(self):
        """Creates an interactive plot for oil price trends."""
        fig = px.line(self.data, x='date', y='Price', title='Brent Oil Price Over Time',
                    labels={'Price': 'Price (USD)', 'date': 'Date'})
        fig.update_traces(line=dict(color='blue'))
        fig.show()

    def plot_indicators(self, indicators, country):
        """Plots specified economic indicators for a specific country with a single Min-Max scaling."""
        
        # Filter data for the specified country
        country_df = self.data[self.data['country'] == country].copy()
        
        # Ensure date_x is in datetime format
        country_df['date_x'] = pd.to_datetime(country_df['date_x'])
        
        # Set the full date range from 1987 to 2022
        full_date_range = pd.date_range(start='1987-01-01', end='2022-12-31', freq='M')
        country_df = country_df.set_index('date_x').reindex(full_date_range).fillna(method='ffill').reset_index()
        country_df.rename(columns={'index': 'date_x'}, inplace=True)
        
        # Check that all indicators are present in the data before scaling
        missing_indicators = [ind for ind in indicators if ind not in country_df.columns]
        if missing_indicators:
            print(f"Warning: {', '.join(missing_indicators)} not found in data. They will be excluded from the plot.")
            indicators = [ind for ind in indicators if ind in country_df.columns]

        # Apply Min-Max scaling across all specified indicators together
        scaler = MinMaxScaler()
        country_df[indicators] = scaler.fit_transform(country_df[indicators])

        # Plot each indicator
        plt.figure(figsize=(14, 8))
        for indicator in indicators:
            sns.lineplot(x='date_x', y=indicator, data=country_df, label=indicator)

        # Set the x-axis limits to ensure the full date range is shown
        plt.xlim(pd.Timestamp('1987-01-01'), pd.Timestamp('2022-12-31'))

        # Format the x-axis dates for better readability
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Set locator to show every 2 years
        plt.xticks(rotation=45)

        # Add titles and labels
        plt.title(f'Economic Indicators Over Time (Scaled) - {country}')
        plt.xlabel('Date')
        plt.ylabel('Scaled Rate / Value (0 to 1)')
        plt.legend(title='Indicators')

        # Enhance grid appearance
        plt.grid(color='white', linestyle='--', linewidth=0.5)
        plt.gca().set_facecolor('#EAEAF2')  # Set a light gray background for the plot area

        plt.tight_layout()  # Adjust layout to prevent clipping
        plt.show()

    def plot_renewable_energy_and_tax_revenue(self):
        """Plots renewable energy consumption and environmental tax revenue by country."""

        # Ensure date_x is in datetime format
        self.data['date_x'] = pd.to_datetime(self.data['date_x'])

        logging.info("Plotting renewable energy consumption over time by country.")
        plt.figure(figsize=(14, 7))
        sns.lineplot(
            x='date_x',
            y='Renewable Energy Consumption (%)',
            hue='country',
            data=self.data,
            palette='tab10'
        )
        plt.title('Renewable Energy Consumption Over Time by Country')
        plt.xlabel('Date')
        plt.ylabel('Renewable Energy Consumption (%)')

        # Format the x-axis dates for better readability
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Set locator to show every 2 years
        plt.xlim(pd.Timestamp('1987-01-01'), pd.Timestamp('2022-12-31'))  # Ensure full range is shown
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        logging.info("Plotting environmental tax revenue over time by country.")
        plt.figure(figsize=(14, 7))
        sns.lineplot(
            x='date_x',
            y='Environmental Tax Revenue (% of GDP)',
            hue='country',
            data=self.data,
            palette='tab10'
        )
        plt.title('Environmental Tax Revenue Over Time by Country')
        plt.xlabel('Date')
        plt.ylabel('Environmental Tax Revenue (% of GDP)')

        # Format the x-axis dates for better readability
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Set locator to show every 2 years
        plt.xlim(pd.Timestamp('1987-01-01'), pd.Timestamp('2022-12-31'))  # Ensure full range is shown
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()



    def plot_trade_balance_and_gas_production(self):
        """Plots net trade balance and natural gas electricity production by country."""
        logging.info("Plotting net trade balance over time by country.")
        plt.figure(figsize=(14, 7))
        sns.lineplot(x='date_x', y='Net Trade (BoP, current US$)', hue='country', data=self.data)
        plt.title('Net Trade Balance Over Time by Country')
        plt.xlabel('Date')
        plt.ylabel('Net Trade (current US$)')
        
        # Format the x-axis dates for better readability
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Set locator to show every 2 years
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()

        logging.info("Plotting natural gas electricity production over time by country.")
        plt.figure(figsize=(14, 7))
        sns.lineplot(x='date_x', y='Natural Gas Electricity Production (%)', hue='country', data=self.data)
        plt.title('Natural Gas Electricity Production Over Time by Country')
        plt.xlabel('Date')
        plt.ylabel('Natural Gas Electricity Production (%)')
        
        # Format the x-axis dates for better readability
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(2))  # Set locator to show every 2 years
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.show()
        
import wbdata
import pandas as pd
import datetime
import os
import logging
from typing import Dict, Optional

# Create logs directory if it doesn't exist
if not os.path.exists('../logs'):
    os.makedirs('../logs')
logging.basicConfig(level=logging.INFO,
                    handlers = [logging.FileHandler('../logs/fetch_external_factors.log')],
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Define global variables
# Start and end date for filtering data
START_DATE = datetime.datetime(1987, 5, 20)
END_DATE = datetime.datetime(2022, 9, 30)
# List of country codes to fetch data for
COUNTRIES = ['USA', 'SAU', 'RUS', 'IRN', 'CHN', 'ARE', 'IRQ', 'KWT', 'EUU']
# Dictionary mapping indicator codes to descriptive names
INDICATORS = {
    'NY.GDP.MKTP.CD': 'GDP Growth (%)',
    'FP.CPI.TOTL': 'Inflation Rate (%)',
    'SL.UEM.TOTL.ZS': 'Unemployment Rate (%)',
    'PA.NUS.FCRF': 'Exchange Rate (Local Currency per USD)',
    'EG.FEC.RNEW.ZS': 'Renewable Energy Consumption (%)',
    'CC.ENTX.ENV.ZS': 'Environmental Tax Revenue (% of GDP)',
    'BN.GSR.GNFS.CD': 'Net Trade (BoP, current US$)',
    'EG.ELC.NGAS.ZS': 'Natural Gas Electricity Production (%)'
}

# Function to fetch data for a single indicator
def fetch_indicator_data(indicator_code: str, indicator_name: str) -> Optional[pd.DataFrame]:
    try:
        # Fetch data for the specified indicator and countries
        data = wbdata.get_dataframe({indicator_code: indicator_name}, country=COUNTRIES)
        # Check if data is returned and filter by date range
        if data is not None and not data.empty:
            data.reset_index(inplace=True)
            data['date'] = pd.to_datetime(data['date'])
            data = data[(data['date'] >= START_DATE) & (data['date'] <= END_DATE)]
            logging.info(f"Successfully fetched {indicator_name} data")
            return data
        else:
            logging.warning(f"No data returned for {indicator_name}")
            return None
    except Exception as e:
        # Log any errors encountered during data fetching
        logging.error(f"Error fetching {indicator_name}: {str(e)}")
        return None

# Function to save fetched data to a CSV file
def save_to_csv(data: pd.DataFrame, filename: str) -> None:
    try:
        # Save data to a CSV file with the provided filename
        data.to_csv(f"data/external_factors/{filename}.csv", index=False)
        logging.info(f"Data saved to data/external_factors/{filename}.csv")
    except Exception as e:
        # Log any errors encountered during saving
        logging.error(f"Error saving data to {filename}.csv: {str(e)}")

# Function to fetch and save data for all indicators
def fetch_all_indicators() -> Dict[str, pd.DataFrame]:
    indicator_data = {}
    # Loop through each indicator, fetch data, and save to CSV
    for code, name in INDICATORS.items():
        data = fetch_indicator_data(code, name)
        if data is not None:
            indicator_data[name] = data
            # Save each indicator's data to a CSV file, adjusting filename format
            save_to_csv(data, name.replace(" ", "_").replace("%", "percent"))
    return indicator_data

# Main execution to fetch and save all indicators data when the script is run
if __name__ == "__main__":
    fetch_all_indicators()

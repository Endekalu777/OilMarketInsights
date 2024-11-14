import pandas as pd
import wbdata
import datetime
import os
import logging
from typing import Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.FileHandler('logs/fetch_external_factors.log')],
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Define global variables
START_DATE = datetime.datetime(1987, 5, 20)
END_DATE = datetime.datetime(2022, 9, 30)
COUNTRIES = ['USA', 'SAU', 'RUS', 'IRN', 'CHN', 'ARE', 'IRQ', 'KWT', 'EUU']
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

def fetch_indicator_data(indicator_code: str, indicator_name: str) -> Optional[pd.DataFrame]:
    """Fetch data for a specific indicator from World Bank."""
    try:
        data = wbdata.get_dataframe({indicator_code: indicator_name}, country=COUNTRIES)
        if data is not None and not data.empty:
            data.reset_index(inplace=True)
            data['date'] = pd.to_datetime(data['date'])
            # Handling missing values here: forward fill followed by backward fill
            data.fillna(method='bfill', inplace=True)  # Backward fill
            logging.info(f"Successfully fetched {indicator_name} data with shape {data.shape}")
            return data[(data['date'] >= START_DATE) & (data['date'] <= END_DATE)]
        else:
            logging.warning(f"No data returned for {indicator_name}")
            return None
    except Exception as e:
        logging.error(f"Error fetching {indicator_name}: {str(e)}")
        return None

def merge_indicators(indicator_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all indicator dataframes, handling missing values professionally."""
    try:
        merged_df = None
        for name, df in indicator_data.items():
            if df is not None:
                if merged_df is None:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, on=['date', 'country'], how='outer')

        if merged_df is not None:
            merged_df = merged_df.sort_values(['country', 'date'])
            # Handling missing values in merged data: forward fill, then backward fill, and then interpolation
            merged_df.fillna(method='ffill', inplace=True)  # Forward fill
            merged_df.fillna(method='bfill', inplace=True)  # Backward fill
            merged_df.interpolate(method='linear', inplace=True)  # Linear interpolation for continuous data
            # Alternatively, you could fill with the mean/median or zero for certain columns if appropriate
            logging.info(f"Successfully merged all indicators with final shape {merged_df.shape}")
            return merged_df
        else:
            logging.warning("No data to merge")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error merging indicators: {str(e)}")
        return pd.DataFrame()

def save_to_csv(data: pd.DataFrame, filename: str):
    """Save a DataFrame to a CSV file."""
    path = os.path.join('data/external_factors', f"{filename}.csv")
    data.to_csv(path, index=False)
    logging.info(f"Data saved to {path}")

if __name__ == "__main__":
    os.makedirs('data/external_factors', exist_ok=True)
    indicator_data = {}
    for code, name in INDICATORS.items():
        indicator_data[name] = fetch_indicator_data(code, name)

    merged_data = merge_indicators(indicator_data)

    if not merged_data.empty:
        save_to_csv(merged_data, 'merged_indicators')
    else:
        logging.warning("No data available after merging indicators; no files were saved.")

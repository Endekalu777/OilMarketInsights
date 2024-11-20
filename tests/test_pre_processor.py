import pytest
import pandas as pd
import os
from scripts.pre_processor import * 

@pytest.fixture
def setup_preprocessor(tmpdir):
    # Create mock Brent oil data
    brent_data = pd.DataFrame({
        'Date': ['01-Jan-20', '02-Jan-20', '03-Jan-20'],
        'Price': [50.0, 51.0, 52.0]
    })
    brent_file = tmpdir.join("brent_oil.csv")
    brent_data.to_csv(brent_file, index=False)

    # Create mock external indicators data
    external_data = pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01'],
        'indicator': [100, 110, 120],
        'country': ['US', 'US', 'US']
    })
    external_file = tmpdir.join("external_indicators.csv")
    external_data.to_csv(external_file, index=False)

    return PreProcessor(brent_oil_path=str(brent_file), external_indicators_path=str(external_file))

def test_load_brent_data(setup_preprocessor):
    preprocessor = setup_preprocessor
    df = preprocessor.load_brent_data()

    assert isinstance(df, pd.DataFrame)
    assert 'date' in df.columns
    assert 'Price' in df.columns
    assert df['Price'].dtype == 'float64'
    assert len(df) == 3  
    assert all(df['year'] == 2020)  

def test_load_external_data(setup_preprocessor):
    preprocessor = setup_preprocessor
    brent_df = preprocessor.load_external_data()

    assert isinstance(brent_df, pd.DataFrame)
    assert 'date' in brent_df.columns
    assert 'indicator' in brent_df.columns
    assert 'country' in brent_df.columns
    assert brent_df['country'].iloc[0] == 'US'
    assert len(brent_df) == 3  
    assert all(brent_df['year'] == 2020)  

def test_merge_data(setup_preprocessor):
    preprocessor = setup_preprocessor
    preprocessor.load_brent_data()
    preprocessor.load_external_data()
    merged_df = preprocessor.merge_data()

    assert isinstance(merged_df, pd.DataFrame)
    assert len(merged_df) > 0  
    assert 'date_x' in merged_df.columns  
    assert 'date_y' in merged_df.columns  
    assert 'Price' in merged_df.columns
    assert 'indicator' in merged_df.columns

"""
Data Preprocessing Module

This module handles loading and cleaning the raw NIFTY50 tick data.
Ensures data is properly formatted and sorted chronologically.
"""

import pandas as pd


def load_and_clean_data(filepath):
    """
    Load and preprocess the NIFTY50 tick data.
    
    Steps:
    1. Load CSV file
    2. Convert timestamp to datetime
    3. Sort chronologically (important for time-series)
    
    Args:
        filepath: Path to the CSV file
    
    Returns:
        Cleaned DataFrame sorted by timestamp
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)

    # Convert timestamp string to datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort chronologically - critical for time-series data
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"Data loaded: {len(df):,} rows")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df
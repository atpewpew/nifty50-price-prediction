"""
Technical Indicators for Financial Analysis

This module implements common technical analysis indicators used in
trading and financial modeling. All indicators are calculated using
standard formulas used in the industry.
"""

import pandas as pd
import numpy as np


def calculate_rsi(series, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    RSI measures momentum by comparing recent gains to recent losses.
    Values range from 0 to 100:
    - RSI > 70: Potentially overbought (may fall)
    - RSI < 30: Potentially oversold (may rise)
    
    Args:
        series: Price series (typically close prices)
        period: Lookback period (default 14)
    
    Returns:
        Series of RSI values
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    MACD shows the relationship between two moving averages.
    Bullish signal when MACD crosses above signal line.
    
    Args:
        series: Price series (typically close prices)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
    
    Returns:
        tuple: (macd_line, signal_line)
    """
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR).
    
    ATR measures market volatility by decomposing the entire
    range of an asset price for the period.
    
    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: Lookback period (default 14)
    
    Returns:
        Series of ATR values
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(window=period).mean()


def calculate_bollinger_bands(series, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of a middle band (SMA) with an upper
    and lower band at standard deviation intervals. Used to identify
    overbought/oversold conditions and volatility.
    
    Args:
        series: Price series (typically close prices)
        window: Moving average window (default 20)
        num_std: Number of standard deviations (default 2)
    
    Returns:
        tuple: (upper_band, lower_band)
    """
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band
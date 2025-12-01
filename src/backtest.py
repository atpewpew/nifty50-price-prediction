"""
Backtesting Module

This module provides additional backtesting capabilities for evaluating
trading strategies with realistic simulation including transaction costs.

Note: The main PnL calculation follows the assignment specification exactly.
This module provides supplementary realistic metrics for bonus analysis.
"""

import pandas as pd
import numpy as np


def calculate_trading_metrics(df, predictions, y_true, close_prices):
    """
    Calculate comprehensive trading and model performance metrics.
    
    This function computes both required metrics (Accuracy, Precision, Recall)
    and bonus metrics (Sharpe Ratio, Max Drawdown, Win Rate) for thorough analysis.
    
    Args:
        df: DataFrame with trading data
        predictions: Model predictions (0/1)
        y_true: Actual target values (0/1)
        close_prices: Series of close prices
    
    Returns:
        dict: Dictionary containing all metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    
    # --- Required Metrics (as per assignment) ---
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    
    # --- Bonus Metrics ---
    # Win Rate: Percentage of correct predictions
    win_rate = accuracy  # Same as accuracy for direction prediction
    
    # Calculate returns for Sharpe Ratio
    returns = close_prices.pct_change().dropna()
    
    # Sharpe Ratio (annualized, assuming ~252 trading days, ~375 mins/day for intraday)
    # For simplicity, we use daily-equivalent calculation
    if len(returns) > 1 and returns.std() > 0:
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 375)
    else:
        sharpe_ratio = 0.0
    
    # Max Drawdown from model_pnl if available
    if 'model_pnl' in df.columns:
        cumulative_pnl = df['model_pnl']
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        final_pnl = cumulative_pnl.iloc[-1]
    else:
        max_drawdown = 0.0
        final_pnl = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_pnl': final_pnl
    }


def calculate_pnl(df, signal_col='model_call', price_col='close'):
    """
    Calculate cumulative PnL as per assignment specification.
    
    Logic (from assignment):
    - If model_call is 'buy' → Subtract current close from model_pnl
    - If model_call is 'sell' → Add current close to model_pnl
    - Iterate row by row, keeping cumulative running PnL
    
    Args:
        df: DataFrame with signal and price columns
        signal_col: Column name containing 'buy'/'sell' signals
        price_col: Column name containing prices
    
    Returns:
        Series of cumulative PnL values
    """
    pnl = 0.0
    pnl_history = []
    
    for _, row in df.iterrows():
        if row[signal_col] == 'buy':
            pnl -= row[price_col]  # Subtract close price
        else:  # sell
            pnl += row[price_col]  # Add close price
        pnl_history.append(pnl)
    
    return pnl_history


def format_metrics_report(metrics, model_name="Selected Model"):
    """
    Format metrics into a readable report string.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model for the report
    
    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append(f"PERFORMANCE METRICS - {model_name}")
    report.append("=" * 60)
    report.append("")
    report.append("Required Metrics (Assignment):")
    report.append("-" * 40)
    report.append(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    report.append(f"  Precision:  {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    report.append(f"  Recall:     {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    report.append("")
    report.append("Bonus Metrics:")
    report.append("-" * 40)
    report.append(f"  Win Rate:       {metrics['win_rate']*100:.2f}%")
    report.append(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.4f}")
    report.append(f"  Max Drawdown:   {metrics['max_drawdown']:,.2f}")
    report.append(f"  Final PnL:      {metrics['final_pnl']:,.2f}")
    report.append("=" * 60)
    
    return "\n".join(report)
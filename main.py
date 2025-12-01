"""
Main execution script for NIFTY50 price direction prediction.

This script orchestrates the entire ML pipeline:
1. Load and preprocess data
2. Engineer features
3. Train and compare models
4. Generate predictions and calculate PnL
5. Save results

Run: python main.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# Local imports
from src.preprocessing import load_and_clean_data
from src.features import add_features, create_target
from src.models import train_and_compare_models
from src.backtest import calculate_pnl, calculate_trading_metrics, format_metrics_report


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    if not os.path.exists('results'):
        os.makedirs('results')
        print("Created 'results/' directory")


def save_metrics_report(all_results, best_results, trading_metrics):
    """
    Save model comparison and metrics to a text file.
    
    Args:
        all_results: List of results for all models
        best_results: Results dict for the selected best model
        trading_metrics: Dictionary of trading performance metrics
    """
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("NIFTY50 PRICE DIRECTION PREDICTION - MODEL RESULTS")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Model Comparison Table
    report_lines.append("MODEL COMPARISON")
    report_lines.append("-" * 70)
    report_lines.append(f"{'Model':<30} {'Accuracy':<15} {'Precision':<15} {'Recall':<15}")
    report_lines.append("-" * 70)
    
    for result in all_results:
        report_lines.append(
            f"{result['name']:<30} "
            f"{result['accuracy']:<15.4f} "
            f"{result['precision']:<15.4f} "
            f"{result['recall']:<15.4f}"
        )
    report_lines.append("-" * 70)
    report_lines.append("")
    
    # Best Model Selection
    report_lines.append(f"SELECTED MODEL: {best_results['name']}")
    report_lines.append("-" * 70)
    report_lines.append("")
    
    # Required Metrics
    report_lines.append("REQUIRED METRICS (Test Set):")
    report_lines.append(f"  - Accuracy:   {trading_metrics['accuracy']:.4f} ({trading_metrics['accuracy']*100:.2f}%)")
    report_lines.append(f"  - Precision:  {trading_metrics['precision']:.4f} ({trading_metrics['precision']*100:.2f}%)")
    report_lines.append(f"  - Recall:     {trading_metrics['recall']:.4f} ({trading_metrics['recall']*100:.2f}%)")
    report_lines.append("")
    
    # Bonus Metrics
    report_lines.append("ADDITIONAL METRICS:")
    report_lines.append(f"  - Win Rate:       {trading_metrics['win_rate']*100:.2f}%")
    report_lines.append(f"  - Sharpe Ratio:   {trading_metrics['sharpe_ratio']:.4f}")
    report_lines.append(f"  - Max Drawdown:   {trading_metrics['max_drawdown']:,.2f}")
    report_lines.append(f"  - Final PnL:      {trading_metrics['final_pnl']:,.2f}")
    report_lines.append("")
    
    # Model Explanation (2-4 lines as required) - Dynamic based on winner
    report_lines.append("=" * 70)
    report_lines.append("MODEL SELECTION EXPLANATION")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    if best_results['name'] == 'Random Forest':
        report_lines.append("Random Forest was selected as the best model due to its superior accuracy")
        report_lines.append("on the test set. It uses an ensemble of decision trees with bootstrap")
        report_lines.append("aggregating (bagging), which reduces overfitting and provides robust")
        report_lines.append("predictions for noisy financial time-series data.")
    else:
        report_lines.append("Gradient Boosting was selected as the best model due to its superior")
        report_lines.append("accuracy on the test set. It uses sequential boosting where each tree")
        report_lines.append("corrects errors of previous trees, making it effective for capturing")
        report_lines.append("complex patterns in financial time-series data.")
    report_lines.append("")
    report_lines.append("=" * 70)
    
    # Write to file
    report_text = "\n".join(report_lines)
    with open('results/metrics_summary.txt', 'w') as f:
        f.write(report_text)
    
    print(f"\n[SAVED] Metrics saved to 'results/metrics_summary.txt'")
    return report_text


def main():
    """
    Main execution function.
    
    Steps:
    1. Load and preprocess data
    2. Engineer features
    3. Create train/test split (time-based)
    4. Train and compare models (Random Forest vs Gradient Boosting)
    5. Generate predictions and trading signals
    6. Calculate PnL as per assignment specification
    7. Save results
    """
    print("\n" + "=" * 70)
    print("NIFTY50 PRICE DIRECTION PREDICTION")
    print("=" * 70)
    
    # Ensure results directory exists
    ensure_results_dir()
    
    # =========================================================================
    # STEP 1: Load & Preprocess Data
    # =========================================================================
    print("\n--- Step 1: Loading Data ---")
    df = load_and_clean_data('data/nifty50_ticks.csv')
    
    # =========================================================================
    # STEP 2: Feature Engineering
    # =========================================================================
    print("\n--- Step 2: Engineering Features ---")
    df = add_features(df)
    df = create_target(df)
    print(f"Features generated: {df.shape[1]} columns, {len(df):,} rows")
    
    # =========================================================================
    # STEP 3: Train/Test Split (Time-Based - 80/20)
    # =========================================================================
    print("\n--- Step 3: Train/Test Split ---")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Training set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set:     {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Define feature columns (exclude non-features and target)
    exclude_cols = ['timestamp', 'id', 'symbol', 'exchange', 'target', 
                    'close', 'open', 'high', 'low', 'volume', 'open_interest', 'next_close']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_test = test_df[feature_cols]
    y_test = test_df['target']
    
    print(f"Number of features: {len(feature_cols)}")
    
    # =========================================================================
    # STEP 4: Train & Compare Models
    # =========================================================================
    best_model, best_results, all_results = train_and_compare_models(
        X_train, y_train, X_test, y_test
    )
    
    # =========================================================================
    # STEP 5: Generate Predictions & Trading Signals
    # =========================================================================
    print("\n--- Step 5: Generating Trading Signals ---")
    
    # Get predictions from best model
    predictions = best_model.predict(X_test)
    
    # Add required columns to test dataframe
    test_df['Predicted'] = predictions                                    # 0/1 as required
    test_df['model_call'] = test_df['Predicted'].apply(                  # buy/sell signal
        lambda x: 'buy' if x == 1 else 'sell'
    )
    
    # =========================================================================
    # STEP 6: Calculate PnL (Assignment Specification)
    # =========================================================================
    print("\n--- Step 6: Calculating PnL ---")
    
    # Calculate cumulative PnL as per assignment:
    # buy → subtract close, sell → add close
    test_df['model_pnl'] = calculate_pnl(test_df, 'model_call', 'close')
    
    final_pnl = test_df['model_pnl'].iloc[-1]
    print(f"Final Cumulative PnL: {final_pnl:,.2f}")
    
    # =========================================================================
    # STEP 7: Calculate & Display Metrics
    # =========================================================================
    print("\n--- Step 7: Performance Metrics ---")
    
    trading_metrics = calculate_trading_metrics(
        test_df, predictions, y_test, test_df['close']
    )
    
    # Display formatted metrics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"\nBest Model: {best_results['name']}")
    print("\nRequired Metrics (Test Set):")
    print(f"  - Accuracy:   {trading_metrics['accuracy']:.4f} ({trading_metrics['accuracy']*100:.2f}%)")
    print(f"  - Precision:  {trading_metrics['precision']:.4f} ({trading_metrics['precision']*100:.2f}%)")
    print(f"  - Recall:     {trading_metrics['recall']:.4f} ({trading_metrics['recall']*100:.2f}%)")
    print("\nAdditional Metrics:")
    print(f"  - Win Rate:       {trading_metrics['win_rate']*100:.2f}%")
    print(f"  - Max Drawdown:   {trading_metrics['max_drawdown']:,.2f}")
    print(f"  - Final PnL:      {trading_metrics['final_pnl']:,.2f}")
    
    # =========================================================================
    # STEP 8: Save Results
    # =========================================================================
    print("\n--- Step 8: Saving Results ---")
    
    # Save metrics report
    save_metrics_report(all_results, best_results, trading_metrics)
    
    # Save submission output (required format) - keep at root for easy access
    output_cols = ['timestamp', 'close', 'Predicted', 'model_call', 'model_pnl']
    test_df[output_cols].to_csv('submission_output.csv', index=False)
    print("[SAVED] Submission saved to 'submission_output.csv'")
    
    # Save full report with all columns
    test_df.to_csv('results/full_backtest_report.csv', index=False)
    print("[SAVED] Full report saved to 'results/full_backtest_report.csv'")
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print("  - submission_output.csv                 - Required submission format")
    print("  - results/metrics_summary.txt           - Model comparison & metrics")
    print("  - results/full_backtest_report.csv      - Complete analysis data")
    print("")


if __name__ == "__main__":
    main()
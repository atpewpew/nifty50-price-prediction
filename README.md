# NIFTY50 Price Direction Prediction

A Machine Learning project to predict whether the next candle's closing price will go **up** or **down** using historical NIFTY50 intraday data.

## 📊 Quick Results

**Best Model: Random Forest (Accuracy: 51.03%)**

Random Forest outperformed Gradient Boosting by 0.19% accuracy on the test set. It uses an ensemble of decision trees with bootstrap aggregating, which provides robust predictions and reduces overfitting - particularly effective for noisy financial time-series data where precision matters as much as raw accuracy.

| Metric | Value |
|--------|-------|
| Accuracy | 51.03% |
| Precision | 50.75% |
| Recall | 50.60% |
| Final PnL | ₹12,132,377 |

---

## Project Structure

```
tradio-gemini/
├── .gitignore             # Git ignore file
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── submission_output.csv # Final submission file
├── data/
│   └── nifty50_ticks.csv # Input dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py  # Data loading & cleaning
│   ├── features.py       # Feature engineering
│   ├── indicators.py     # Technical indicators (RSI, MACD, etc.)
│   ├── models.py         # ML models (Random Forest, Gradient Boosting)
│   └── backtest.py       # PnL calculation & metrics
└── results/              # Generated after running
    ├── metrics_summary.txt         # Model comparison report
    └── full_backtest_report.csv    # Complete analysis data
```


## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/atpewpew/nifty50-price-prediction.git
   cd nifty50-price-prediction
   ```

2. **Create virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Execution

Run the main script:

```bash
python main.py
```

This will:
1. Load and preprocess the NIFTY50 data
2. Engineer technical features (RSI, MACD, Bollinger Bands, etc.)
3. Train and compare two ML models
4. Generate predictions and trading signals
5. Calculate cumulative PnL
6. Save results

## 📁 Quick Results Access

After running `python main.py`, check:

| File | What's Inside |
|------|---------------|
| **`submission_output.csv`** | Final submission with Timestamp, Close, Predicted, model_call, model_pnl |
| **`results/metrics_summary.txt`** | Model comparison and all performance metrics |
| **`results/full_backtest_report.csv`** | Complete dataset with all features and predictions |

## Models Compared

1. **Random Forest Classifier** - Ensemble of decision trees using bagging
2. **Histogram Gradient Boosting Classifier** - Sequential boosting (similar to XGBoost/LightGBM)

### Model Selection

The best model is automatically selected based on highest accuracy on the test set. Both models are trained and compared, with detailed metrics saved to `results/metrics_summary.txt`.

**Why these models?**
- **Random Forest**: Robust to overfitting due to bagging, handles noisy data well
- **Gradient Boosting**: Captures complex patterns through sequential error correction

## Evaluation Metrics

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of predicted "up" signals, how many were actually up
- **Recall**: Of actual "up" movements, how many did we correctly predict

## Assumptions

- No transaction costs or slippage considered in PnL calculation
- Instant execution at close prices assumed
- Binary buy/sell signals only (no position sizing)
- Time-based train/test split (80/20) to prevent data leakage

## Author

**Tradio ML Internship Assessment**  
December 2024  
Repository: https://github.com/atpewpew/nifty50-price-prediction

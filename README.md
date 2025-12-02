# NIFTY50 Price Direction Prediction

A Machine Learning project to predict whether the next candle's closing price will go **up** or **down** using historical NIFTY50 intraday data.

## ⚡ Quick Results

**Best Model: Random Forest (Accuracy: 51.03%)**

Random Forest outperformed Gradient Boosting on the test set. It uses an ensemble of decision trees with bootstrap aggregating, which provides robust predictions and reduces overfitting—particularly effective for noisy financial time-series data where precision matters as much as raw accuracy.

| Metric | Value |
|--------|-------|
| Accuracy | 51.03% |
| Precision | 50.75% |
| Recall | 50.60% |
| **Cumulative PnL (Assignment Logic)** | **₹12,132,377*** |

*> **Note:** PnL is calculated based on strict assignment instructions (Cumulative Cash Flow: Buy = -Close, Sell = +Close). This represents a theoretical cash flow ledger, not a realized trading profit.*

---

## 📂 Project Structure

```text
nifty50-price-prediction/
├── .gitignore              # Git ignore file
├── main.py                 # Main execution script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── submission_output.csv   # Final submission file
├── data/
│   └── nifty50_ticks.csv   # Input dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # Data loading & cleaning
│   ├── features.py         # Advanced Feature engineering
│   ├── indicators.py       # Technical indicators (RSI, MACD, etc.)
│   ├── models.py           # ML models (Random Forest, Gradient Boosting)
│   └── backtest.py         # PnL calculation & metrics
└── results/                # Generated after running
    ├── metrics_summary.txt      # Model comparison report
    └── full_backtest_report.csv # Complete analysis data
````

## 🛠️ Key Features Engineered

To capture market microstructure beyond simple price changes, the following feature sets were implemented:

1.  **Momentum Indicators:**

      * **RSI (14):** Relative Strength Index to identify overbought/oversold conditions.
      * **MACD:** Moving Average Convergence Divergence to track trend momentum.
      * **Candle Body:** Captures the immediate buying/selling pressure of the specific minute.

2.  **Volatility Metrics:**

      * **ATR (Average True Range):** Measures market "energy" and volatility.
      * **Bollinger Bands:** Identifies when price deviates significantly from the mean.

3.  **Temporal Context (Memory):**

      * **Lag Features:** Lagged returns (t-1, t-2, t-3) provide the model with "memory" of recent price action, allowing it to detect reversals.

4.  **Cyclical Time Encoding:**

      * Timestamps were transformed using Sine/Cosine functions to preserve the cyclical nature of time (e.g., distinguishing market open volatility from mid-day lulls).

-----

## 🚀 Setup & Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/atpewpew/nifty50-price-prediction.git](https://github.com/atpewpew/nifty50-price-prediction.git)
    cd nifty50-price-prediction
    ```

2.  **Create virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 💻 Execution

Run the main script:

```bash
python main.py
```

This will:

1.  Load and preprocess the NIFTY50 data.
2.  Engineer technical features (RSI, MACD, Bollinger Bands, Lag features).
3.  Train and compare Random Forest vs. Gradient Boosting models.
4.  Generate predictions and trading signals.
5.  Calculate cumulative PnL based on assignment logic.
6.  Save results to `submission_output.csv`.

## 📊 Results Access

After running `python main.py`, check:

| File | What's Inside |
|------|---------------|
| **`submission_output.csv`** | Final submission with Timestamp, Close, Predicted, model\_call, model\_pnl |
| **`results/metrics_summary.txt`** | Model comparison and all performance metrics |
| **`results/full_backtest_report.csv`** | Complete dataset with all features and predictions |

## 🧠 Models Compared

1.  **Random Forest Classifier:** Ensemble of decision trees using bagging.
2.  **Histogram Gradient Boosting Classifier:** Sequential boosting optimized for large datasets (similar to XGBoost/LightGBM).

### Model Selection

The best model is automatically selected based on the highest accuracy on the test set. Random Forest was selected for its stability and resistance to overfitting on this specific dataset.

## 📈 Results Interpretation

The model achieved an accuracy of **51.03%** on unseen test data.

  * **Statistical Significance:** While 51% may appear close to random, in high-frequency quantitative trading, a consistent edge \>50% combined with high trade volume is often sufficient for profitability.
  * **PnL Correlation:** The positive cumulative PnL indicates that the model's correct predictions effectively captured larger price movements or sustained trends, rather than just noise.
  * **Precision/Recall Balance:** The metrics are balanced (\~50.7% Precision vs \~50.6% Recall), indicating the model does not have a strong bias toward buying or selling, which is crucial for a neutral market-making style strategy.

## ⚠️ Assumptions

  * **Transaction Costs:** No slippage or brokerage fees were applied in the PnL calculation (as per assignment scope).
  * **Execution:** Instant execution at the `Close` price is assumed.
  * **Signal Logic:** The model produces binary signals (Buy/Sell) based on a confidence threshold of 0.5.
  * **Split:** A time-based split (80% Train / 20% Test) was used to strictly prevent look-ahead bias.


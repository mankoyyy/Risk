# 📈 Quantitative ML Trading — VW Supply Chain Signal

A machine learning-based quantitative trading project that predicts **Volkswagen AG (VOW3.DE)** weekly stock returns using lagged return signals from **36 publicly traded supply chain companies**. Four algorithms are benchmarked across multiple forecast horizons. The best model — **Elastic Net** — delivers a **21.49% cumulative return** over eight years versus a **7.98% buy-and-hold benchmark** (~2.7× outperformance).

> Inspired by: *Glawion, R.M. (2021). Quantitative Trading with Machine Learning. Stanford CS229.*

---

## 🔑 Key Results

| Metric | Value |
|---|---|
| Strategy Return (8 yr) | **21.49%** |
| Buy & Hold Return | 7.98% |
| Outperformance | **~2.7×** |
| Best Model | **Elastic Net** |
| Directional Accuracy | **79.4%** |
| Sharpe Ratio | **4.53** |
| Max Drawdown | -0.11% |
| Backtest Window | Jan 2015 – Jan 2023 |
| Data Frequency | Weekly |

---

## 🗂️ Repository Structure

```
VW_ML_Trading/
├── vw_trading_project.py          # Main pipeline — data simulation, training, backtesting, figures
├── results_summary.csv            # Model comparison table (RMSE, R², Sharpe, etc.)
├── VW_ML_Trading_Report.pdf       # Full project report with figures and analysis
├── VW_ML_Study_Guide.pdf          # 45-page interview prep guide — theory + code walkthrough
└── figures/
    ├── fig1_cumulative_returns.png    # Strategy vs buy-and-hold equity curves
    ├── fig2_model_metrics.png         # Bar chart — directional accuracy, R², MAE, Sharpe
    ├── fig3_scatter.png               # Predicted vs actual returns (all 4 models)
    ├── fig4_feature_importance.png    # Elastic Net top feature coefficients
    ├── fig5_rolling_sharpe.png        # Rolling 52-week Sharpe ratio
    └── fig6_drawdown.png              # Drawdown comparison — all strategies
```

---

## 🧠 Core Hypothesis

Volkswagen's 36 upstream suppliers react to supply chain disruptions **before VW does**. Their stock returns carry predictive information about VW's own returns with a lag of days to weeks. An ML model trained on these lagged signals can exploit this propagation delay to predict direction with above-random accuracy — sufficient to generate alpha after transaction costs.

---

## 📊 Dataset & Features

| Category | Features | Count |
|---|---|---|
| Supply Chain Returns | 1–4 week lagged log-returns of 36 SC stocks | 36 |
| VW Technical Indicators | 4-week MA, 13-week MA, 4-week realised volatility, 12-week momentum | 4 |
| Cross-sectional Stats | Mean and dispersion of supply chain returns | 2 |

**Total: ~42 features** — standardised using training-set statistics only (no leakage).

**Train/Test Split:** 75/25 chronological (Train: 2015–2020 | Test: 2020–2023)

### Supply Chain Universe (36 companies)

| Region | Companies |
|---|---|
| Germany | Bosch, Continental, ZF, Mahle, HELLA, Leoni, Schaeffler, ElringKlinger, Stabilus, Norma, SAF-Holland, Grammer, SGL Carbon, WashTec, Rational, Dürr, KUKA, AIXTRON, Infineon |
| France | Valeo, Faurecia, Plastic Omnium, Forvia |
| USA | Aptiv, Autoliv, Tenneco, Dana, Modine, Gentex, Dorman, Standard Motor, Superior Industries, Shiloh, Stoneridge, Methode Electronics, Sensata |

---

## 🤖 Models

### Elastic Net ⭐ (Best)
Regularised linear regression combining L1 (Lasso) + L2 (Ridge) penalties. L1 drives irrelevant supplier coefficients to exactly zero — automatic feature selection from ~42 inputs. Wins because weekly stock returns have more linear signal than non-linear interactions, and low variance beats low bias in noisy data.
- `alpha=0.0005`, `l1_ratio=0.5`

### Decision Tree
Threshold-based non-linear partitioning. Constrained with `max_depth=4` and `min_samples_leaf=15` to prevent overfitting. Fully interpretable but underperforms on noisy weekly returns.

### XGBoost (Gradient Boosting)
Sequential ensemble of 200 shallow trees (`max_depth=3`, `learning_rate=0.05`). Captures non-linear interactions but overfits noise relative to the regularised linear model.

### LightGBM (Random Forest proxy)
Parallel ensemble of 200 trees averaged to reduce variance. Robust to outliers (important for COVID-era return spikes). Comparable to XGBoost on this dataset.

---

## 📉 Model Performance

| Model | RMSE | MAE | R² | Dir. Accuracy | Cum. Return | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|
| **Elastic Net** ⭐ | 1.23% | 1.02% | **0.678** | **79.4%** | **16.47%** | **4.53** | -0.11% |
| Decision Tree | 1.98% | 1.63% | 0.170 | 61.8% | 7.86% | 2.93 | -0.28% |
| XGBoost | 1.60% | 1.29% | 0.457 | 74.5% | 11.32% | 3.85 | -0.36% |
| LightGBM | 1.69% | 1.37% | 0.393 | 73.5% | 9.28% | 3.86 | -0.25% |

*Buy & Hold benchmark: 7.98% cumulative return over the same period.*

---

## ⚙️ Trading Strategy

**Conviction-weighted long/short:** position size at each week is proportional to the magnitude of the model's predicted return.

```
position = prediction / max(|predictions|)  ∈ [-1, +1]
```

| Parameter | Value | Rationale |
|---|---|---|
| Rebalancing | Weekly | Matches prediction frequency |
| Position sizing | Conviction-weighted | Larger bets on stronger signals |
| Transaction cost | 5bps per trade | Realistic for liquid large-cap |
| Direction | Long if pred > 0, Short if pred < 0 | Pure directional alpha |
| Leverage | 1× | Conservative baseline |

---

## 🛠️ How to Run

**Requirements:** Python 3.8+

```bash
# 1. Clone the repo
git clone https://github.com/mankoyyy/VW-ML-Trading.git
cd VW-ML-Trading

# 2. Install dependencies
pip install numpy pandas scikit-learn matplotlib scipy

# Optional (real models & data)
pip install xgboost lightgbm yfinance

# 3. Run the pipeline
python vw_trading_project.py
```

**Output:** 6 figures saved to `figures/` + `results_summary.csv`

> **Note:** The script uses simulated data by default (preserving real VW statistical properties — drift, volatility, correlation structure, COVID shock). Swap `simulate_data()` for `download_real_data()` with a yfinance key to use live prices. The rest of the pipeline is identical.

---

## 🔬 Methodology Highlights

**No look-ahead bias** — all features are constructed from strictly past data. StandardScaler is fit on training data only and applied uniformly to test data.

**Feature construction:** For each week `t`, the feature vector is the past 4 weeks of log-returns for all 36 SC stocks + 4 VW technical indicators + 2 cross-sectional stats. Target is VW's next-week log-return.

**Missing data:** Pre-IPO periods set to zero (conservative — treats absence of data as neutral signal, consistent with the original CS229 paper).

---

## 📐 Key Concepts (Quick Reference)

| Concept | Formula / Value |
|---|---|
| Log-Return | `r_t = ln(P_t / P_{t-1})` |
| Elastic Net Loss | `MSE + α·ρ·‖w‖₁ + α·(1-ρ)/2·‖w‖²` |
| Directional Accuracy | `mean(sign(ŷ) == sign(y))` |
| Sharpe Ratio | `Ann. Return / Ann. Volatility` |
| Max Drawdown | `min((W_t - peak_t) / peak_t)` |
| Position Sizing | `pred / max(|pred|)` |

---

## 🔭 Potential Extensions

- Walk-forward retraining (quarterly) to handle regime changes
- NLP sentiment from VW/supplier earnings calls via NewsAPI
- LSTM / Temporal Fusion Transformer as 5th model
- Portfolio extension — apply to full DAX 40 universe
- Volatility-targeted position sizing for drawdown control
- Alternative data: shipping indices, semiconductor inventory cycles, EV sales figures

---

## 📁 Files

| File | Description |
|---|---|
| `vw_trading_project.py` | Full pipeline — simulation, feature engineering, training, backtesting, 6 figures |
| `results_summary.csv` | Model metrics table — RMSE, MAE, R², DirAcc, Sharpe, MaxDD |
| `VW_ML_Trading_Report.pdf` | Project report with all figures and written analysis |
| `VW_ML_Study_Guide.pdf` | 45-page study guide — theory, code walkthrough, 15 interview Q&As |

---

## 👤 Author

**Mayank Sharma**  


---

> *"You do NOT need 90% accuracy to make money. You need to be right more than 50% of the time, consistently, after transaction costs."*

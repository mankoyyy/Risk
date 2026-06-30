import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT_DIR = '/home/claude/vw_ml_trading/figures'

np.random.seed(42)

# Tier-1/Tier-2 VW suppliers. Private ones (ZF, Mahle) don't have tickers
SUPPLY_CHAIN_TICKERS = [
    "BOSCH_DE", "CONTI_DE", "ZF_PRIV", "MAHLE_PRIV", "HELLA_DE",
    "LEONI_DE", "SCHAEFFLER_DE", "ELRING_DE", "STABILUS_DE", "NORMA_DE",
    "SAF_HOLLAND", "GRAMMER_DE", "SGL_CARBON", "WASHTEC_DE", "RATIONAL_DE",
    "DUERR_DE", "KUKA_DE", "AIXTRON_DE", "INFINEON_DE", "VALEO_FR",
    "FAURECIA_FR", "PLASTIC_OMNIUM", "FORVIA_FR", "APTIV_US", "AUTOLIV_US",
    "TENECO_US", "DANA_US", "MODINE_US", "GENTEX_US", "DORMAN_US",
    "STANDARD_MOTOR", "SUPERIOR_IND", "SHILOH_US", "STONERIDGE_US",
    "METHODE_US", "SENSATA_US",
]

dates = pd.date_range("2015-01-01", "2023-01-01", freq="W")
n = len(dates)

log_ret = np.random.normal(0.0012, 0.024, n)

# Q1 2020 crash (~-45% peak-to-trough) + partial recovery
crash_start = int(n * 0.535)
log_ret[crash_start : crash_start + 6]      -= np.linspace(0.01, 0.06, 6)
log_ret[crash_start + 6 : crash_start + 16] += np.linspace(0.01, 0.04, 10)

vw_returns = log_ret

features = {}
for ticker in SUPPLY_CHAIN_TICKERS:
    beta  = np.random.uniform(0.35, 0.85)
    lag   = np.random.randint(1, 5)
    noise = np.random.normal(0, 0.014, n)
    features[f"{ticker}_ret"] = np.roll(vw_returns, lag) * beta + noise

df = pd.DataFrame(features, index=dates)
df['VW_ret'] = vw_returns

df['VW_ma4']  = df['VW_ret'].rolling(4).mean()
df['VW_ma13'] = df['VW_ret'].rolling(13).mean()
df['VW_vol4'] = df['VW_ret'].rolling(4).std()

supplier_cols = [f"{t}_ret" for t in SUPPLY_CHAIN_TICKERS]
df['SC_avg_ret']    = df[supplier_cols].mean(axis=1)
df['SC_dispersion'] = df[supplier_cols].std(axis=1)

df['momentum_12w'] = df['VW_ret'].rolling(12).sum()

df = df.dropna()

feature_cols = [c for c in df.columns if c != 'VW_ret']
X = df[feature_cols].values
y = df['VW_ret'].values

split = int(len(X) * 0.75)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_test = df.index[split:]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ElasticNet handles correlated supplier features better than pure Lasso.
# Tree-based models included for comparison.
models = {
    'Elastic Net':    ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=10_000, random_state=42),
    'Decision Tree':  DecisionTreeRegressor(max_depth=4, min_samples_leaf=15, random_state=42),
    'Gradient Boost': GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    'Random Forest':  RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42),
}

predictions = {}
for name, mdl in models.items():
    mdl.fit(X_train_sc, y_train)
    predictions[name] = mdl.predict(X_test_sc)


def compute_metrics(y_true, y_pred):
    return {
        'RMSE':   np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE':    mean_absolute_error(y_true, y_pred),
        'R2':     r2_score(y_true, y_pred),
        'DirAcc': np.mean(np.sign(y_true) == np.sign(y_pred)),
    }


def backtest(pred, actual, transaction_cost=0.0005):
    pos = pred / (np.abs(pred).max() + 1e-9)
    tc  = transaction_cost * np.abs(np.diff(pos, prepend=pos[0]))

    strategy_ret = pos * actual - tc
    cum_strat = np.cumprod(1 + strategy_ret) - 1
    cum_bnh   = np.cumprod(1 + actual) - 1

    sharpe = strategy_ret.mean() / (strategy_ret.std() + 1e-9) * np.sqrt(52)

    wealth = np.cumprod(1 + strategy_ret)
    peak   = np.maximum.accumulate(wealth)
    max_dd = ((wealth - peak) / peak).min()

    return {
        'cum':    cum_strat[-1],
        'bnh':    cum_bnh[-1],
        'sharpe': sharpe,
        'mdd':    max_dd,
        'cs':     cum_strat,
        'cb':     cum_bnh,
        'sr':     strategy_ret,
    }


metrics = {name: compute_metrics(y_test, predictions[name]) for name in predictions}
bt      = {name: backtest(predictions[name], y_test)        for name in predictions}

print("Prediction metrics")
for name, m in metrics.items():
    print(f"  {name:<16}  RMSE={m['RMSE']:.5f}  R2={m['R2']:.4f}  DirAcc={m['DirAcc']:.1%}")

print("\nBacktest summary")
for name, b in bt.items():
    print(f"  {name:<16}  cum={b['cum']:.2%}  Sharpe={b['sharpe']:.2f}  MaxDD={b['mdd']:.2%}")


MODEL_COLORS = {
    'Elastic Net':    '#1a6b3a',
    'Decision Tree':  '#1a3c5e',
    'Gradient Boost': '#c47a10',
    'Random Forest':  '#b03030',
}
GRAY = '#6c757d'

model_names = list(models.keys())

# fig 1 — cumulative returns
fig, ax = plt.subplots(figsize=(13, 6))

for name, b in bt.items():
    lw    = 2.8 if name == 'Elastic Net' else 1.6
    alpha = 0.9 if name == 'Elastic Net' else 0.7
    ax.plot(dates_test, b['cs'] * 100, color=MODEL_COLORS[name], lw=lw,
            label=f"{name} ({b['cum']:.1%})", alpha=alpha)

ax.plot(dates_test, bt['Elastic Net']['cb'] * 100, color=GRAY, lw=2.2, ls='--',
        label=f"Buy & Hold ({bt['Elastic Net']['bnh']:.1%})")
ax.fill_between(
    dates_test,
    bt['Elastic Net']['cs'] * 100,
    bt['Elastic Net']['cb'] * 100,
    where=bt['Elastic Net']['cs'] > bt['Elastic Net']['cb'],
    alpha=0.12, color=MODEL_COLORS['Elastic Net'],
)

ax.set_title("Cumulative Returns — ML Strategies vs Buy & Hold (2015–2023)",
             fontsize=14, fontweight='bold')
ax.set_ylabel("Cumulative Return (%)")
ax.set_xlabel("Date")
ax.legend(loc='upper left', fontsize=10)
ax.annotate(
    "COVID-19 crash",
    xy=(pd.Timestamp("2020-03-15"), -3),
    xytext=(pd.Timestamp("2019-03-01"), -8),
    arrowprops=dict(arrowstyle='->', color='gray'),
    fontsize=9, color='gray',
)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig1_cumulative_returns.png', dpi=160, bbox_inches='tight')
plt.close()

# fig 2 — per-metric bars
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
colors = [MODEL_COLORS[n] for n in model_names]


def metric_bar(ax, values, title, fmt_fn, higher_is_better=True):
    bars = ax.bar(model_names, values, color=colors, alpha=0.85,
                  edgecolor='white', linewidth=1.5)
    best_idx = values.index(max(values) if higher_is_better else min(values))
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=20, ha='right', fontsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.015,
                fmt_fn(val), ha='center', fontsize=8, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.22)


metric_bar(axes[0], [metrics[n]['DirAcc'] for n in model_names],
           "Directional accuracy", lambda v: f"{v:.1%}")
metric_bar(axes[1], [metrics[n]['R2'] for n in model_names],
           "R2", lambda v: f"{v:.3f}")
metric_bar(axes[2], [1000 * metrics[n]['MAE'] for n in model_names],
           "MAE x1000", lambda v: f"{v:.2f}", higher_is_better=False)
metric_bar(axes[3], [bt[n]['sharpe'] for n in model_names],
           "Sharpe ratio", lambda v: f"{v:.2f}")

plt.suptitle("Model comparison", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_model_metrics.png', dpi=160, bbox_inches='tight')
plt.close()

# fig 3 — predicted vs actual scatter
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, name in zip(axes, model_names):
    y_pred = predictions[name]
    lim = max(np.abs(y_test).max(), np.abs(y_pred).max()) * 1.1
    ax.scatter(y_test, y_pred, alpha=0.3, s=12, color=MODEL_COLORS[name])
    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    m = metrics[name]
    ax.set_title(f"{name}\nR2={m['R2']:.3f}  Dir={m['DirAcc']:.1%}",
                 fontweight='bold', fontsize=9)
    ax.set_xlabel("Actual")
    if name == 'Elastic Net':
        ax.set_ylabel("Predicted")

plt.suptitle("Predicted vs actual weekly returns", fontsize=13, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig3_scatter.png', dpi=160, bbox_inches='tight')
plt.close()

# fig 4 — elastic net coefficients
en_model = models['Elastic Net']
coef = pd.Series(en_model.coef_, index=feature_cols)
top_coef = pd.concat([coef.nlargest(12), coef.nsmallest(8)]).sort_values()

fig, ax = plt.subplots(figsize=(10, 7))
bar_colors = [MODEL_COLORS['Elastic Net'] if v > 0 else '#b03030'
              for v in top_coef.values]
ax.barh(range(len(top_coef)), top_coef.values, color=bar_colors, alpha=0.85)
ax.set_yticks(range(len(top_coef)))
ax.set_yticklabels(
    [f.replace('_ret', '').replace('_', ' ') for f in top_coef.index],
    fontsize=9,
)
ax.axvline(0, color='black', lw=1)
ax.set_title("Elastic Net: feature coefficients", fontsize=12, fontweight='bold')
ax.set_xlabel("Coefficient value")
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig4_feature_importance.png', dpi=160, bbox_inches='tight')
plt.close()

# fig 5 — rolling sharpe (elastic net)
fig, ax = plt.subplots(figsize=(13, 4))

sr_series = pd.Series(bt['Elastic Net']['sr'])
rolling_sharpe = sr_series.rolling(52).apply(
    lambda x: x.mean() / (x.std() + 1e-9) * np.sqrt(52)
)

ax.plot(dates_test, rolling_sharpe, color=MODEL_COLORS['Elastic Net'], lw=1.8)
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.axhline(1, color='#c47a10', lw=0.8, ls='--', label='Sharpe = 1')
ax.fill_between(dates_test, rolling_sharpe, 0,
                where=rolling_sharpe > 0, alpha=0.18,
                color=MODEL_COLORS['Elastic Net'], label='Positive')
ax.fill_between(dates_test, rolling_sharpe, 0,
                where=rolling_sharpe < 0, alpha=0.18,
                color='#b03030', label='Negative')
ax.set_title("Rolling 52-week Sharpe — Elastic Net", fontsize=13, fontweight='bold')
ax.set_ylabel("Sharpe ratio")
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig5_rolling_sharpe.png', dpi=160, bbox_inches='tight')
plt.close()

# fig 6 — drawdown
fig, ax = plt.subplots(figsize=(13, 4))

for name, b in bt.items():
    wealth = np.cumprod(1 + b['sr'])
    peak   = np.maximum.accumulate(wealth)
    dd     = (wealth - peak) / peak * 100
    alpha  = 0.85 if name == 'Elastic Net' else 0.6
    ax.plot(dates_test, dd, color=MODEL_COLORS[name], lw=1.6, label=name, alpha=alpha)

ax.axhline(0, color='gray', lw=0.5)
ax.set_title("Drawdown — all strategies", fontsize=13, fontweight='bold')
ax.set_ylabel("Drawdown (%)")
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig6_drawdown.png', dpi=160, bbox_inches='tight')
plt.close()

print("figures saved.")

rows = []
for name in model_names:
    m, b = metrics[name], bt[name]
    rows.append({
        'Model':       name,
        'RMSE':        m['RMSE'],
        'MAE':         m['MAE'],
        'R2':          m['R2'],
        'DirAcc':      m['DirAcc'],
        'CumReturn':   b['cum'],
        'BuyAndHold':  b['bnh'],
        'Sharpe':      b['sharpe'],
        'MaxDrawdown': b['mdd'],
    })

pd.DataFrame(rows).to_csv('/home/claude/vw_ml_trading/results_summary.csv', index=False)
print("csv saved.")

"""
Run user regime-switching models with configurable parameters.
Can be triggered from the Shiny dashboard frontend.

Usage:
    python run_user_models.py                          # run all models with defaults
    python run_user_models.py --config params.yaml    # run with custom config
    python run_user_models.py --model simple_hmm      # run single model
"""

import argparse
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.regime_switching import markov_regression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import warnings
warnings.filterwarnings('ignore')


# ── Default Configuration ────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "data": {
        "ticker": "SPY",
        "train_start": "2000-01-01",
        "train_end": "2014-12-31",
        "test_start": "2015-01-01",
        "test_end": "2026-03-25",
        "transaction_cost": 0.001,
    },
    "simple_hmm": {
        "n_states": 3,
        "features": ["Returns_lag1"],
        "covariance_type": "full",
        "n_iter": 1000,
        "random_state": 44,
    },
    "hmm": {
        "n_states": 5,
        "features": ["Returns_lag1", "Volatility_lag1", "Volume_Change_lag1", "HL_Range_lag1"],
        "covariance_type": "full",
        "n_iter": 1000,
        "random_state": 42,
    },
    "hsmm": {
        "n_states": 5,
        "features": ["Returns_lag1", "Volatility_lag1", "Volume_Change_lag1", "HL_Range_lag1"],
        "covariance_type": "full",
        "n_iter": 1000,
        "random_state": 43,
    },
    "msgarch": {
        "k_regimes": 3,
        "switching_variance": True,
        "maxiter": 1000,
    },
}


# ── Helper Functions ─────────────────────────────────────────────────────────

def load_data(cfg):
    """Download and prepare feature data."""
    data_cfg = cfg["data"]
    ticker = data_cfg["ticker"]
    print(f"  Downloading {ticker}...")
    data = yf.download(ticker, start=data_cfg["train_start"], end=data_cfg["test_end"], progress=False)
    # Flatten MultiIndex columns from newer yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(20).std()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['HL_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Returns_lag1'] = data['Returns'].shift(1)
    data['Volatility_lag1'] = data['Volatility'].shift(1)
    data['Volume_Change_lag1'] = data['Volume_Change'].shift(1)
    data['HL_Range_lag1'] = data['HL_Range'].shift(1)
    data = data.dropna()

    train = data[data_cfg["train_start"]:data_cfg["train_end"]]
    test = data[data_cfg["test_start"]:data_cfg["test_end"]]
    return data, train, test


def calculate_strategy_returns(returns, regimes, bear_regime, transaction_cost=0.001):
    """Calculate strategy returns with transaction costs."""
    positions = np.ones(len(returns))
    positions[regimes == bear_regime] = 0
    trades = np.abs(np.diff(positions, prepend=1))
    n_trades = int(trades.sum())
    strategy_returns = returns.copy()
    strategy_returns[trades > 0] -= transaction_cost
    strategy_returns = strategy_returns * positions
    return strategy_returns, positions, n_trades


def calculate_metrics(strategy_returns, market_returns):
    """Calculate comprehensive performance and risk metrics."""
    strategy_cum = (1 + strategy_returns).cumprod()[-1] - 1
    market_cum = (1 + market_returns).cumprod()[-1] - 1
    
    # Basic metrics
    strategy_sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    strategy_vol = np.sqrt(252) * strategy_returns.std()
    
    # Drawdown
    strategy_cum_series = (1 + strategy_returns).cumprod()
    strategy_running_max = np.maximum.accumulate(strategy_cum_series)
    strategy_dd = ((strategy_cum_series - strategy_running_max) / strategy_running_max).min()
    
    # Tracking error and IR
    tracking_error = (strategy_returns - market_returns).std() * np.sqrt(252)
    ir = (strategy_cum - market_cum) / tracking_error if tracking_error > 0 else 0
    
    # Risk metrics
    # Sortino ratio (downside deviation)
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else strategy_returns.std()
    sortino = np.sqrt(252) * strategy_returns.mean() / downside_std if downside_std > 0 else 0
    
    # Calmar ratio (return / max drawdown)
    annual_return = (1 + strategy_cum)**(252/len(strategy_returns)) - 1
    calmar = annual_return / abs(strategy_dd) if strategy_dd != 0 else 0
    
    # VaR and CVaR (95% confidence)
    var_95 = np.percentile(strategy_returns, 5) * np.sqrt(252) * 100
    cvar_95 = strategy_returns[strategy_returns <= np.percentile(strategy_returns, 5)].mean() * np.sqrt(252) * 100
    
    # Win rate
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns) * 100

    return {
        'total_return': strategy_cum * 100,
        'market_return': market_cum * 100,
        'outperformance': (strategy_cum - market_cum) * 100,
        'sharpe': strategy_sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'volatility': strategy_vol * 100,
        'max_drawdown': strategy_dd * 100,
        'ir_vs_market': ir,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': win_rate,
    }


def save_results_csv(output_dir, metrics, n_trades, total_days):
    """Save results in the format expected by generic_model_tab."""
    turnover = n_trades / total_days if total_days > 0 else 0
    results_df = pd.DataFrame([{
        'target_te': 0.03,
        'sharpe': metrics['sharpe'],
        'sortino': metrics['sortino'],
        'calmar': metrics['calmar'],
        'ir_vs_market': metrics['ir_vs_market'],
        'max_drawdown': metrics['max_drawdown'],
        'volatility': metrics['volatility'],
        'active_return': metrics['outperformance'],
        'var_95': metrics['var_95'],
        'cvar_95': metrics['cvar_95'],
        'win_rate': metrics['win_rate'],
        'turnover': turnover,
    }])
    results_df.to_csv(output_dir / 'results.csv', index=False)


def save_returns_csv(output_dir, test_data, strategy_returns, positions):
    """Save daily returns."""
    returns_df = pd.DataFrame({
        'date': test_data.index,
        'strategy': strategy_returns,
        'market': test_data['Returns'].values[:len(strategy_returns)],
        'buy_hold': test_data['Returns'].values[:len(strategy_returns)],
        'position': positions
    })
    returns_df.to_csv(output_dir / 'returns.csv', index=False)


def save_regimes_csv(output_dir, test_data, regimes, n_states):
    """Save regime classifications."""
    if n_states <= 3:
        regime_names = {0: 'Bull', 1: 'Sideways', 2: 'Bear'}
    else:
        regime_names = {0: 'Bull', 1: 'Moderate Bull', 2: 'Sideways', 3: 'Moderate Bear', 4: 'Bear'}
    regimes_df = pd.DataFrame({
        'date': test_data.index[:len(regimes)],
        'regime': regimes,
        'regime_name': [regime_names.get(r, f'State {r}') for r in regimes]
    })
    regimes_df.to_csv(output_dir / 'regimes.csv', index=False)


def plot_cumulative_returns(output_dir, test_data, strategy_returns):
    """Plot cumulative returns comparison."""
    market_returns = test_data['Returns'].values[:len(strategy_returns)]
    fig, ax = plt.subplots(figsize=(12, 6))
    strategy_cum = (1 + pd.Series(strategy_returns)).cumprod()
    market_cum = (1 + pd.Series(market_returns)).cumprod()
    dates = test_data.index[:len(strategy_returns)]
    ax.plot(dates, strategy_cum, label='Strategy', linewidth=2, color='#2196F3')
    ax.plot(dates, market_cum, label='Buy & Hold', linewidth=2, color='#FF9800', alpha=0.7)
    ax.set_title('Cumulative Returns: Strategy vs Buy & Hold', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Growth of $1')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plots/cumulative_returns.png', dpi=100)
    plt.close()


def plot_regime_timeline(output_dir, test_data, regimes, n_states):
    """Plot regime classifications over time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    dates = test_data.index[:len(regimes)]
    colors = ['#4CAF50', '#FFC107', '#FF5722', '#9C27B0', '#607D8B']
    for s in range(n_states):
        mask = np.array(regimes) == s
        ax.fill_between(dates, 0, 1, where=mask, alpha=0.6, color=colors[s % len(colors)], label=f'State {s}')
    ax.set_title('Regime Timeline', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'plots/regime_timeline.png', dpi=100)
    plt.close()


def plot_regime_characteristics(output_dir, test_data, regimes, n_states):
    """Plot regime characteristics (mean return, volatility per regime)."""
    returns = test_data['Returns'].values[:len(regimes)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    means, vols, counts = [], [], []
    for s in range(n_states):
        mask = np.array(regimes) == s
        r = returns[mask]
        means.append(r.mean() * 252 * 100)
        vols.append(r.std() * np.sqrt(252) * 100)
        counts.append(mask.sum())

    colors = ['#4CAF50', '#FFC107', '#FF5722', '#9C27B0', '#607D8B']
    axes[0].bar(range(n_states), means, color=[colors[i % len(colors)] for i in range(n_states)])
    axes[0].set_title('Annualized Return by Regime (%)')
    axes[0].set_xlabel('Regime')
    axes[1].bar(range(n_states), vols, color=[colors[i % len(colors)] for i in range(n_states)])
    axes[1].set_title('Annualized Volatility by Regime (%)')
    axes[1].set_xlabel('Regime')
    plt.tight_layout()
    plt.savefig(output_dir / 'plots/regime_characteristics.png', dpi=100)
    plt.close()


def plot_transition_matrix(output_dir, regimes, n_states):
    """Plot transition probability matrix."""
    transitions = np.zeros((n_states, n_states))
    for i in range(len(regimes) - 1):
        transitions[regimes[i], regimes[i + 1]] += 1
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    trans_prob = transitions / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(trans_prob, annot=True, fmt='.3f', cmap='Blues', ax=ax,
                xticklabels=[f'State {i}' for i in range(n_states)],
                yticklabels=[f'State {i}' for i in range(n_states)])
    ax.set_title('Transition Probability Matrix', fontsize=14)
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    plt.tight_layout()
    plt.savefig(output_dir / 'plots/transition_matrix.png', dpi=100)
    plt.close()


def plot_drawdown(output_dir, test_data, strategy_returns):
    """Plot drawdown over time."""
    strategy_cum_series = (1 + pd.Series(strategy_returns)).cumprod()
    strategy_running_max = np.maximum.accumulate(strategy_cum_series)
    drawdown = (strategy_cum_series - strategy_running_max) / strategy_running_max
    
    fig, ax = plt.subplots(figsize=(12, 5))
    dates = test_data.index[:len(strategy_returns)]
    ax.fill_between(dates, drawdown * 100, 0, alpha=0.3, color='#dc3545')
    ax.plot(dates, drawdown * 100, color='#dc3545', linewidth=1.5)
    ax.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / 'plots/drawdown.png', dpi=100)
    plt.close()


def plot_rolling_sharpe(output_dir, test_data, strategy_returns):
    """Plot rolling 52-week Sharpe ratio."""
    returns_series = pd.Series(strategy_returns, index=test_data.index[:len(strategy_returns)])
    rolling_sharpe = returns_series.rolling(window=252).apply(
        lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
    )
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, color='#2196F3', linewidth=2)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='Sharpe = 1')
    ax.set_title('Rolling 52-Week Sharpe Ratio', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'plots/rolling_sharpe.png', dpi=100)
    plt.close()


def plot_annual_heatmap(output_dir, test_data, strategy_returns):
    """Plot annual returns heatmap."""
    returns_series = pd.Series(strategy_returns, index=test_data.index[:len(strategy_returns)])
    
    # Calculate monthly returns
    monthly_returns = returns_series.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    
    # Pivot to year x month
    monthly_returns_df = pd.DataFrame({
        'year': monthly_returns.index.year,
        'month': monthly_returns.index.month,
        'return': monthly_returns.values * 100
    })
    
    if len(monthly_returns_df) == 0:
        return
    
    pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Return (%)'}, ax=ax, linewidths=0.5)
    ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_labels, rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'plots/annual_heatmap.png', dpi=100)
    plt.close()


# ── Model Runners ────────────────────────────────────────────────────────────

def run_simple_hmm(cfg, train, test):
    """Run Simple-HMM model."""
    model_cfg = cfg["simple_hmm"]
    features = model_cfg["features"]
    n_states = model_cfg["n_states"]
    txn_cost = cfg["data"]["transaction_cost"]

    X_train = train[features].values
    X_test = test[features].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=model_cfg["covariance_type"],
        n_iter=model_cfg["n_iter"],
        random_state=model_cfg["random_state"]
    )
    model.fit(X_train_scaled)
    regimes = model.predict(X_test_scaled)

    # Identify bear regime
    regime_returns = {}
    for s in range(n_states):
        mask = regimes == s
        regime_returns[s] = test['Returns'].values[mask].mean()
    bear_regime = min(regime_returns.keys(), key=lambda k: regime_returns[k])

    strategy_returns, positions, n_trades = calculate_strategy_returns(
        test['Returns'].values, regimes, bear_regime, txn_cost
    )
    metrics = calculate_metrics(strategy_returns, test['Returns'].values)

    # Save
    output_dir = Path('outputs/simple_hmm')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)

    save_results_csv(output_dir, metrics, n_trades, len(test))
    save_returns_csv(output_dir, test, strategy_returns, positions)
    save_regimes_csv(output_dir, test, regimes, n_states)
    plot_cumulative_returns(output_dir, test, strategy_returns)
    plot_drawdown(output_dir, test, strategy_returns)
    plot_rolling_sharpe(output_dir, test, strategy_returns)
    plot_regime_timeline(output_dir, test, regimes, n_states)
    plot_regime_characteristics(output_dir, test, regimes, n_states)
    plot_transition_matrix(output_dir, regimes, n_states)
    plot_annual_heatmap(output_dir, test, strategy_returns)

    return metrics, n_trades


def run_hmm(cfg, train, test):
    """Run 5-state HMM model."""
    model_cfg = cfg["hmm"]
    features = model_cfg["features"]
    n_states = model_cfg["n_states"]
    txn_cost = cfg["data"]["transaction_cost"]

    X_train = train[features].values
    X_test = test[features].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=model_cfg["covariance_type"],
        n_iter=model_cfg["n_iter"],
        random_state=model_cfg["random_state"]
    )
    model.fit(X_train_scaled)
    regimes = model.predict(X_test_scaled)

    regime_returns = {}
    for s in range(n_states):
        mask = regimes == s
        regime_returns[s] = test['Returns'].values[mask].mean()
    bear_regime = min(regime_returns.keys(), key=lambda k: regime_returns[k])

    strategy_returns, positions, n_trades = calculate_strategy_returns(
        test['Returns'].values, regimes, bear_regime, txn_cost
    )
    metrics = calculate_metrics(strategy_returns, test['Returns'].values)

    output_dir = Path('outputs/hmm')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)

    save_results_csv(output_dir, metrics, n_trades, len(test))
    save_returns_csv(output_dir, test, strategy_returns, positions)
    save_regimes_csv(output_dir, test, regimes, n_states)
    plot_cumulative_returns(output_dir, test, strategy_returns)
    plot_drawdown(output_dir, test, strategy_returns)
    plot_rolling_sharpe(output_dir, test, strategy_returns)
    plot_regime_timeline(output_dir, test, regimes, n_states)
    plot_regime_characteristics(output_dir, test, regimes, n_states)
    plot_transition_matrix(output_dir, regimes, n_states)
    plot_annual_heatmap(output_dir, test, strategy_returns)

    return metrics, n_trades


def run_hsmm(cfg, train, test):
    """Run 5-state HSMM model."""
    model_cfg = cfg["hsmm"]
    features = model_cfg["features"]
    n_states = model_cfg["n_states"]
    txn_cost = cfg["data"]["transaction_cost"]

    X_train = train[features].values
    X_test = test[features].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type=model_cfg["covariance_type"],
        n_iter=model_cfg["n_iter"],
        random_state=model_cfg["random_state"]
    )
    model.fit(X_train_scaled)
    regimes = model.predict(X_test_scaled)

    regime_returns = {}
    for s in range(n_states):
        mask = regimes == s
        regime_returns[s] = test['Returns'].values[mask].mean()
    bear_regime = min(regime_returns.keys(), key=lambda k: regime_returns[k])

    strategy_returns, positions, n_trades = calculate_strategy_returns(
        test['Returns'].values, regimes, bear_regime, txn_cost
    )
    metrics = calculate_metrics(strategy_returns, test['Returns'].values)

    output_dir = Path('outputs/hsmm')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)

    save_results_csv(output_dir, metrics, n_trades, len(test))
    save_returns_csv(output_dir, test, strategy_returns, positions)
    save_regimes_csv(output_dir, test, regimes, n_states)
    plot_cumulative_returns(output_dir, test, strategy_returns)
    plot_drawdown(output_dir, test, strategy_returns)
    plot_rolling_sharpe(output_dir, test, strategy_returns)
    plot_regime_timeline(output_dir, test, regimes, n_states)
    plot_regime_characteristics(output_dir, test, regimes, n_states)
    plot_transition_matrix(output_dir, regimes, n_states)
    plot_annual_heatmap(output_dir, test, strategy_returns)

    return metrics, n_trades


def run_msgarch(cfg, train, test):
    """Run MS-GARCH model."""
    model_cfg = cfg["msgarch"]
    k_regimes = model_cfg["k_regimes"]
    txn_cost = cfg["data"]["transaction_cost"]

    # Prepare training data — align lagged returns and drop NaN/inf
    train_returns = train['Returns'].values.astype(float)
    train_lagged = np.roll(train_returns, 1)
    train_lagged[0] = 0.0
    # Skip first row (invalid lag)
    endog = train_returns[1:]
    exog = train_lagged[1:]
    # Remove any remaining NaN/inf
    valid = np.isfinite(endog) & np.isfinite(exog)
    endog = endog[valid]
    exog = exog[valid]

    model_msgarch = markov_regression.MarkovRegression(
        endog=endog,
        k_regimes=k_regimes,
        exog=exog,
        switching_variance=model_cfg["switching_variance"]
    )
    results_msgarch = model_msgarch.fit(maxiter=model_cfg["maxiter"], disp=False)

    # Get regime assignments and stats from training
    sp = results_msgarch.smoothed_marginal_probabilities  # numpy array (n, k)
    regimes_train = np.argmax(sp, axis=1)

    # Compute regime volatilities from training data
    regime_stds = []
    regime_means = {}
    for r in range(k_regimes):
        mask = regimes_train == r
        regime_stds.append(endog[mask].std() if mask.sum() > 1 else 0.01)
        regime_means[r] = endog[mask].mean() if mask.sum() > 0 else 0.0
    regime_order = np.argsort(regime_stds)  # low vol → high vol
    bear_regime = min(regime_means.keys(), key=lambda k: regime_means[k])

    # Classify test regimes using rolling volatility thresholds
    test_returns = test['Returns'].values.astype(float)
    rolling_vol = pd.Series(test_returns).rolling(20).std().values
    vol_33 = np.nanquantile(rolling_vol, 0.33)
    vol_66 = np.nanquantile(rolling_vol, 0.66)

    regimes_msgarch = np.full(len(test_returns), regime_order[1], dtype=int)
    for i in range(len(test_returns)):
        if np.isnan(rolling_vol[i]):
            regimes_msgarch[i] = regime_order[1]  # medium vol default
        elif rolling_vol[i] < vol_33:
            regimes_msgarch[i] = regime_order[0]  # low vol
        elif rolling_vol[i] < vol_66:
            regimes_msgarch[i] = regime_order[1]  # medium vol
        else:
            regimes_msgarch[i] = regime_order[2]  # high vol

    strategy_returns, positions, n_trades = calculate_strategy_returns(
        test_returns, regimes_msgarch, bear_regime, txn_cost
    )
    metrics = calculate_metrics(strategy_returns, test_returns)

    output_dir = Path('outputs/msgarch')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)

    save_results_csv(output_dir, metrics, n_trades, len(test))
    save_returns_csv(output_dir, test, strategy_returns, positions)
    save_regimes_csv(output_dir, test, regimes_msgarch, k_regimes)
    plot_cumulative_returns(output_dir, test, strategy_returns)
    plot_drawdown(output_dir, test, strategy_returns)
    plot_rolling_sharpe(output_dir, test, strategy_returns)
    plot_regime_timeline(output_dir, test, regimes_msgarch, k_regimes)
    plot_regime_characteristics(output_dir, test, regimes_msgarch, k_regimes)
    plot_transition_matrix(output_dir, regimes_msgarch, k_regimes)
    plot_annual_heatmap(output_dir, test, strategy_returns)

    return metrics, n_trades


# ── Main ─────────────────────────────────────────────────────────────────────

MODEL_RUNNERS = {
    "simple_hmm": run_simple_hmm,
    "hmm": run_hmm,
    "hsmm": run_hsmm,
    "msgarch": run_msgarch,
}


def main():
    parser = argparse.ArgumentParser(description="Run regime-switching models")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--model", default=None, choices=list(MODEL_RUNNERS.keys()),
                        help="Run a specific model (default: all)")
    args = parser.parse_args()

    # Load config
    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        with open(args.config) as f:
            user_cfg = yaml.safe_load(f)
        # Deep merge
        for key, val in user_cfg.items():
            if isinstance(val, dict) and key in cfg:
                cfg[key].update(val)
            else:
                cfg[key] = val

    models_to_run = [args.model] if args.model else list(MODEL_RUNNERS.keys())

    print("=" * 60)
    print("REGIME-SWITCHING MODELS PIPELINE")
    print("=" * 60)
    print(f"  Ticker: {cfg['data']['ticker']}")
    print(f"  Train: {cfg['data']['train_start']} to {cfg['data']['train_end']}")
    print(f"  Test:  {cfg['data']['test_start']} to {cfg['data']['test_end']}")
    print(f"  Models: {', '.join(models_to_run)}")
    print("=" * 60)

    # Load data
    print(f"\n[1/{len(models_to_run)+1}] Loading data...")
    data, train, test = load_data(cfg)
    print(f"  Train: {len(train)} days, Test: {len(test)} days")

    # Run models
    all_metrics = {}
    for i, model_name in enumerate(models_to_run, start=2):
        print(f"\n[{i}/{len(models_to_run)+1}] Running {model_name}...")
        runner = MODEL_RUNNERS[model_name]
        metrics, n_trades = runner(cfg, train, test)
        all_metrics[model_name] = metrics
        print(f"  Sharpe: {metrics['sharpe']:.3f}, Max DD: {metrics['max_drawdown']:.2f}%, "
              f"Active Return: {metrics['outperformance']:.2f}%")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    for name, m in all_metrics.items():
        print(f"  {name:12s} | Sharpe: {m['sharpe']:.3f} | Max DD: {m['max_drawdown']:.1f}% | "
              f"Active: {m['outperformance']:+.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()

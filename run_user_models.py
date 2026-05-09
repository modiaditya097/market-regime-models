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
    """Calculate performance metrics."""
    strategy_cum = (1 + strategy_returns).cumprod()[-1] - 1
    market_cum = (1 + market_returns).cumprod()[-1] - 1
    strategy_sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    strategy_vol = np.sqrt(252) * strategy_returns.std()
    strategy_cum_series = (1 + strategy_returns).cumprod()
    strategy_running_max = np.maximum.accumulate(strategy_cum_series)
    strategy_dd = ((strategy_cum_series - strategy_running_max) / strategy_running_max).min()
    tracking_error = (strategy_returns - market_returns).std() * np.sqrt(252)
    ir = (strategy_cum - market_cum) / tracking_error if tracking_error > 0 else 0

    return {
        'total_return': strategy_cum * 100,
        'market_return': market_cum * 100,
        'outperformance': (strategy_cum - market_cum) * 100,
        'sharpe': strategy_sharpe,
        'volatility': strategy_vol * 100,
        'max_drawdown': strategy_dd * 100,
        'ir_vs_market': ir,
    }


def save_results_csv(output_dir, metrics, n_trades, total_days):
    """Save results in the format expected by generic_model_tab."""
    turnover = n_trades / total_days if total_days > 0 else 0
    results_df = pd.DataFrame([{
        'target_te': 0.03,
        'sharpe': metrics['sharpe'],
        'ir_vs_market': metrics['ir_vs_market'],
        'max_drawdown': metrics['max_drawdown'],
        'volatility': metrics['volatility'],
        'active_return': metrics['outperformance'],
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
    plot_regime_timeline(output_dir, test, regimes, n_states)
    plot_regime_characteristics(output_dir, test, regimes, n_states)
    plot_transition_matrix(output_dir, regimes, n_states)

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
    plot_regime_timeline(output_dir, test, regimes, n_states)
    plot_regime_characteristics(output_dir, test, regimes, n_states)
    plot_transition_matrix(output_dir, regimes, n_states)

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
    plot_regime_timeline(output_dir, test, regimes, n_states)
    plot_regime_characteristics(output_dir, test, regimes, n_states)
    plot_transition_matrix(output_dir, regimes, n_states)

    return metrics, n_trades


def run_msgarch(cfg, train, test):
    """Run MS-GARCH model."""
    model_cfg = cfg["msgarch"]
    k_regimes = model_cfg["k_regimes"]
    txn_cost = cfg["data"]["transaction_cost"]

    returns_series = pd.Series(train['Returns'].values, index=train.index)
    returns_lagged = returns_series.shift(1).dropna()
    returns_current = returns_series.iloc[1:]

    model_msgarch = markov_regression.MarkovRegression(
        endog=returns_current,
        k_regimes=k_regimes,
        exog=returns_lagged,
        switching_variance=model_cfg["switching_variance"]
    )
    results_msgarch = model_msgarch.fit(maxiter=model_cfg["maxiter"], disp=False)

    # Classify test regimes using volatility
    test_returns_series = pd.Series(test['Returns'].values, index=test.index)
    test_returns_current = test_returns_series.iloc[1:]
    regimes_msgarch = np.zeros(len(test_returns_current), dtype=int)

    regime_stds = []
    for regime in range(k_regimes):
        regime_stds.append(np.sqrt(results_msgarch.params[f'sigma2[{regime}]']))
    regime_order = np.argsort(regime_stds)

    rolling_vol = pd.Series(test_returns_current.values).rolling(20).std()
    vol_33 = rolling_vol.quantile(0.33)
    vol_66 = rolling_vol.quantile(0.66)

    for i in range(len(test_returns_current)):
        if pd.isna(rolling_vol.iloc[i]):
            regimes_msgarch[i] = regime_order[1]
        elif rolling_vol.iloc[i] < vol_33:
            regimes_msgarch[i] = regime_order[0]
        elif rolling_vol.iloc[i] < vol_66:
            regimes_msgarch[i] = regime_order[1]
        else:
            regimes_msgarch[i] = regime_order[2]

    # Get bear regime from training
    smoothed_probs = results_msgarch.smoothed_marginal_probabilities
    regimes_train = smoothed_probs.idxmax(axis=1).values
    regime_returns_train = {}
    for s in range(k_regimes):
        mask = regimes_train == s
        regime_returns_train[s] = returns_current.values[mask].mean()
    bear_regime = min(regime_returns_train.keys(), key=lambda k: regime_returns_train[k])

    strategy_returns, positions, n_trades = calculate_strategy_returns(
        test_returns_current.values, regimes_msgarch, bear_regime, txn_cost
    )
    metrics = calculate_metrics(strategy_returns, test_returns_current.values)

    output_dir = Path('outputs/msgarch')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)

    test_aligned = test.iloc[1:]
    save_results_csv(output_dir, metrics, n_trades, len(test_aligned))
    save_returns_csv(output_dir, test_aligned, strategy_returns, positions)
    save_regimes_csv(output_dir, test_aligned, regimes_msgarch, k_regimes)
    plot_cumulative_returns(output_dir, test_aligned, strategy_returns)
    plot_regime_timeline(output_dir, test_aligned, regimes_msgarch, k_regimes)
    plot_regime_characteristics(output_dir, test_aligned, regimes_msgarch, k_regimes)
    plot_transition_matrix(output_dir, regimes_msgarch, k_regimes)

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

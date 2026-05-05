# SJM Model 1 Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 7 configurable macro features, 2/3-state regime toggle, and walk-forward evaluation to the SJM pipeline and dashboard.

**Architecture:** Backend changes flow from data → features → regime → portfolio → backtest → main.py; frontend changes are isolated to `model1.py` and `analytics.py`. All new params are written to `tmp_config.yaml` by the sidebar before each run.

**Tech Stack:** Python, Shiny for Python, pandas, numpy, yfinance, FRED (urllib), SparseJumpModel (jumpmodels), matplotlib, PyYAML.

**Spec:** `docs/specs/2026-05-04-sjm-enhancements-design.md`

---

## File Map

| File | Change |
|---|---|
| `config.yaml` | Add `training.test_start`, `macro_features`, `walk_forward` |
| `src/data.py` | 7 new fetchers; updated `load_macro()` and `load_all_data()` |
| `src/features.py` | Extended `compute_market_features()` and `compute_all_features()` |
| `src/regime.py` | Use `test_start` from config; forward `macro_extras`/`enabled` |
| `src/portfolio.py` | 3-state logic in `compute_view_returns()` |
| `src/backtest.py` | New `run_walk_forward()` |
| `main.py` | Phase 2 WFE block; pass `macro_extras` to `run_regime_detection()` |
| `shiny_app/components/analytics.py` | New `load_wfe_results()`, `wfe_metrics_html()`, `wfe_folds_plot()` |
| `shiny_app/modules/model1.py` | Accordion sidebar; inner Results/Validation tabs |
| `tests/test_data.py` | Tests for new fetchers and `load_macro()` |
| `tests/test_features.py` | Tests for extended `compute_market_features()` |
| `tests/test_portfolio.py` | Tests for 3-state `compute_view_returns()` |
| `tests/test_backtest.py` | Tests for `run_walk_forward()` |
| `tests/test_analytics.py` | Tests for WFE analytics functions |

---

## Task 1: Update config.yaml with new fields

**Files:**
- Modify: `config.yaml`

- [ ] **Step 1: Add the three new config blocks**

Open `config.yaml` and add the following — insert `test_start` under `training`, and add the two new top-level blocks at the end:

```yaml
training:
  min_train_years: 8
  max_train_years: 12
  refit_freq: "M"
  test_start: "2008-01-01"    # NEW — previously implicit as data_start + min_train_years

macro_features:               # NEW
  enabled:
    - vix_level
    - dxy
    - oil_ret
    - gold_ret
    - real_yield
    - unemployment
    - consumer_sent

walk_forward:                 # NEW
  enabled: false
  n_folds: 6
  fold_test_months: 36
```

- [ ] **Step 2: Verify config loads cleanly**

```bash
python -c "from src.utils import load_config; c = load_config('config.yaml'); print(c['training']['test_start'], c['macro_features'], c['walk_forward'])"
```

Expected output:
```
2008-01-01 {'enabled': ['vix_level', 'dxy', 'oil_ret', 'gold_ret', 'real_yield', 'unemployment', 'consumer_sent']} {'enabled': False, 'n_folds': 6, 'fold_test_months': 36}
```

- [ ] **Step 3: Commit**

```bash
git add config.yaml
git commit -m "feat(config): add test_start, macro_features, walk_forward blocks"
```

---

## Task 2: New macro data fetchers in src/data.py

**Files:**
- Modify: `src/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write failing tests for the new load_macro() signature**

Add to the bottom of `tests/test_data.py`:

```python
from unittest.mock import patch, MagicMock

def _make_fake_series(name, n=300):
    dates = pd.date_range("2000-01-03", periods=n, freq="B")
    return pd.Series(np.abs(np.random.randn(n)) + 1.0, index=dates, name=name)

def test_load_macro_empty_enabled_returns_three_keys():
    fake_vix = _make_fake_series("VIX")
    fake_y   = _make_fake_series("DGS2")
    with patch("src.data._fetch_vix", return_value=fake_vix), \
         patch("src.data._fetch_fred", return_value=fake_y):
        from src.data import load_macro
        result = load_macro("2000-01-01", "2001-01-01", enabled=[])
    assert set(result.keys()) == {"vix", "y2", "y10"}

def test_load_macro_enabled_adds_extra_keys():
    fake_series = _make_fake_series("fake")
    with patch("src.data._fetch_vix", return_value=fake_series), \
         patch("src.data._fetch_fred", return_value=fake_series), \
         patch("src.data._fetch_yf_log_return", return_value=fake_series):
        from src.data import load_macro
        result = load_macro("2000-01-01", "2001-01-01", enabled=["dxy", "oil_ret"])
    assert "dxy"     in result
    assert "oil_ret" in result
    assert "vix"     in result  # base keys still present

def test_load_macro_unknown_key_is_ignored():
    fake_series = _make_fake_series("fake")
    with patch("src.data._fetch_vix", return_value=fake_series), \
         patch("src.data._fetch_fred", return_value=fake_series):
        from src.data import load_macro
        result = load_macro("2000-01-01", "2001-01-01", enabled=["nonexistent_key"])
    assert "nonexistent_key" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py::test_load_macro_empty_enabled_returns_three_keys tests/test_data.py::test_load_macro_enabled_adds_extra_keys tests/test_data.py::test_load_macro_unknown_key_is_ignored -v
```

Expected: FAIL — `load_macro` doesn't accept `enabled` parameter yet.

- [ ] **Step 3: Add helper `_fetch_yf_log_return` and 7 new fetchers to src/data.py**

Add after the existing `_fetch_vix` function (around line 87):

```python
def _fetch_yf_log_return(ticker: str, start: str, end: str) -> pd.Series:
    """Download a Yahoo Finance ticker and return daily log returns."""
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    price = raw["Close"].squeeze()
    price.index = pd.to_datetime(price.index).normalize()
    return np.log(price / price.shift(1)).rename(ticker)


def _fetch_vix_level(start: str, end: str) -> pd.Series:
    raw = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=True)
    s = raw["Close"].squeeze()
    s.index = pd.to_datetime(s.index).normalize()
    return np.log(s).rename("vix_level")


def _fetch_dxy(start: str, end: str) -> pd.Series:
    return _fetch_yf_log_return("DX-Y.NYB", start, end).rename("dxy")


def _fetch_oil(start: str, end: str) -> pd.Series:
    return _fetch_yf_log_return("CL=F", start, end).rename("oil_ret")


def _fetch_gold(start: str, end: str) -> pd.Series:
    return _fetch_yf_log_return("GC=F", start, end).rename("gold_ret")


def _fetch_real_yield(start: str, end: str) -> pd.Series:
    return _fetch_fred("REAINTRATREARAT10Y").rename("real_yield")


def _fetch_unemployment(start: str, end: str) -> pd.Series:
    return _fetch_fred("UNRATE").rename("unemployment")


def _fetch_consumer_sent(start: str, end: str) -> pd.Series:
    return _fetch_fred("UMCSENT").rename("consumer_sent")


_MACRO_EXTRA_FETCHERS: dict = {
    "vix_level":     _fetch_vix_level,
    "dxy":           _fetch_dxy,
    "oil_ret":       _fetch_oil,
    "gold_ret":      _fetch_gold,
    "real_yield":    _fetch_real_yield,
    "unemployment":  _fetch_unemployment,
    "consumer_sent": _fetch_consumer_sent,
}
```

- [ ] **Step 4: Replace the existing `load_macro` function**

Replace the current `load_macro` (around line 122) with:

```python
def load_macro(start: str, end: str, enabled: list = None) -> dict:
    """Download VIX, 2Y yield, 10Y yield, plus any enabled extra macro series.

    enabled: list of keys from _MACRO_EXTRA_FETCHERS to also fetch.
    """
    vix = _fetch_vix(start, end)
    y2  = _fetch_fred("DGS2")
    y10 = _fetch_fred("DGS10")
    result = {"vix": vix, "y2": y2, "y10": y10}
    for key in (enabled or []):
        if key in _MACRO_EXTRA_FETCHERS:
            result[key] = _MACRO_EXTRA_FETCHERS[key](start, end)
    return result
```

- [ ] **Step 5: Update `load_all_data` to pass enabled extras and cache per-feature**

In `load_all_data`, find the block that calls `load_macro` (around line 165) and replace the entire else-branch with:

```python
    else:
        # --- Ken French ---
        ff5_content = _fetch_zip_csv(_FF5_URL)
        mom_content = _fetch_zip_csv(_MOM_URL)
        ff5 = _parse_ken_french_csv(ff5_content, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
        mom = _parse_ken_french_csv(mom_content, ["Mom"])

        start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
        ff5 = ff5.loc[start:end]
        mom = mom.loc[start:end]
        total, active = build_asset_returns(ff5, mom)
        rf = ff5["RF"]

        # --- Core Macro (vix, y2, y10) ---
        enabled_extras = cfg.get("macro_features", {}).get("enabled", [])
        macro_raw = load_macro(start, end, enabled=enabled_extras)
        y2_aligned  = macro_raw["y2"].reindex(total.index, method="ffill")
        y10_aligned = macro_raw["y10"].reindex(total.index, method="ffill")
        vix_aligned = macro_raw["vix"].reindex(total.index, method="ffill")
        macro = {"vix": vix_aligned, "y2": y2_aligned, "y10": y10_aligned}

        # --- Extra Macro Features ---
        for key in enabled_extras:
            if key in macro_raw:
                macro[key] = macro_raw[key].reindex(total.index, method="ffill")

        # Drop dates with NaN in core data
        macro_core_valid = pd.concat([vix_aligned, y2_aligned, y10_aligned], axis=1).notna().all(axis=1)
        valid = total.notna().all(axis=1) & active.notna().all(axis=1) & macro_core_valid
        total, active, rf = total[valid], active[valid], rf[valid]
        for k in macro:
            macro[k] = macro[k][valid]

        # Cache core
        total.to_parquet(total_path)
        active.to_parquet(active_path)
        rf.to_frame().to_parquet(rf_path)
        pd.DataFrame({k: macro[k] for k in ["vix", "y2", "y10"]}).to_parquet(macro_path)

        # Cache extras individually
        for key in enabled_extras:
            if key in macro:
                extra_path = cache_dir / f"macro_{key}.parquet"
                macro[key].to_frame().to_parquet(extra_path)
```

Also update the cache-hit branch (the `if not refresh and all(...)` block) to load cached extras:

```python
    if not refresh and all(p.exists() for p in [total_path, active_path, rf_path, macro_path]):
        total  = pd.read_parquet(total_path)
        active = pd.read_parquet(active_path)
        rf     = pd.read_parquet(rf_path).iloc[:, 0]
        macro_df = pd.read_parquet(macro_path)
        macro  = {col: macro_df[col] for col in macro_df.columns}
        # Load cached extras (fetch missing ones)
        enabled_extras = cfg.get("macro_features", {}).get("enabled", [])
        start, end = cfg["data"]["start_date"], cfg["data"]["end_date"]
        for key in enabled_extras:
            extra_path = cache_dir / f"macro_{key}.parquet"
            if extra_path.exists():
                macro[key] = pd.read_parquet(extra_path).iloc[:, 0]
            elif key in _MACRO_EXTRA_FETCHERS:
                s = _MACRO_EXTRA_FETCHERS[key](start, end)
                macro[key] = s.reindex(total.index, method="ffill")
                macro[key].to_frame().to_parquet(extra_path)
```

- [ ] **Step 6: Run the new tests**

```bash
pytest tests/test_data.py::test_load_macro_empty_enabled_returns_three_keys tests/test_data.py::test_load_macro_enabled_adds_extra_keys tests/test_data.py::test_load_macro_unknown_key_is_ignored -v
```

Expected: all 3 PASS.

- [ ] **Step 7: Run full data test suite to check for regressions**

```bash
pytest tests/test_data.py -v
```

Expected: all existing tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat(data): add 7 macro fetchers, update load_macro() with enabled list"
```

---

## Task 3: Extended macro features in src/features.py

**Files:**
- Modify: `src/features.py`
- Modify: `tests/test_features.py`

- [ ] **Step 1: Write failing tests**

Add to the bottom of `tests/test_features.py`:

```python
from src.features import compute_market_features

def test_compute_market_features_base_shape():
    """Without extras, returns 4 columns."""
    result = compute_market_features(mkt_ret, vix, y2, y10)
    assert result.shape[1] == 4
    assert "mkt_ret_21" in result.columns

def test_compute_market_features_with_extras():
    """With one extra enabled series, returns 5 columns."""
    extras = {"dxy": pd.Series(np.random.randn(N) * 0.01, index=dates)}
    result = compute_market_features(mkt_ret, vix, y2, y10,
                                     macro_extras=extras, enabled=["dxy"])
    assert result.shape[1] == 5
    assert "dxy_21" in result.columns

def test_compute_market_features_unknown_key_skipped():
    """Keys not in macro_extras are silently skipped."""
    result = compute_market_features(mkt_ret, vix, y2, y10,
                                     macro_extras={}, enabled=["nonexistent"])
    assert result.shape[1] == 4

def test_compute_all_features_with_extras():
    """compute_all_features forwards extras to compute_market_features."""
    from src.features import compute_all_features
    extras = {"vix_level": pd.Series(np.log(vix), index=dates)}
    active_rets = {"value": rand_ret, "size": rand_ret}
    result = compute_all_features(active_rets, mkt_ret, vix, y2, y10,
                                  macro_extras=extras, enabled=["vix_level"])
    for factor in ["value", "size"]:
        assert "vix_level" in result[factor].columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_features.py::test_compute_market_features_base_shape tests/test_features.py::test_compute_market_features_with_extras tests/test_features.py::test_compute_market_features_unknown_key_skipped tests/test_features.py::test_compute_all_features_with_extras -v
```

Expected: FAIL — `compute_market_features` doesn't accept `macro_extras` yet.

- [ ] **Step 3: Add `_MACRO_FEATURE_BUILDERS` dict and update `compute_market_features`**

Add after the existing `compute_active_beta` function in `src/features.py`:

```python
_MACRO_FEATURE_BUILDERS: dict = {
    "vix_level":     lambda s: s.rename("vix_level"),
    "dxy":           lambda s: s.ewm(span=21, adjust=False).mean().rename("dxy_21"),
    "oil_ret":       lambda s: s.ewm(span=21, adjust=False).mean().rename("oil_ret_21"),
    "gold_ret":      lambda s: s.ewm(span=21, adjust=False).mean().rename("gold_ret_21"),
    "real_yield":    lambda s: s.diff().ewm(span=21, adjust=False).mean().rename("real_yield_diff_21"),
    "unemployment":  lambda s: s.diff().ewm(span=21, adjust=False).mean().rename("unemployment_diff_21"),
    "consumer_sent": lambda s: s.diff().ewm(span=21, adjust=False).mean().rename("consumer_sent_diff_21"),
}
```

Then replace the existing `compute_market_features` function with:

```python
def compute_market_features(
    mkt_ret: pd.Series,
    vix: pd.Series,
    y2: pd.Series,
    y10: pd.Series,
    macro_extras: dict = None,
    enabled: list = None,
) -> pd.DataFrame:
    """4 always-on market features plus any enabled extras."""
    mkt_feat    = mkt_ret.ewm(span=21, adjust=False).mean().rename("mkt_ret_21")
    vix_logret  = np.log(vix / vix.shift(1))
    vix_feat    = vix_logret.ewm(span=21, adjust=False).mean().rename("vix_21")
    y2_diff     = y2.diff().ewm(span=21, adjust=False).mean().rename("y2_diff_21")
    slope_diff  = (y10 - y2).diff().ewm(span=21, adjust=False).mean().rename("slope_diff_21")
    parts = [mkt_feat, vix_feat, y2_diff, slope_diff]

    extras = macro_extras or {}
    for key in (enabled or []):
        if key in extras and key in _MACRO_FEATURE_BUILDERS:
            parts.append(_MACRO_FEATURE_BUILDERS[key](extras[key]))

    return pd.concat(parts, axis=1)
```

- [ ] **Step 4: Update `compute_all_features` to forward extras**

Replace the existing `compute_all_features` function with:

```python
def compute_all_features(
    active_rets: dict,
    mkt_ret: pd.Series,
    vix: pd.Series,
    y2: pd.Series,
    y10: pd.Series,
    macro_extras: dict = None,
    enabled: list = None,
) -> dict:
    """17+ feature matrix for each factor (13 factor-specific + 4+ market)."""
    mkt_feats = compute_market_features(mkt_ret, vix, y2, y10,
                                        macro_extras=macro_extras, enabled=enabled)
    result = {}
    for factor, ret in active_rets.items():
        factor_feats = compute_factor_features(ret, mkt_ret)
        X = pd.concat([factor_feats, mkt_feats], axis=1)
        X = X.reindex(ret.index)
        result[factor] = X
    return result
```

- [ ] **Step 5: Run new tests**

```bash
pytest tests/test_features.py::test_compute_market_features_base_shape tests/test_features.py::test_compute_market_features_with_extras tests/test_features.py::test_compute_market_features_unknown_key_skipped tests/test_features.py::test_compute_all_features_with_extras -v
```

Expected: all 4 PASS.

- [ ] **Step 6: Run full feature test suite**

```bash
pytest tests/test_features.py -v
```

Expected: all existing tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat(features): add configurable macro extras to compute_market_features"
```

---

## Task 4: Use test_start from config + macro_extras passthrough in src/regime.py

**Files:**
- Modify: `src/regime.py`
- Modify: `tests/test_regime.py`

- [ ] **Step 1: Write a failing test**

Add to the bottom of `tests/test_regime.py`:

```python
def test_run_regime_detection_respects_test_start():
    """Regime labels should not exist before test_start."""
    import numpy as np
    N = 252 * 10
    dates = pd.date_range("2000-01-03", periods=N, freq="B")
    active = {"value": pd.Series(np.random.randn(N) * 0.01, index=dates)}
    mkt    = pd.Series(np.random.randn(N) * 0.01, index=dates)
    vix    = pd.Series(np.abs(np.random.randn(N)) + 15.0, index=dates)
    y2     = pd.Series(np.full(N, 2.5), index=dates)
    y10    = pd.Series(np.full(N, 3.5), index=dates)
    cfg = {
        "data":     {"start_date": "2000-01-03"},
        "training": {"min_train_years": 8, "max_train_years": 12,
                     "refit_freq": "M", "test_start": "2009-01-01"},
        "sjm":      {"n_components": 2, "jump_penalty": 50.0, "max_feats": 9.5,
                     "max_iter": 5, "n_init_jm": 2, "random_state": 42},
    }
    result = run_regime_detection(active, mkt, vix, y2, y10, cfg)
    labels = result["value"]
    assert (labels.index >= pd.Timestamp("2009-01-01")).all(), \
        "Labels exist before test_start"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_regime.py::test_run_regime_detection_respects_test_start -v
```

Expected: FAIL — function doesn't accept `test_start` yet (also slow — uses actual SJM, may time out; just verify the TypeError).

- [ ] **Step 3: Update `run_regime_detection` signature and test_start logic**

In `src/regime.py`, replace the function from its `def` line through the line `test_dates = all_dates[all_dates >= test_start]`. Everything from `if len(test_dates) == 0:` onward is **unchanged**. The new version of those first ~20 lines:

```python
def run_regime_detection(
    active_rets: dict,
    mkt_ret: pd.Series,
    vix: pd.Series,
    y2: pd.Series,
    y10: pd.Series,
    cfg: dict,
    macro_extras: dict = None,
    enabled: list = None,
) -> dict:
    """
    Monthly expanding-window SJM refit + online inference for each factor.

    Returns dict[factor -> pd.Series of regime labels (0=bull, 1=bear, [2=neutral]).
    Labels use a 1-day delay: label on day T reflects inference from day T-1.
    """
    data_start = pd.Timestamp(cfg["data"]["start_date"])
    min_years  = cfg["training"]["min_train_years"]
    max_years  = cfg["training"]["max_train_years"]

    # Pass macro_extras through to feature engineering
    features_dict = compute_all_features(
        active_rets, mkt_ret, vix, y2, y10,
        macro_extras=macro_extras or {},
        enabled=enabled or [],
    )

    all_dates = features_dict[next(iter(features_dict))].dropna(how="all").index

    # Use explicit test_start from config if present; else derive from min_train_years
    if "test_start" in cfg.get("training", {}):
        test_start = pd.Timestamp(cfg["training"]["test_start"])
    else:
        test_start = data_start + pd.DateOffset(years=min_years)

    test_dates = all_dates[all_dates >= test_start]
    # ── everything below this line is IDENTICAL to the original ──────────────
```

- [ ] **Step 4: Run full regime test suite**

```bash
pytest tests/test_regime.py -v
```

Expected: all tests pass (the new test may be slow — it runs actual SJM; skip it with `-k "not respects_test_start"` if needed during development).

- [ ] **Step 5: Commit**

```bash
git add src/regime.py tests/test_regime.py
git commit -m "feat(regime): use test_start from config, forward macro_extras to features"
```

---

## Task 5: 3-state portfolio view returns in src/portfolio.py

**Files:**
- Modify: `src/portfolio.py`
- Modify: `tests/test_portfolio.py`

- [ ] **Step 1: Write failing tests**

Add to the bottom of `tests/test_portfolio.py`:

```python
from src.utils import FACTORS

def _make_label_history(dates, labels_list):
    """labels_list: list of int labels, same length as dates."""
    return pd.Series(labels_list, index=dates)

def test_compute_view_returns_2state_bear_is_negative():
    """In 2-state mode, bear regime (label=1) gives negative view."""
    N = 252 * 3
    dates = pd.date_range("2000-01-03", periods=N, freq="B")
    # factor returns: positive on first half (bull), negative on second half
    ret = pd.Series(
        [0.005] * (N // 2) + [-0.005] * (N - N // 2), index=dates
    )
    labels = pd.Series(
        [0] * (N // 2) + [1] * (N - N // 2), index=dates
    )
    active_ret_history = {f: ret for f in FACTORS}
    in_sample_labels   = {f: labels for f in FACTORS}
    regime_labels_now  = {f: 1 for f in FACTORS}  # current: bear

    q = compute_view_returns(regime_labels_now, active_ret_history, in_sample_labels)
    assert (q < 0).all(), "Bear regime views should be negative"

def test_compute_view_returns_3state_neutral_is_zero():
    """In 3-state mode, neutral regime (label=1) gives q=0."""
    N = 252 * 3
    dates = pd.date_range("2000-01-03", periods=N, freq="B")
    ret = pd.Series(np.random.randn(N) * 0.01, index=dates)
    # Label 0=bull, 1=neutral, 2=bear
    labels = pd.Series([0] * 84 + [1] * 84 + [2] * (N - 168), index=dates)
    active_ret_history = {f: ret for f in FACTORS}
    in_sample_labels   = {f: labels for f in FACTORS}
    regime_labels_now  = {f: 1 for f in FACTORS}  # current: neutral

    q = compute_view_returns(regime_labels_now, active_ret_history,
                              in_sample_labels, n_components=3)
    assert (q == 0.0).all(), "Neutral regime (label=1) views must be exactly 0"

def test_compute_view_returns_3state_bear_label2_is_negative():
    """In 3-state mode, bear regime (label=2) gives negative view."""
    N = 252 * 3
    dates = pd.date_range("2000-01-03", periods=N, freq="B")
    ret = pd.Series([-0.005] * N, index=dates)  # always negative
    labels = pd.Series([2] * N, index=dates)    # always bear
    active_ret_history = {f: ret for f in FACTORS}
    in_sample_labels   = {f: labels for f in FACTORS}
    regime_labels_now  = {f: 2 for f in FACTORS}  # current: bear

    q = compute_view_returns(regime_labels_now, active_ret_history,
                              in_sample_labels, n_components=3)
    assert (q < 0).all(), "Bear (label=2) views should be negative in 3-state"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_portfolio.py::test_compute_view_returns_2state_bear_is_negative tests/test_portfolio.py::test_compute_view_returns_3state_neutral_is_zero tests/test_portfolio.py::test_compute_view_returns_3state_bear_label2_is_negative -v
```

Expected: FAIL — `compute_view_returns` doesn't accept `n_components` yet.

- [ ] **Step 3: Update `compute_view_returns` in src/portfolio.py**

Replace the existing `compute_view_returns` function:

```python
def compute_view_returns(
    regime_labels: dict,
    active_ret_history: dict,
    in_sample_labels: dict,
    n_components: int = 2,
) -> np.ndarray:
    """
    q[k] = mean daily active return of factor k during its current regime
           computed over the training period.
    Capped at ±5%/252 (daily equivalent of ±5% annualized).

    3-state convention (n_components=3):
      label 0 = bull  → positive view
      label 1 = neutral → q[k] = 0 (no view, posterior reverts to prior)
      label 2 = bear  → negative view
    """
    cap = 0.05 / TRADING_DAYS
    q = np.zeros(5)
    for i, factor in enumerate(FACTORS):
        current_regime = regime_labels.get(factor, 0)

        # Neutral state in 3-state mode → no view
        if n_components == 3 and current_regime == 1:
            q[i] = 0.0
            continue

        ret_hist    = active_ret_history[factor]
        labels_hist = in_sample_labels[factor]
        aligned     = labels_hist.reindex(ret_hist.index).dropna()
        ret_aligned = ret_hist.reindex(aligned.index)
        mask        = aligned == current_regime
        if mask.sum() > 0:
            q[i] = float(ret_aligned[mask].mean())
        q[i] = float(np.clip(q[i], -cap, cap))
    return q
```

Also update `compute_portfolio_weights` to pass `n_components` through to `compute_view_returns`. Find the call to `compute_view_returns` inside `compute_portfolio_weights` and update it:

```python
    n_components = cfg.get("sjm", {}).get("n_components", 2)
    q = compute_view_returns(today_regime, active_ret_history, in_sample_labels,
                             n_components=n_components)
```

- [ ] **Step 4: Run new tests**

```bash
pytest tests/test_portfolio.py::test_compute_view_returns_2state_bear_is_negative tests/test_portfolio.py::test_compute_view_returns_3state_neutral_is_zero tests/test_portfolio.py::test_compute_view_returns_3state_bear_label2_is_negative -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run full portfolio test suite**

```bash
pytest tests/test_portfolio.py -v
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/portfolio.py tests/test_portfolio.py
git commit -m "feat(portfolio): add n_components param to compute_view_returns for 3-state support"
```

---

## Task 6: Add run_walk_forward to src/backtest.py

**Files:**
- Modify: `src/backtest.py`
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Write a failing test**

Add to the bottom of `tests/test_backtest.py`:

```python
from unittest.mock import patch

def test_run_walk_forward_returns_dataframe_with_expected_columns():
    from src.backtest import run_walk_forward
    import numpy as np

    np.random.seed(0)
    N = 252 * 5
    d = pd.date_range("2000-01-03", periods=N, freq="B")
    total = pd.DataFrame(np.random.randn(N, 6) * 0.01, index=d,
                         columns=["market","value","size","quality","growth","momentum"])
    active = pd.DataFrame(np.random.randn(N, 5) * 0.005, index=d,
                          columns=["value","size","quality","growth","momentum"])
    rf_ser = pd.Series(np.full(N, 0.0001), index=d)
    macro  = {
        "vix": pd.Series(np.abs(np.random.randn(N)) + 15, index=d),
        "y2":  pd.Series(np.full(N, 2.5), index=d),
        "y10": pd.Series(np.full(N, 3.5), index=d),
    }
    data = {"total_returns": total, "active_returns": active, "rf": rf_ser, "macro": macro}
    cfg = {
        "data":     {"start_date": "2000-01-03", "end_date": "2004-12-31"},
        "training": {"min_train_years": 1, "max_train_years": 3,
                     "refit_freq": "M", "test_start": "2001-01-01"},
        "sjm":      {"n_components": 2, "jump_penalty": 50.0, "max_feats": 9.5,
                     "max_iter": 5, "n_init_jm": 2, "random_state": 42},
        "black_litterman": {"risk_aversion": 2.5, "cov_halflife": 63, "tau": 0.05,
                            "target_tracking_error": 0.03, "transaction_cost_bps": 5,
                            "benchmark_rebalance_freq": "Q"},
        "macro_features": {"enabled": []},
        "walk_forward":   {"enabled": True, "n_folds": 2, "fold_test_months": 12},
    }

    # Mock the expensive pipeline steps
    test_start = pd.Timestamp("2001-01-01")
    test_dates = d[d >= test_start]
    mock_labels = {f: pd.Series(np.zeros(len(test_dates), dtype=int), index=test_dates)
                   for f in ["value","size","quality","growth","momentum"]}
    mock_weights = pd.DataFrame(
        np.ones((len(test_dates), 6)) / 6, index=test_dates,
        columns=["market","value","size","quality","growth","momentum"]
    )
    mock_port_ret = pd.Series(np.random.randn(len(test_dates)) * 0.01, index=test_dates)

    with patch("src.backtest.run_regime_detection", return_value=mock_labels), \
         patch("src.backtest.run_portfolio_construction", return_value=mock_weights), \
         patch("src.backtest.compute_portfolio_returns", return_value=mock_port_ret):
        result = run_walk_forward(data, cfg)

    assert isinstance(result, pd.DataFrame)
    required_cols = {"fold", "period_start", "period_end", "sharpe",
                     "ir_vs_market", "max_drawdown", "active_ret_vs_market",
                     "volatility", "turnover"}
    assert required_cols.issubset(set(result.columns))
    # Last row is the average
    assert result.iloc[-1]["fold"] == "avg"
    # Non-avg rows count <= n_folds
    data_rows = result[result["fold"] != "avg"]
    assert len(data_rows) <= cfg["walk_forward"]["n_folds"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_backtest.py::test_run_walk_forward_returns_dataframe_with_expected_columns -v
```

Expected: FAIL — `run_walk_forward` doesn't exist yet.

- [ ] **Step 3: Add imports and run_walk_forward to src/backtest.py**

Add these imports at the top of `src/backtest.py` (after the existing imports):

```python
from src.regime import run_regime_detection
from src.portfolio import run_portfolio_construction
```

Then add the following function at the end of `src/backtest.py`:

```python
def run_walk_forward(data: dict, cfg: dict) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation.

    For each fold: train on data_start → fold_test_start (expanding),
    test on fold_test_start → fold_test_end.
    Returns a DataFrame with one row per completed fold plus an average row.
    """
    total_ret  = data["total_returns"]
    active_ret = data["active_returns"]
    rf         = data["rf"]
    macro      = data["macro"]

    enabled      = cfg.get("macro_features", {}).get("enabled", [])
    macro_extras = {k: macro[k] for k in enabled if k in macro}
    factors      = list(active_ret.columns)

    test_start       = pd.Timestamp(cfg["training"]["test_start"])
    n_folds          = cfg["walk_forward"]["n_folds"]
    fold_test_months = cfg["walk_forward"]["fold_test_months"]
    cost_bps         = cfg["black_litterman"]["transaction_cost_bps"]

    rows = []
    for fold in range(1, n_folds + 1):
        fold_test_start = test_start + pd.DateOffset(months=(fold - 1) * fold_test_months)
        fold_test_end   = fold_test_start + pd.DateOffset(months=fold_test_months)

        mask = total_ret.index < fold_test_end
        if mask.sum() == 0:
            break

        fold_total  = total_ret[mask]
        fold_active = active_ret[mask]
        fold_rf     = rf[mask]
        fold_macro  = {k: v[mask] for k, v in macro.items() if isinstance(v, pd.Series)}
        fold_extras = {k: fold_macro[k] for k in enabled if k in fold_macro}

        fold_cfg = {
            **cfg,
            "training": {**cfg["training"], "test_start": str(fold_test_start.date())},
            "data":     {**cfg["data"], "end_date": str(fold_test_end.date())},
        }

        print(f"[{fold}/{n_folds}] walk-forward fold {fold} "
              f"({fold_test_start.date()} – {fold_test_end.date()})")

        try:
            regime_labels = run_regime_detection(
                {f: fold_active[f] for f in factors},
                mkt_ret=fold_total["market"],
                vix=fold_macro["vix"],
                y2=fold_macro["y2"],
                y10=fold_macro["y10"],
                cfg=fold_cfg,
                macro_extras=fold_extras,
                enabled=enabled,
            )

            test_label_dates = {
                f: regime_labels[f].index >= fold_test_start for f in factors
            }
            test_labels = {f: regime_labels[f][test_label_dates[f]] for f in factors}
            in_sample_labels = {
                f: test_labels[f].shift(-1).dropna().astype(int) for f in factors
            }

            weights = run_portfolio_construction(
                test_labels,
                in_sample_labels,
                total_returns=fold_total,
                active_returns=fold_active,
                cfg=fold_cfg,
            )

            test_dates    = weights.index
            port_ret      = compute_portfolio_returns(
                weights, fold_total.reindex(test_dates), cost_bps=cost_bps
            )
            mkt_ret_fold  = fold_total["market"].reindex(test_dates)
            ew_ret_fold   = compute_ew_returns(fold_total.reindex(test_dates))
            rf_fold       = fold_rf.reindex(test_dates)

            metrics = compute_performance_table(
                port_ret, mkt_ret_fold, ew_ret_fold, rf_fold, weights
            )
            rows.append({
                "fold":                 fold,
                "period_start":         str(fold_test_start.date()),
                "period_end":           str(fold_test_end.date()),
                "sharpe":               metrics["sharpe"],
                "ir_vs_market":         metrics["ir_vs_market"],
                "max_drawdown":         metrics["max_drawdown"],
                "active_ret_vs_market": metrics["active_ret_vs_market"],
                "volatility":           metrics["volatility"],
                "turnover":             metrics["turnover"],
            })
        except Exception as exc:
            print(f"      Fold {fold} failed: {exc}")
            continue

    if not rows:
        return pd.DataFrame(columns=["fold","period_start","period_end","sharpe",
                                     "ir_vs_market","max_drawdown",
                                     "active_ret_vs_market","volatility","turnover"])

    df = pd.DataFrame(rows)
    numeric_cols = ["sharpe","ir_vs_market","max_drawdown","active_ret_vs_market",
                    "volatility","turnover"]
    avg_row = {"fold": "avg", "period_start": "", "period_end": ""}
    for col in numeric_cols:
        avg_row[col] = df[col].mean()
    return pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
```

- [ ] **Step 4: Run the new test**

```bash
pytest tests/test_backtest.py::test_run_walk_forward_returns_dataframe_with_expected_columns -v
```

Expected: PASS.

- [ ] **Step 5: Run full backtest test suite**

```bash
pytest tests/test_backtest.py -v
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/backtest.py tests/test_backtest.py
git commit -m "feat(backtest): add run_walk_forward() with expanding-window folds"
```

---

## Task 7: Update main.py — pass macro_extras and add Phase 2

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Update the regime detection call to pass macro_extras**

In `main.py`, find the `[3/5]` block and replace the `run_regime_detection` call:

```python
    print("[3/5] Running SJM regime detection (this may take several minutes)...")
    enabled_extras = cfg.get("macro_features", {}).get("enabled", [])
    macro_extras   = {k: macro[k] for k in enabled_extras if k in macro}
    regime_labels  = run_regime_detection(
        active_rets_dict,
        mkt_ret=total_ret["market"],
        vix=macro["vix"],
        y2=macro["y2"],
        y10=macro["y10"],
        cfg=cfg,
        macro_extras=macro_extras,
        enabled=enabled_extras,
    )
```

- [ ] **Step 2: Add Phase 2 WFE block after the existing print("Done. Results in outputs/")**

Add `run_walk_forward` to the existing backtest import block at the top of `main.py`:

```python
from src.backtest import (
    compute_portfolio_returns,
    compute_ew_returns,
    compute_performance_table,
    save_results,
    save_weights_csv,
    plot_cumulative_returns,
    plot_regime,
    plot_portfolio_weights,
    save_returns_csv,
    run_walk_forward,          # NEW
)
```

Then find `print("\nDone. Results in outputs/")` and add Phase 2 after the summary printout:

```python
    # ── Phase 2: Walk-Forward Evaluation (optional) ──────────────────────────
    if cfg.get("walk_forward", {}).get("enabled", False):
        import os
        n_folds = cfg["walk_forward"]["n_folds"]
        print(f"\n[1/{n_folds}] Starting walk-forward evaluation ({n_folds} folds)...")
        wfe_df = run_walk_forward(data, cfg)
        wfe_path = os.path.join(output_dir, "wfe_results.csv")
        wfe_df.to_csv(wfe_path, index=False)
        print(f"Walk-forward results saved to {wfe_path}")
        avg = wfe_df[wfe_df["fold"] == "avg"].iloc[0]
        print(f"  Avg Sharpe: {avg['sharpe']:.3f}  "
              f"Avg IR: {avg['ir_vs_market']:.3f}  "
              f"Avg Max DD: {avg['max_drawdown']*100:.1f}%")
```

- [ ] **Step 3: Verify main.py still runs without walk_forward enabled**

```bash
python main.py --config config.yaml 2>&1 | head -20
```

Expected: starts loading data with no import errors (first line: `====...`, `[1/5] Loading data...`). Cancel after confirming startup succeeds (Ctrl+C is fine).

- [ ] **Step 4: Commit**

```bash
git add main.py
git commit -m "feat(main): pass macro_extras to regime detection, add Phase 2 walk-forward"
```

---

## Task 8: WFE analytics functions in shiny_app/components/analytics.py

**Files:**
- Modify: `shiny_app/components/analytics.py`
- Modify: `tests/test_analytics.py`

- [ ] **Step 1: Write failing tests**

Check if `tests/test_analytics.py` exists; add the following tests (create the file if absent):

```python
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shiny_app.components.analytics import (
    load_wfe_results,
    wfe_metrics_html,
    wfe_folds_plot,
)
from pathlib import Path
import tempfile

_WFE_DATA = pd.DataFrame([
    {"fold": 1, "period_start": "2008-01-01", "period_end": "2011-01-01",
     "sharpe": 0.21, "ir_vs_market": -0.05, "max_drawdown": -0.58,
     "active_ret_vs_market": -0.012, "volatility": 0.18, "turnover": 0.4},
    {"fold": 2, "period_start": "2011-01-01", "period_end": "2014-01-01",
     "sharpe": 0.74, "ir_vs_market": 0.18, "max_drawdown": -0.22,
     "active_ret_vs_market": 0.021, "volatility": 0.14, "turnover": 0.3},
    {"fold": "avg", "period_start": "", "period_end": "",
     "sharpe": 0.475, "ir_vs_market": 0.065, "max_drawdown": -0.40,
     "active_ret_vs_market": 0.0045, "volatility": 0.16, "turnover": 0.35},
])


def test_load_wfe_results_returns_none_when_missing():
    result = load_wfe_results(Path("/nonexistent/path"))
    assert result is None


def test_load_wfe_results_returns_dataframe(tmp_path):
    _WFE_DATA.to_csv(tmp_path / "wfe_results.csv", index=False)
    result = load_wfe_results(tmp_path)
    assert isinstance(result, pd.DataFrame)
    assert "sharpe" in result.columns


def test_wfe_metrics_html_returns_string():
    html = wfe_metrics_html(_WFE_DATA)
    assert isinstance(html, str)
    assert "<table" in html
    assert "avg" in html.lower() or "Avg" in html


def test_wfe_metrics_html_empty_df():
    html = wfe_metrics_html(pd.DataFrame())
    assert "No walk-forward" in html


def test_wfe_folds_plot_returns_figure():
    import matplotlib.pyplot as plt
    fig = wfe_folds_plot(_WFE_DATA, metric="sharpe")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_wfe_folds_plot_ir_metric():
    import matplotlib.pyplot as plt
    fig = wfe_folds_plot(_WFE_DATA, metric="ir_vs_market")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_analytics.py::test_load_wfe_results_returns_none_when_missing tests/test_analytics.py::test_load_wfe_results_returns_dataframe tests/test_analytics.py::test_wfe_metrics_html_returns_string tests/test_analytics.py::test_wfe_folds_plot_returns_figure -v
```

Expected: FAIL — functions don't exist yet.

- [ ] **Step 3: Add the three new functions to the end of shiny_app/components/analytics.py**

```python
# ── Walk-Forward Evaluation ──────────────────────────────────────────────────

def load_wfe_results(output_dir: Path) -> pd.DataFrame | None:
    path = Path(output_dir) / "wfe_results.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def wfe_metrics_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<p class='text-muted'>No walk-forward results — enable Walk-Forward and run the model.</p>"

    numeric_cols = ["sharpe", "ir_vs_market", "max_drawdown", "active_ret_vs_market", "turnover"]
    display_cols = {
        "sharpe":               "Sharpe",
        "ir_vs_market":         "IR vs Mkt",
        "max_drawdown":         "Max DD",
        "active_ret_vs_market": "Active Ret",
        "turnover":             "Turnover",
    }
    color_fn = {
        "Sharpe":     lambda v: "#198754" if v >= 0.5 else ("#dc3545" if v < 0 else "inherit"),
        "IR vs Mkt":  lambda v: "#198754" if v > 0 else "#dc3545",
        "Max DD":     lambda v: "#dc3545",
        "Active Ret": lambda v: "#198754" if v > 0 else "#dc3545",
        "Turnover":   lambda v: "inherit",
    }
    fmt_fn = {
        "Sharpe":     lambda v: f"{v:.3f}",
        "IR vs Mkt":  lambda v: f"{v:.3f}",
        "Max DD":     lambda v: f"{v * 100:.1f}%",
        "Active Ret": lambda v: f"{v * 100:.2f}%",
        "Turnover":   lambda v: f"{v:.4f}",
    }

    fold_rows = df[df["fold"] != "avg"]
    avg_row   = df[df["fold"] == "avg"]

    header = ("<tr style='border-bottom:2px solid #dee2e6'>"
              "<th style='padding:5px 8px'>Fold</th>"
              "<th style='padding:5px 8px'>Period</th>")
    for col in numeric_cols:
        header += f"<th style='padding:5px 8px;text-align:right'>{display_cols[col]}</th>"
    header += "</tr>"

    body = ""
    for _, row in pd.concat([fold_rows, avg_row]).iterrows():
        is_avg = row["fold"] == "avg"
        bg     = "#fff8e1" if is_avg else "#f8f9fa"
        fw     = "700"     if is_avg else "normal"
        fold_label   = "Avg" if is_avg else int(row["fold"])
        period_label = "" if is_avg else f"{row['period_start'][:7]} – {row['period_end'][:7]}"
        row_html = (f"<tr style='background:{bg}'>"
                    f"<td style='padding:5px 8px;font-weight:{fw}'>{fold_label}</td>"
                    f"<td style='padding:5px 8px;color:#888;font-size:11px'>{period_label}</td>")
        for col in numeric_cols:
            disp  = display_cols[col]
            val   = row.get(col, float("nan"))
            try:
                fval  = float(val)
                color = color_fn[disp](fval)
                txt   = fmt_fn[disp](fval)
            except (TypeError, ValueError):
                color, txt = "inherit", "—"
            row_html += (f"<td style='padding:5px 8px;text-align:right;"
                         f"color:{color};font-weight:{fw}'>{txt}</td>")
        row_html += "</tr>"
        body += row_html

    return (f"<table class='table table-sm' style='max-width:750px'>"
            f"<thead>{header}</thead><tbody>{body}</tbody></table>")


def wfe_folds_plot(df: pd.DataFrame, metric: str = "sharpe") -> plt.Figure:
    """Bar chart of a metric across walk-forward folds. Avg shown as dashed line."""
    fold_rows = df[df["fold"] != "avg"].copy()
    avg_rows  = df[df["fold"] == "avg"]

    if fold_rows.empty:
        return _blank_fig("No fold data")

    fold_rows["fold"] = fold_rows["fold"].astype(str)
    values = fold_rows[metric].astype(float).values
    labels = [f"F{r}" for r in fold_rows["fold"]]
    colors = ["#198754" if v >= 0 else "#dc3545" for v in values]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)

    if not avg_rows.empty:
        avg_val = float(avg_rows.iloc[0][metric])
        ax.axhline(avg_val, color="#7c3aed", linewidth=1.5, linestyle="--",
                   label=f"Avg {avg_val:.3f}")
        ax.legend(fontsize=9)

    display = {"sharpe": "Sharpe Ratio", "ir_vs_market": "IR vs Market",
               "max_drawdown": "Max Drawdown", "active_ret_vs_market": "Active Return",
               "volatility": "Volatility", "turnover": "Turnover"}
    ax.set_ylabel(display.get(metric, metric))
    ax.set_title(f"{display.get(metric, metric)} by Walk-Forward Fold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
```

- [ ] **Step 4: Run the analytics tests**

```bash
pytest tests/test_analytics.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add shiny_app/components/analytics.py tests/test_analytics.py
git commit -m "feat(analytics): add load_wfe_results, wfe_metrics_html, wfe_folds_plot"
```

---

## Task 9: Rewrite shiny_app/modules/model1.py — accordion sidebar + sub-tabs

**Files:**
- Modify: `shiny_app/modules/model1.py`

This task replaces model1.py in full. Read the current file before editing.

- [ ] **Step 1: Add module-level constants after the existing imports**

After the `_FACTORS` and `_STEP_RE` definitions, add:

```python
_MACRO_KEYS = [
    "vix_level", "dxy", "oil_ret", "gold_ret",
    "real_yield", "unemployment", "consumer_sent",
]
_MACRO_LABELS = {
    "vix_level":     "VIX Level (absolute)",
    "dxy":           "Dollar Index (DXY)",
    "oil_ret":       "Oil Returns (WTI)",
    "gold_ret":      "Gold Returns",
    "real_yield":    "10Y Real Yield",
    "unemployment":  "Unemployment Rate",
    "consumer_sent": "Consumer Sentiment",
}
```

- [ ] **Step 2: Update `_load_param_defaults` to read new config fields**

Replace the existing `_load_param_defaults` function:

```python
def _load_param_defaults(project_root: Path) -> dict:
    cfg_path = project_root / "config.yaml"
    if not cfg_path.exists():
        return {
            "jump_penalty": 50.0, "max_feats": 9.5,
            "risk_aversion": 2.5, "txn_cost": 5,
            "data_start": "2000-01-01", "data_end": "2026-01-30",
            "test_start": "2008-01-01", "n_components": "2",
            "macro_enabled": _MACRO_KEYS,
            "wfe_enabled": False, "n_folds": 6, "fold_test_months": "36",
        }
    with open(cfg_path) as f:
        cfg = _yaml.safe_load(f)
    return {
        "jump_penalty":    cfg["sjm"]["jump_penalty"],
        "max_feats":       cfg["sjm"]["max_feats"],
        "risk_aversion":   cfg["black_litterman"]["risk_aversion"],
        "txn_cost":        cfg["black_litterman"]["transaction_cost_bps"],
        "data_start":      cfg["data"].get("start_date", "2000-01-01"),
        "data_end":        cfg["data"].get("end_date", "2026-01-30"),
        "test_start":      cfg["training"].get("test_start", "2008-01-01"),
        "n_components":    str(cfg["sjm"].get("n_components", 2)),
        "macro_enabled":   cfg.get("macro_features", {}).get("enabled", _MACRO_KEYS),
        "wfe_enabled":     cfg.get("walk_forward", {}).get("enabled", False),
        "n_folds":         cfg.get("walk_forward", {}).get("n_folds", 6),
        "fold_test_months": str(cfg.get("walk_forward", {}).get("fold_test_months", 36)),
    }
```

- [ ] **Step 3: Replace the sidebar in `model_tab_ui`**

Replace the entire `sidebar` variable definition inside `model_tab_ui` with:

```python
    sidebar = ui.sidebar(
        ui.HTML(_SIDEBAR_CSS),
        # Anchor navigation
        ui.div(
            ui.tags.b("Sections"),
            ui.div(
                ui.tags.a("Metrics",           href="#metrics"),
                ui.tags.a("Cumulative Returns", href="#returns"),
                ui.tags.a("Rolling Sharpe",    href="#rolling-sharpe"),
                ui.tags.a("Drawdown",          href="#drawdown"),
                ui.tags.a("Realized TE",       href="#realized-te"),
                ui.tags.a("Portfolio Weights", href="#weights"),
                ui.tags.a("Regime Plots",      href="#regimes"),
                class_="anchor-links",
            ),
            class_="mb-2",
        ),
        ui.hr(),
        # Accordion
        ui.accordion(
            # ── Section 1: Run Configuration ──────────────────────────────
            ui.accordion_panel(
                "⚙ Run Configuration",
                ui.input_date("data_start", "Data Start",
                              value=defaults["data_start"]),
                ui.input_date("data_end",   "Data End",
                              value=defaults["data_end"]),
                ui.input_date("test_start", "Test Period Start",
                              value=defaults["test_start"]),
                ui.input_select("te", "TE Target", choices=te_choices,
                                selected=default_te),
                ui.div(ui.tags.b("Regime States"), class_="mt-2 mb-1"),
                ui.input_radio_buttons(
                    "n_components", label=None,
                    choices={"2": "2-state", "3": "3-state"},
                    selected=defaults["n_components"],
                    inline=True,
                ),
                value="run_config",
            ),
            # ── Section 2: Model Parameters ───────────────────────────────
            ui.accordion_panel(
                "⚙ Model Parameters",
                ui.input_numeric("jump_penalty", "Jump Penalty (λ)",
                                 value=defaults["jump_penalty"], step=1, min=1),
                ui.input_numeric("max_feats",    "Feature Sparsity (κ²)",
                                 value=defaults["max_feats"], step=0.5, min=0.5),
                ui.input_numeric("risk_aversion","Risk Aversion (δ)",
                                 value=defaults["risk_aversion"], step=0.1, min=0.1),
                ui.input_numeric("txn_cost",     "Txn Cost (bps)",
                                 value=defaults["txn_cost"], step=1, min=0),
                value="model_params",
            ),
            # ── Section 3: Macro Features ─────────────────────────────────
            ui.accordion_panel(
                "📊 Macro Features",
                *[
                    ui.input_checkbox(
                        k, _MACRO_LABELS[k],
                        value=(k in defaults["macro_enabled"]),
                    )
                    for k in _MACRO_KEYS
                ],
                value="macro_features",
            ),
            # ── Section 4: Walk-Forward ───────────────────────────────────
            ui.accordion_panel(
                "🔁 Walk-Forward",
                ui.input_checkbox("wfe_enabled", "Enable Walk-Forward",
                                  value=defaults["wfe_enabled"]),
                ui.output_ui("wfe_fold_controls"),
                value="walk_forward",
            ),
            id="sidebar_accordion",
            open="run_config",
            multiple=False,
        ),
        ui.hr(),
        ui.input_action_button(
            "rerun", "Run Model",
            class_="btn-primary btn-sm w-100 mt-2",
            disabled=cfg.get("run_command") is None,
        ),
        ui.output_ui("run_progress"),
        ui.output_ui("run_status"),
        width=230,
        class_="model-sidebar",
    )
```

- [ ] **Step 4: Replace the `main` content area in `model_tab_ui` to use inner tabs**

Replace the `main` variable definition:

```python
    results_content = ui.div(
        section("Performance Metrics",     "metrics",       ui.output_ui("metrics_tbl")),
        section("Cumulative Returns",      "returns",       ui.output_ui("returns_plot")),
        section("Rolling Sharpe",          "rolling-sharpe",ui.output_ui("rolling_sharpe_plot")),
        section("Drawdown",                "drawdown",      ui.output_ui("drawdown_plot")),
        section("Realized Tracking Error", "realized-te",   ui.output_ui("realized_te_plot")),
        section("Portfolio Weights",       "weights",       ui.output_ui("weights_plot")),
        section("Regime Plots",            "regimes",
                ui.div(*[
                    ui.div(
                        ui.h6(f.capitalize(), class_="text-muted"),
                        ui.output_ui(f"regime_{f}_plot"),
                        class_="mb-3",
                    )
                    for f in _FACTORS
                ])),
        style="padding:1rem",
    )

    validation_content = ui.div(
        section("Walk-Forward Results", "wfe-results",    ui.output_ui("wfe_metrics_tbl")),
        section("Sharpe by Fold",       "wfe-sharpe",     ui.output_ui("wfe_sharpe_plot")),
        section("IR vs Market by Fold", "wfe-ir",         ui.output_ui("wfe_ir_plot")),
        style="padding:1rem",
    )

    main = ui.navset_tab(
        ui.nav_panel("Results",    results_content),
        ui.nav_panel("Validation", validation_content),
    )
```

- [ ] **Step 5: Add `wfe_fold_controls` renderer and validation tab renderers to `model_tab_server`**

Inside `model_tab_server`, after the existing `run_status` renderer, add:

```python
    @render.ui
    def wfe_fold_controls():
        wfe_on = input.wfe_enabled()
        style  = "" if wfe_on else "opacity:0.4;pointer-events:none"
        return ui.div(
            ui.input_slider("n_folds", "Folds", min=3, max=8, value=6, step=1),
            ui.input_select(
                "fold_test_months", "Test Window",
                choices={"12": "12 months", "24": "24 months", "36": "36 months"},
                selected="36",
            ),
            style=style,
        )

    @render.ui
    def wfe_metrics_tbl():
        from shiny_app.components.analytics import load_wfe_results, wfe_metrics_html
        df = load_wfe_results(output_dir)
        if df is None:
            if input.wfe_enabled():
                return ui.p("Run the model to generate walk-forward results.",
                            class_="text-muted")
            return ui.p("Enable Walk-Forward in the sidebar and run the model.",
                        class_="text-muted")
        return ui.HTML(wfe_metrics_html(df))

    @render.ui
    def wfe_sharpe_plot():
        from shiny_app.components.analytics import load_wfe_results, wfe_folds_plot
        df = load_wfe_results(output_dir)
        if df is None:
            return ui.div()
        return fig_to_img(wfe_folds_plot(df, metric="sharpe"), alt="Sharpe by fold")

    @render.ui
    def wfe_ir_plot():
        from shiny_app.components.analytics import load_wfe_results, wfe_folds_plot
        df = load_wfe_results(output_dir)
        if df is None:
            return ui.div()
        return fig_to_img(wfe_folds_plot(df, metric="ir_vs_market"), alt="IR by fold")
```

- [ ] **Step 6: Update `_launch_pipeline` to write all new config fields**

Inside the `_launch_pipeline` async function, replace the section that writes `cfg_data` before `yaml.dump`:

```python
        cfg_data["data"]["start_date"]                          = str(input.data_start())
        cfg_data["data"]["end_date"]                            = str(input.data_end())
        cfg_data["training"]["test_start"]                      = str(input.test_start())
        cfg_data["sjm"]["n_components"]                         = int(input.n_components())
        cfg_data["sjm"]["jump_penalty"]                         = float(input.jump_penalty())
        cfg_data["sjm"]["max_feats"]                            = float(input.max_feats())
        cfg_data["black_litterman"]["risk_aversion"]            = float(input.risk_aversion())
        cfg_data["black_litterman"]["transaction_cost_bps"]     = int(input.txn_cost())
        cfg_data.setdefault("macro_features", {})["enabled"]   = [
            k for k in _MACRO_KEYS if getattr(input, k)()
        ]
        wfe_on = bool(input.wfe_enabled())
        cfg_data.setdefault("walk_forward", {})["enabled"]     = wfe_on
        try:
            cfg_data["walk_forward"]["n_folds"]          = int(input.n_folds())
            cfg_data["walk_forward"]["fold_test_months"] = int(input.fold_test_months())
        except Exception:
            cfg_data["walk_forward"]["n_folds"]          = 6
            cfg_data["walk_forward"]["fold_test_months"] = 36
        cfg_data["black_litterman"]["target_tracking_error"]    = int(input.te()) / 100
```

- [ ] **Step 7: Start the dashboard and verify it loads without errors**

```bash
python -m shiny run shiny_app/app.py --reload
```

Open `http://127.0.0.1:8000` and confirm:
- Model 1 tab shows the accordion sidebar with 4 collapsible sections
- "Run Configuration" is open by default with date pickers, TE select, and 2/3-state radio
- "Macro Features" section shows 7 checkboxes
- "Walk-Forward" section shows Enable checkbox; fold controls appear greyed-out when unchecked
- Inner tabs show "Results" and "Validation"
- "Validation" tab shows the placeholder message (not an error)

- [ ] **Step 8: Commit**

```bash
git add shiny_app/modules/model1.py
git commit -m "feat(dashboard): accordion sidebar, Results/Validation tabs, WFE display"
```

---

## Task 10: Run full test suite and verify end-to-end

**Files:** none (verification only)

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v --tb=short
```

Expected: all tests pass with no failures.

- [ ] **Step 2: Quick smoke test of the pipeline with WFE disabled**

```bash
python main.py --config config.yaml 2>&1 | tail -10
```

Expected: completes Phase 1, prints summary table, no walk-forward output (since `walk_forward.enabled: false` in config.yaml).

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "test: verify all SJM enhancement tests pass"
```

---

## Quick Reference — Key Signatures

```python
# src/data.py
load_macro(start: str, end: str, enabled: list = None) -> dict

# src/features.py
compute_market_features(mkt_ret, vix, y2, y10, macro_extras: dict = None, enabled: list = None) -> pd.DataFrame
compute_all_features(active_rets, mkt_ret, vix, y2, y10, macro_extras: dict = None, enabled: list = None) -> dict

# src/regime.py
run_regime_detection(active_rets, mkt_ret, vix, y2, y10, cfg, macro_extras: dict = None, enabled: list = None) -> dict

# src/portfolio.py
compute_view_returns(regime_labels, active_ret_history, in_sample_labels, n_components: int = 2) -> np.ndarray

# src/backtest.py
run_walk_forward(data: dict, cfg: dict) -> pd.DataFrame

# shiny_app/components/analytics.py
load_wfe_results(output_dir: Path) -> pd.DataFrame | None
wfe_metrics_html(df: pd.DataFrame) -> str
wfe_folds_plot(df: pd.DataFrame, metric: str = "sharpe") -> plt.Figure

# WFE output columns: fold, period_start, period_end, sharpe, ir_vs_market,
#                     max_drawdown, active_ret_vs_market, volatility, turnover
```

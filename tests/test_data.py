import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data import (
    _parse_ken_french_csv,
    _parse_fred_csv,
    build_asset_returns,
)

# --- Synthetic Ken French CSV string (comma-delimited, date YYYYMMDD) ---
FF5_CSV = """,Mkt-RF,SMB,HML,RMW,CMA,RF
20000103,0.10,0.05,-0.02,0.03,0.01,0.002
20000104,-0.05,-0.01,0.02,-0.01,0.00,0.002
20000105,0.08,0.02,0.01,0.02,-0.01,0.002
"""

MOM_CSV = """,Mom
20000103,0.12
20000104,-0.08
20000105,0.05
"""

FRED_CSV = """DATE,DGS2
2000-01-03,6.25
2000-01-04,6.28
2000-01-05,
2000-01-06,6.30
"""


def test_parse_ken_french_ff5():
    df = _parse_ken_french_csv(FF5_CSV, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
    assert list(df.columns) == ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
    assert len(df) == 3
    assert df.index[0] == pd.Timestamp('2000-01-03')
    # Values divided by 100
    assert df.loc[pd.Timestamp('2000-01-03'), 'Mkt-RF'] == pytest.approx(0.001)


def test_parse_ken_french_mom():
    df = _parse_ken_french_csv(MOM_CSV, ['Mom'])
    assert list(df.columns) == ['Mom']
    assert df.loc[pd.Timestamp('2000-01-03'), 'Mom'] == pytest.approx(0.0012)


def test_parse_fred_csv():
    s = _parse_fred_csv(FRED_CSV)
    assert isinstance(s, pd.Series)
    # Missing value on 2000-01-05 should be NaN
    assert pd.isna(s.loc[pd.Timestamp('2000-01-05')])
    assert s.loc[pd.Timestamp('2000-01-03')] == pytest.approx(6.25)


def test_build_asset_returns():
    ff5 = _parse_ken_french_csv(FF5_CSV, ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
    mom = _parse_ken_french_csv(MOM_CSV, ['Mom'])
    total, active = build_asset_returns(ff5, mom)

    assert list(total.columns) == ['market', 'value', 'size', 'quality', 'growth', 'momentum']
    assert list(active.columns) == ['value', 'size', 'quality', 'growth', 'momentum']

    # market total = Mkt-RF + RF
    assert total['market'].iloc[0] == pytest.approx(ff5['Mkt-RF'].iloc[0] + ff5['RF'].iloc[0])
    # value total = market + HML
    assert total['value'].iloc[0] == pytest.approx(total['market'].iloc[0] + ff5['HML'].iloc[0])
    # growth total = market - CMA
    assert total['growth'].iloc[0] == pytest.approx(total['market'].iloc[0] - ff5['CMA'].iloc[0])
    # active return = total - market
    assert active['value'].iloc[0] == pytest.approx(ff5['HML'].iloc[0])
    assert active['growth'].iloc[0] == pytest.approx(-ff5['CMA'].iloc[0])


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

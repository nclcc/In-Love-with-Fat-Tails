import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from src._analytics.paretian_analytics_ import _ParetoAnalytics
from src._data_handler.financial_data_handler_ import _FinancialDataHandler


def analytics(values, tmp_path):
    return _ParetoAnalytics({'lag_1': {'positive': np.asarray(values), 'negative': np.asarray(values)}},
                            _t_horizon=1, _lags=np.array([1]), _path=str(tmp_path))


def test_empirical_survival_handles_ties_and_preserves_index():
    result = _ParetoAnalytics._get_survival_probability(pd.Series([3., 1., 3., 2.], index=[9, 2, 5, 7]))
    assert result.index.tolist() == [9, 2, 5, 7]
    assert_allclose(result.survival_probability, [.5, 1., .5, .75])


def test_fixed_threshold_pareto_mle_matches_formula(tmp_path):
    a = analytics([1., 2., 4., 8.], tmp_path)
    fitted = a.fit_pareto(xmin=2.)
    assert_allclose(fitted['alpha'], 3 / np.log([1., 2., 4.]).sum())
    assert fitted['n_tail'] == 3


def test_pareto_recovery_on_seeded_synthetic_data(tmp_path):
    rng = np.random.default_rng(15)
    a = analytics(2 * (1 + rng.pareto(2.5, 20000)), tmp_path)
    assert abs(a.fit_pareto(xmin=2)['alpha'] - 2.5) < .08


def test_infinite_conditional_mean_is_not_reported_as_finite(tmp_path):
    a = analytics([1., 10., 100., 1000.], tmp_path)
    assert np.isinf(a.fit_pareto(xmin=1.)['conditional_mean'])


def test_loglog_diagnostic_uses_log_magnitudes(tmp_path):
    # Deterministic Pareto quantiles give an exactly linear empirical log survival.
    n = 100
    sample = (n / np.arange(n, 0, -1)) ** .5
    a = analytics(sample, tmp_path)
    fits = a._visual_paretianity(tail=1.)
    assert_allclose([fit.c[0] for fit in fits], [-2., -2.], atol=1e-10)
    assert (tmp_path / 'PowerLawsInTails.png').exists()


def test_maximum_sum_includes_every_prefix_and_requested_moment():
    actual = _ParetoAnalytics._maximum_sum([1., 2., 3.], 2)
    assert_allclose(actual, [[1., 1.], [2/3, 4/5], [3/6, 9/14]])


def test_bootstrap_is_reproducible_and_returns_original_mle(tmp_path):
    a = analytics([1., 1.5, 2., 3., 5., 8., 13., 21.], tmp_path)
    estimate = a._bootstrap_moment_estimation(xmin=1., sample_size=30, simulation=100, random_state=4)
    first = a.bootstrap_summary['bootstrap_alpha'].copy()
    assert estimate == a.fit_pareto(xmin=1.)['alpha']
    a._bootstrap_moment_estimation(xmin=1., sample_size=30, simulation=100, random_state=4)
    assert_allclose(first, a.bootstrap_summary['bootstrap_alpha'])


def test_return_horizons_preserve_real_observations_without_padding(monkeypatch):
    raw = pd.DataFrame({'Close': [1.01, 1.03, .99, 1.04, .98]})
    monkeypatch.setattr('src._data_handler.financial_data_handler_.yf.download', lambda *a, **kw: raw)
    handler = _FinancialDataHandler('X', _lags=np.array([1, 3, 5]))
    result = handler._process_returns()
    assert set(result) == {'lag_1', 'lag_3'}
    returns = np.array([1.03, .99, 1.04, .98]) / np.array([1.01, 1.03, .99, 1.04]) - 1
    assert_allclose(result['lag_1']['positive'], returns[returns > 0])
    assert_allclose(result['lag_1']['negative'], -returns[returns < 0])
    assert_allclose(result['lag_1']['absolute'], abs(returns))


def test_kurtosis_uses_actual_nonconsecutive_lag_keys(tmp_path):
    from scipy.stats import kurtosis
    a = analytics([1, 2, 3, 8], tmp_path)
    a._data['lag_5'] = {'positive': np.array([1, 2, 4, 20])}
    a._lags = np.array([1, 5])
    assert_allclose(a._convergence_kurtosis(), [kurtosis([1,2,3,8], fisher=False), kurtosis([1,2,4,20], fisher=False)])


@pytest.mark.parametrize('values', [[0., 1.], [-1., 2.], [np.nan, 2.], []])
def test_invalid_tail_samples_are_rejected(values, tmp_path):
    with pytest.raises(ValueError):
        analytics(values, tmp_path).fit_pareto(xmin=1.)

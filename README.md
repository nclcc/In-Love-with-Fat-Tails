# In Love with Fat Tails

Exploratory diagnostics for financial-return magnitudes, with a fixed-threshold continuous Pareto estimator.

## Install and check

Use Python 3.11 or 3.12 from the repository root:

```sh
python -m pip install -r requirements.txt
MPLBACKEND=Agg python -m pytest -q
```

Tests are offline. They check survival probabilities with ties, return preprocessing, a hand-computed Pareto likelihood estimate, recovery from a seeded synthetic Pareto sample, infinite moments, log-log plot coordinates, maximum-to-sum prefixes, bootstrap reproducibility, and nonconsecutive horizons. GitHub Actions repeats them on both supported Python versions.

## Fixed-threshold estimation

```python
import numpy as np
from src._analytics.paretian_analytics_ import _ParetoAnalytics

rng = np.random.default_rng(42)
magnitudes = 2 * (1 + rng.pareto(2.5, 5000))
analysis = _ParetoAnalytics(
    {'lag_1': {'positive': magnitudes, 'negative': magnitudes}},
    _t_horizon=1, _lags=np.array([1]), _path='results'
)
print(analysis.fit_pareto(xmin=2))
analysis._visual_paretianity(tail=.33)
analysis._bootstrap_moment_estimation(xmin=2, simulation=1000, random_state=42)
```

For magnitudes x at or above a fixed positive xmin:

```
P(X >= x | X >= xmin) = (xmin / x)**alpha
alpha_MLE = n_tail / sum(log(x / xmin))
conditional_tail_mean = alpha * xmin / (alpha - 1), if alpha > 1
```

Here alpha is the **survival exponent**; the density exponent is alpha+1. The mean is reported as infinite when alpha<=1. It is a conditional tail mean, not an estimate of the unconditional mean of all returns. `fit_pareto` returns alpha, xmin, tail size and conditional mean. If xmin is omitted, `tail` selects a quantile threshold; this is a heuristic, not a fitted optimal cutoff. The MLE replaces the previous softplus-based statistic, which was not a Pareto shape estimator.

`_bootstrap_moment_estimation` returns the original-sample MLE and exposes bootstrap estimates and a percentile interval in `bootstrap_summary`. The threshold remains fixed during resampling. This IID bootstrap is descriptive for independent observations; it does not provide calibrated intervals for dependent or overlapping financial returns. Undefined bootstrap MLEs and infinite conditional means are counted explicitly rather than silently converted to finite values.

## Diagnostics and preprocessing

- `_get_survival_probability` computes empirical P(X>=x), preserving order/index and giving ties equal probabilities.
- `_visual_paretianity` plots log survival against log magnitude. Its returned callable lines accept log magnitudes; they are descriptive fits, not likelihood estimates or proof of a power law.
- `_MS_plots` includes every prefix and moments 1 through the requested order, using positive magnitudes and scale-invariant arithmetic.
- `_convergence_kurtosis` retains its legacy name but plots sample Pearson kurtosis against actual return horizons. Variation across horizons does not prove that a population fourth moment diverges.
- `_FinancialDataHandler` uses unrounded adjusted Close prices and simple horizon returns. Negative tails contain absolute loss magnitudes; `absolute` contains absolute returns. Valid price pairs are retained without zero padding or truncation after splitting signs. Horizons without enough price observations are omitted.

A straight log-log line alone cannot establish a Pareto law. Threshold sensitivity, goodness-of-fit tests and comparisons against alternatives are not implemented here. See [Clauset, Shalizi and Newman](https://arxiv.org/abs/0706.1062) for a fuller inference procedure; match the exponent convention when comparing formulas. The original `tests/_test_paretian_analytics.py` remains a manual online-data example, separate from the offline regression suite. Historical figures and numerical outputs must be regenerated after these corrections.

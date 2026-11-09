"""Exploratory tail diagnostics. Plots are not tests of a power-law hypothesis."""
from dataclasses import dataclass, field
from datetime import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis


@dataclass
class _ParetoAnalytics:
    _data: dict
    _start: dt = field(default_factory=lambda: dt(1950, 1, 1))
    _end: dt = field(default_factory=dt.today)
    _lags: np.ndarray = field(default_factory=lambda: np.arange(1, 554))
    _t_horizon: int = 66
    _path: str = '.'

    @staticmethod
    def _positive_values(values):
        values = np.asarray(values, dtype=float).reshape(-1)
        if not len(values) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('Tail observations must be nonempty, finite positive magnitudes')
        return values

    @staticmethod
    def _get_survival_probability(arr):
        """Empirical P(X >= x), with equal values assigned equal probabilities."""
        values = pd.Series(arr, dtype=float)
        if values.empty or not np.isfinite(values.to_numpy()).all():
            raise ValueError('Observations must be nonempty and finite')
        rank = values.rank(method='min')
        return pd.DataFrame({'input_values': values,
                             'survival_probability': (len(values) - rank + 1) / len(values)}, index=values.index)

    def _save(self, fig, filename):
        destination = Path(self._path)
        destination.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(destination / filename, dpi=150)
        plt.close(fig)

    def _tail(self, type_returns='positive', xmin=None, tail=.33):
        if not 0 < tail <= 1:
            raise ValueError('tail must be in (0, 1]')
        values = self._positive_values(self._data[f'lag_{self._t_horizon}'][type_returns])
        xmin = float(np.quantile(values, 1 - tail)) if xmin is None else float(xmin)
        if not np.isfinite(xmin) or xmin <= 0:
            raise ValueError('xmin must be finite and positive')
        sample = values[values >= xmin]
        if sample.size < 2 or np.unique(sample).size < 2:
            raise ValueError('At least two distinct tail observations are required')
        return sample, xmin

    def fit_pareto(self, type_returns='positive', xmin=None, tail=.33):
        """Continuous Pareto MLE at a fixed threshold.

        alpha is the SURVIVAL exponent: P(X >= x | X >= xmin)=(xmin/x)^alpha.
        The density exponent used in some papers is alpha+1. No automatic
        threshold selection, goodness-of-fit claim or independence test is made.
        """
        sample, xmin = self._tail(type_returns, xmin, tail)
        alpha = sample.size / np.log(sample / xmin).sum()
        return {'alpha': float(alpha), 'xmin': xmin, 'n_tail': int(sample.size),
                'conditional_mean': float(alpha * xmin / (alpha - 1)) if alpha > 1 else np.inf}

    def _visual_paretianity(self, tail=.33):
        """Plot log survival against log magnitude, returning two callable lines.

        Lines are descriptive least-squares fits in log-log space, not MLEs.
        """
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        fits = []
        for side, ax in zip(('positive', 'negative'), axes):
            sample, _ = self._tail(side, tail=tail)
            empirical = self._get_survival_probability(sample).sort_values('input_values')
            x = np.log(empirical.input_values.to_numpy())
            y = np.log(empirical.survival_probability.to_numpy())
            fit = np.poly1d(np.polyfit(x, y, 1))
            fits.append(fit)
            ax.scatter(x, y, s=8, label='Empirical tail survival')
            ax.plot(x, fit(x), label='Descriptive log-log fit')
            ax.set(xlabel=f'log({side} return magnitude)', ylabel='log survival probability', title=f'{side.title()} tail diagnostic')
            ax.legend()
        self._save(fig, 'PowerLawsInTails.png')
        return tuple(fits)

    @staticmethod
    def _maximum_sum(values, moments):
        values = _ParetoAnalytics._positive_values(values)
        if not isinstance(moments, (int, np.integer)) or moments < 1:
            raise ValueError('moments must be a positive integer')
        # Rescaling leaves each ratio unchanged and prevents large-value overflow.
        scaled = values / values.max()
        powers = scaled[:, None] ** np.arange(1, moments + 1)[None, :]
        cumulative = np.cumsum(powers, axis=0)
        if np.any(cumulative == 0):
            raise ValueError('Moment powers underflowed; reduce the moment order')
        return np.maximum.accumulate(powers, axis=0) / cumulative

    def _MS_plots(self, moments=5):
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        results = []
        for side, ax in zip(('positive', 'negative'), axes):
            ratios = self._maximum_sum(self._data[f'lag_{self._t_horizon}'][side], moments)
            results.append(ratios)
            for j in range(moments):
                ax.plot(np.arange(1, len(ratios) + 1), ratios[:, j], label=f'Moment {j+1}')
            ax.set(xlabel='Number of observations', ylabel='Maximum / sum', title=f'{side.title()} magnitudes')
            ax.legend()
        self._save(fig, 'MaximumSumPlots.png')
        return tuple(results)

    def _convergence_kurtosis(self, type_returns='positive'):
        """Describe kurtosis across return horizons; this does not prove divergence."""
        horizons, values = [], []
        for lag in self._lags:
            key = f'lag_{int(lag)}'
            if key not in self._data:
                continue
            sample = self._positive_values(self._data[key][type_returns])
            if len(sample) < 4 or np.ptp(sample) == 0:
                continue
            horizons.append(int(lag))
            values.append(float(kurtosis(sample, fisher=False)))
        if not values:
            raise ValueError('No horizons with four nonconstant observations')
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(horizons, values)
        ax.set(xlabel='Return horizon (observations)', ylabel='Pearson kurtosis', title='Sample kurtosis by return horizon')
        self._save(fig, 'EmpiricalKurtosis.png')
        return values

    def _bootstrap_moment_estimation(self, type_returns='positive', sample_size=500,
                                     simulation=10000, xmin=None, tail=.33, random_state=None):
        """IID nonparametric bootstrap of fixed-xmin Pareto survival alpha.

        Returns the original-sample MLE; diagnostics are in bootstrap_summary.
        The conditional tail mean exists only for alpha > 1. For serially
        dependent/overlapping returns, an IID bootstrap is not calibrated.
        """
        for name, value in [('sample_size', sample_size), ('simulation', simulation)]:
            if not isinstance(value, (int, np.integer)) or value < 2:
                raise ValueError(f'{name} must be an integer of at least two')
        sample, xmin = self._tail(type_returns, xmin, tail)
        fitted = self.fit_pareto(type_returns, xmin=xmin)
        rng = np.random.default_rng(random_state)
        log_ratios = np.log(sample / xmin)
        estimates = np.empty(simulation)
        for start in range(0, simulation, 256):
            count = min(256, simulation - start)
            denominators = rng.choice(log_ratios, size=(count, sample_size), replace=True).sum(axis=1)
            estimates[start:start + count] = np.divide(sample_size, denominators,
                                                       out=np.full(count, np.inf), where=denominators > 0)
        self.bootstrap_summary = {**fitted, 'bootstrap_alpha': estimates,
                                  'alpha_interval_95': np.quantile(estimates, [.025, .975], method='inverted_cdf'),
                                  'undefined_mle_count': int(np.isinf(estimates).sum())}
        fig, axes = plt.subplots(2, 1, figsize=(8, 8))
        finite_alpha = estimates[np.isfinite(estimates)]
        if finite_alpha.size:
            axes[0].hist(finite_alpha, bins=min(50, simulation))
        axes[0].axvline(fitted['alpha'], color='orange', label='Original-sample MLE')
        axes[0].set(xlabel='Survival exponent alpha', title='IID bootstrap at a fixed threshold')
        axes[0].legend()
        valid_mean = np.isfinite(estimates) & (estimates > 1)
        means = estimates[valid_mean] * xmin / (estimates[valid_mean] - 1)
        if means.size:
            axes[1].hist(means, bins=min(50, simulation))
        axes[1].set(xlabel='Conditional mean above xmin',
                    title=f'{np.sum(estimates <= 1)} replicates have infinite tail mean; '
                          f'{np.sum(~np.isfinite(estimates))} have undefined MLE')
        self._save(fig, 'Bootstrap_alpha_estimation.png')
        return fitted['alpha']

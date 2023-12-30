import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime as dt
from copy import copy
from scipy.stats import kurtosis
from scipy.interpolate import UnivariateSpline
from dataclasses import dataclass


@dataclass
class _ParetoAnalytics:
    _data: dict
    _start: dt.date = dt(1950, 1, 1)
    _end: dt.date = dt.today()
    _lags: np.arange = np.arange(1, 554)
    _t_horizon: int = 66
    _path = str = '.'

    @staticmethod
    def _get_survival_probability(
            arr: np.array
    ):
        """
        This function calculates the empirical survival probability for each value in the input array.

        Parameters
        ----------
        arr : array_like
            Numeric values on the real number line.

        Returns
        -------
        survival_probability_sr : Pandas Series
            A Pandas Series containing the survival probabilities for each input value.
        """
        # Sort values from low to high.
        arr = copy(arr)  # Copy to avoid accidental mutation
        sr = pd.Series(arr)  # Ensure we have a pandas series

        # Keep a copy of the original index
        input_index = sr.index.copy()

        # Create index of input order
        df = sr.reset_index(name='input_values')  # Keeps the input index as a column
        df.index.name = 'input_order'  # Name the new index

        # Sort from low to high and reindex
        df = df.sort_values(by='input_values')  # sort from low to high
        df = df.reset_index()
        df.index.name = 'sorted_order'  # Name the new index

        # Label relative positions
        gap_count = len(sr) + 1  # Think of the Posts and Fences analogy
        df['left_gap_count'] = df.index + 1  # Count values <= x
        df['right_gap_count'] = gap_count - df.left_gap_count  # Count values >= x

        # Get survival Probability
        df['survival_probability'] = df.right_gap_count / gap_count

        # Reset Input Order and Index
        df = df.sort_values(by='input_order')  # sort from low to high
        df.index = input_index

        return df

    def _visual_paretianity(self, tail: float = 0.33) -> tuple:
        """
        This function creates graphics for visual paretianity of tails

        Parameters
        ----------
        tail, float: share of the total number of observation to be considered as a tail

        Returns
        -------
        spline_1, spline_2: tuple(UnivariateSpline, UnivariateSpline)
        """

        # Compute the ECDF for positive
        df = self._get_survival_probability(self._data[f'lag_{self._t_horizon}']['positive'])
        df = df[df.survival_probability < tail]
        SF_positive = df.sort_values(by='survival_probability', ascending=False)[['survival_probability']]
        X_positive = df.sort_values(by='survival_probability', ascending=False)[['input_values']]

        # Compute the ECDF for negative
        df = self._get_survival_probability(self._data[f'lag_{self._t_horizon}']['negative'])
        df = df[df.survival_probability < tail]
        SF_negative = df.sort_values(by='survival_probability', ascending=False)[['survival_probability']]
        X_negative = df.sort_values(by='survival_probability', ascending=False)[['input_values']]

        # Fit a least squares regression line (polynomial of degree 1)
        spline_1 = UnivariateSpline(X_positive, np.log(SF_positive), k=1)

        # Fit a least squares regression line (polynomial of degree 1)
        spline_2 = UnivariateSpline(X_negative, np.log(SF_negative), k=1)

        # Plot the ECDF
        figure(figsize=(15, 14), dpi=800)

        plt.subplot(211)
        plt.scatter(X_positive, np.log(SF_positive), marker='o', color='orange')
        plt.plot(X_positive, spline_1(X_positive), linestyle='--', color='g',
                 label='Positive Power Law')
        plt.xlabel('Positive returns')
        plt.ylabel('Survival function')
        plt.title('Empirical Cumulative Distribution Function (ECDF) for Positive Returns')

        plt.subplot(212)
        plt.scatter(X_negative, np.log(SF_negative), marker='o', color='blue')
        plt.plot(X_negative, spline_2(X_negative), linestyle='--', color='g',
                 label='Negative Power Law')
        plt.xlabel('Negative returns')
        plt.ylabel('Survival Function')
        plt.title('Empirical Cumulative Distribution Function (ECDF) for Negative Returns')
        plt.tight_layout()
        plt.savefig(f"{self._path}//PowerLawsInTails.png")

        return spline_1, spline_2

    def _MS_plots(self, moments: int = 5):
        """
        This function returns Maximum-to-Sum plots of random variables

        Parameters
        ----------
        moments: int,
            No. of moments of the distribution to plot

        Returns
        ----------
        MS_index: np.array,
            Array with Maximum-to-Sum values
        """

        # Initialize picture
        figure(figsize=(15, 14), dpi=800)

        # Plot Maximum Sum plots for positive returns
        plt.subplot(411)
        MS_index = np.zeros([self._data[f'lag_{self._t_horizon}']['positive'].shape[0], moments])

        for j in np.arange(1, moments):
            for i in np.arange(self._t_horizon, self._data[f'lag_{self._t_horizon}']['positive'].shape[0]):
                MS_index[i, j] = max(self._data[f'lag_{self._t_horizon}']['positive'][:i] ** j) / sum(
                    self._data[f'lag_{self._t_horizon}']['positive'][:i] ** j)

            plt.plot(MS_index[self._t_horizon - 1:, j], label=f'Moment {j}-th')

        plt.xlabel('Number of Obs')
        plt.ylabel('Max/Sum')
        plt.title('Maximum/Sum ratio for Positive Returns')
        plt.legend()

        # Plot Maximum Sum plots for negative returns
        plt.subplot(412)
        MS_index_ = np.zeros([self._data[f'lag_{self._t_horizon}']['negative'].shape[0], moments])

        for j in np.arange(1, moments):
            for i in np.arange(self._t_horizon, self._data[f'lag_{self._t_horizon}']['negative'].shape[0]):
                MS_index_[i, j] = max(self._data[f'lag_{self._t_horizon}']['negative'][:i] ** j) / sum(
                    self._data[f'lag_{self._t_horizon}']['negative'][:i] ** j)

            plt.plot(MS_index_[self._t_horizon - 1:, j], label=f'Moment {j}-th')

        plt.xlabel('Number of Obs')
        plt.ylabel('Max/Sum')
        plt.title('Maximum/Sum ratio for negative Returns')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self._path}//MaximumSumPlots.png", format='png')

        return MS_index, MS_index_

    def _convergence_kurtosis(
            self,
            type_returns: str = 'positive'
    ):
        """
        This function proves the absence of a convergence of kurtosis

        Parameters
        ----------
        type_returns: str = positive
            string with positive

        Returns
        -------
        lagged_kurtosis: list,
            value for the lagged kurtosis
        """

        # Compute the kurtosis at each lagged set of the dataset
        lagged_kurtosis = []
        for i, lag_i in enumerate(self._lags):
            if i == 0:
                lagged_kurtosis.append(kurtosis(self._data[f'lag_{i + 1}'][f'{type_returns}'], fisher=False))
            else:
                lagged_kurtosis.append(kurtosis(self._data[f'lag_{i}'][f'{type_returns}'], fisher=False))

        # Lagged Kurtosis
        Kr = np.array(lagged_kurtosis)

        # Plot the ECDF
        figure(figsize=(10, 11), dpi=500)
        plt.subplot(311)
        plt.plot(np.arange(Kr.shape[0]), Kr, linestyle='dashdot', color='g', label='Positive Power Law')
        plt.xlabel('Lags')
        plt.ylabel('Kurtosis')
        plt.ylim([0, 20])
        plt.title('Empirical Kurtosis for Returns')
        plt.savefig(f"{self._path}//EmpiricalKurtosis.png")

        return lagged_kurtosis

    def _bootstrap_moment_estimation(self,
                                     type_returns: str = 'positive',
                                     sample_size: int = 500,
                                     simulation: int = 10000
                                     ) -> float:
        """
        This function proves the absence of a convergence of kurtosis

        Parameters
        ----------
        type_returns: str = positive
            string with positive

        sample_size: int = 500
            sample_size for the bootstrap estimation

        simulation: int = 10000
            no. of simulations

        Returns
        -------
        lagged_kurtosis: list,
            value for the lagged kurtosis
        """

        # Original vector
        original_vector = self._data[f'lag_{self._t_horizon}'][f'{type_returns}']

        # Simulate with replacement
        bootstrap_sample = np.random.choice(original_vector, size=[sample_size, simulation], replace=True)

        # Bootstrap statistic
        a_stat = np.zeros(bootstrap_sample.shape[1])
        for i in range(bootstrap_sample.shape[1]):
            a_stat[i] = (bootstrap_sample.shape[0]) / (np.log(1 + np.exp(bootstrap_sample[:, i])).sum())

        # Plot the ECDF
        figure(figsize=(10, 11), dpi=800)
        plt.subplot(311)
        plt.hist(a_stat, bins=200, color='blue', alpha=0.7)
        plt.xlabel('Tail component')
        plt.ylabel('Frequency')
        plt.axvline(x=pd.Series(a_stat).describe()[1], linewidth=2, color='y')  # mean
        plt.title('Bootstrap alpha tail for returns distribution')

        plt.subplot(312)
        plt.hist(a_stat / (a_stat - 1), bins=200, color='orange', alpha=0.7)
        plt.xlabel('Tail component')
        plt.ylabel('Frequency')
        plt.title('Bootstrap unconditional mean for returns distribution')
        plt.axvline(x=pd.Series(a_stat / (a_stat - 1)).describe()[1], linewidth=2, color='b')  # mean
        plt.tight_layout()
        plt.savefig(f"{self._path}//Bootstrap_alpha_estimation.png")

        return a_stat.mean()

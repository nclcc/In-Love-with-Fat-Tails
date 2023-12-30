import yfinance as yf
import numpy as np
from datetime import datetime as dt
from dataclasses import dataclass


@dataclass
class _FinancialDataHandler:
    _ticker: str
    _start: dt.date = dt(1950, 1, 1)
    _end: dt.date = dt.today()
    _lags: np.arange = np.arange(1, 554)
    _t_horizon: int = 66

    def __post_init__(self):
        """
        This function runs the hard-core init operations

        :return: None
        """

        # Call the yfinance package
        _shares = yf.download(self._ticker, start=self._start, end=self._end)

        # Round the shares
        self._closed_shares = np.round(_shares['Close'])

        # Save the shares in numpy format
        self._closed_shares = self._closed_shares.to_numpy()

    def _process_returns(self):
        """
        This function process the returns and creates a wrapper dictionary

        :return: DataJSON, dict: log-returns with positive, negative and absolute value in np.array format
        """

        # Final dictionary
        self.DataJSON = {}

        # Define an empty object
        lagged_St = np.zeros([self._closed_shares.shape[0], self._lags.shape[0]])

        # Create Lags
        for i, lag_i in enumerate(self._lags):
            lagged_St[:-lag_i, i] = ((self._closed_shares[lag_i:] / self._closed_shares[:-lag_i]) - 1)

        # Create final dictionary
        for i in range(1, lagged_St.shape[1]):
            self.DataJSON[f'lag_{i}'] = {'positive': abs(lagged_St[:, i - 1][lagged_St[:, i - 1] >= 0][:-i]),
                                         'negative': abs(lagged_St[:, i - 1][lagged_St[:, i - 1] < 0][:-i]),
                                         'absolute': lagged_St[:, i - 1][:-i]}

        return self.DataJSON

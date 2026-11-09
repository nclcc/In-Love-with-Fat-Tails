import yfinance as yf
import numpy as np
from datetime import datetime as dt
from dataclasses import dataclass, field


@dataclass
class _FinancialDataHandler:
    _ticker: str
    _start: dt.date = dt(1950, 1, 1)
    _end: dt.date = field(default_factory=dt.today)
    _lags: np.ndarray = field(default_factory=lambda: np.arange(1, 554))
    _t_horizon: int = 66

    def __post_init__(self):
        """
        This function runs the hard-core init operations

        :return: None
        """

        # Call the yfinance package
        _shares = yf.download(self._ticker, start=self._start, end=self._end, auto_adjust=True)

        # Round the shares
        self._closed_shares = _shares['Close']

        # Save the shares in numpy format
        self._closed_shares = self._closed_shares.to_numpy(dtype=float).reshape(-1)
        if (self._closed_shares.size < 2 or not np.isfinite(self._closed_shares).all()
                or np.any(self._closed_shares <= 0)):
            raise ValueError("Prices must contain at least two finite positive observations")

    def _process_returns(self):
        """Simple horizon returns; negative/absolute tails are positive magnitudes.

        Only valid price pairs are retained. No zero padding, price rounding or
        side-specific truncation is applied. Keys reflect the actual lag values.
        """
        prices = np.asarray(self._closed_shares, dtype=float).reshape(-1)
        self.DataJSON = {}
        for lag in self._lags:
            if not isinstance(lag, (int, np.integer)) or lag < 1:
                raise ValueError("Return horizons must be positive integers")
            if lag >= len(prices):
                continue
            returns = prices[lag:] / prices[:-lag] - 1
            self.DataJSON[f'lag_{int(lag)}'] = {
                'positive': returns[returns > 0],
                'negative': -returns[returns < 0],
                'absolute': np.abs(returns)
            }
        return self.DataJSON

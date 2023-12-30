from src._data_handler.financial_data_handler_ import _FinancialDataHandler
from src._analytics.paretian_analytics_ import _ParetoAnalytics


if __name__ == "__main__":

    # Financial Data Handler
    FinancialData = _FinancialDataHandler(_ticker="SI=F")

    # Process Data and create returns
    FinancialData._process_returns()

    # Paretian Analytics
    Analysis = _ParetoAnalytics(_data=FinancialData.DataJSON)

    # Visual Paretianity
    Analysis._visual_paretianity()

    # MS Plots
    Analysis._MS_plots()

    # Convergence Kurtosis
    Analysis._convergence_kurtosis()

    # Bootstrap Moment Estimation
    Analysis._bootstrap_moment_estimation()


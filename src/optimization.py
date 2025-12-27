import numpy as np
from scipy.optimize import minimize
from src.utils import get_alpha, get_portfolio_volatility

def objective_maximize_alpha(weights, stock_returns, benchmark_returns):
    return -get_alpha(weights, stock_returns, benchmark_returns)

def optimize_single_period(stock_returns, benchmark_returns, max_volatility=0.30):
    n_stocks = stock_returns.shape[1]
    cov_matrix = np.cov(stock_returns, rowvar=False)

    initial_weights = np.array([1/n_stocks] * n_stocks)
    bounds = tuple((0, 1) for _ in range(n_stocks))

    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: max_volatility - get_portfolio_volatility(x, cov_matrix)}
    )

    result = minimize(
        objective_maximize_alpha,
        initial_weights,
        args=(stock_returns, benchmark_returns),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-6
    )

    if result.success:
        return result.x
    return None
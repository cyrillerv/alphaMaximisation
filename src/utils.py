import numpy as np

def get_portfolio_returns(weights, stock_returns):
    return np.dot(stock_returns, weights)

def get_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

def get_alpha(weights, stock_returns, benchmark_returns):
    port_returns = np.dot(stock_returns, weights)
    
    cov_matrix = np.cov(port_returns, benchmark_returns)
    covariance = cov_matrix[0, 1]
    variance_bench = cov_matrix[1, 1]
    
    if variance_bench == 0:
        return -np.inf
        
    beta = covariance / variance_bench
    
    mean_ret_port = np.mean(port_returns) * 252
    mean_ret_bench = np.mean(benchmark_returns) * 252
    
    alpha = mean_ret_port - (beta * mean_ret_bench)
    return alpha
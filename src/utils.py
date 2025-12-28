import numpy as np
import pandas as pd
import os

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


def format_financial_report(raw_metrics):
    """Function to format the backtesting metrics df"""

    format_mapping = {
        'sharpe_ratio': '{:.2f}',
        'sortino_ratio': '{:.2f}',
        'calmar_ratio': '{:.2f}',
        'max_cash_needed': '{:,.0f}',
        'total_profit': '{:,.0f}',
        'total_return': '{:.2%}',
        'annualized_return': '{:.2%}',
        'volatility': '{:.2%}',
        'max_drawdown': '{:.2%}',
        'nb_transacs_total': '{:.0f}',
        'Hit_ratio': '{:.1%}',           
        'Winner_median': '{:.2%}',       
        'Loser_median': '{:.2%}',        
        'Median_holding_period': '{:.0f} jours', 
        'alpha_annualized': '{:.2%}',
        'r_squared': '{:.3f}',
        'n_observations': '{:.0f}'
    }

    label_mapping = {
        'sharpe_ratio': 'Sharpe Ratio',
        'max_drawdown': 'Max Drawdown',
        'annualized_return': 'Annual Return',
        'alpha_annualized': 'Alpha (Ann.)',
        'Hit_ratio': 'Win Rate',         
        'total_profit': 'Net Profit ($)',
        'volatility': 'Volatility',
        'r_squared': 'R-Squared',
        'Winner_median': 'Avg Winner',   
        'Loser_median': 'Avg Loser',    
        'Median_holding_period': 'Holding Period' 
    }
    
    categories = {
        'Performance': ['total_profit', 'annualized_return', 'alpha_annualized', 'total_return'],
        'Risk Profile': ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'volatility', 'r_squared'],
        'Trade Stats': ['nb_transacs_total', 'Hit_ratio', 'Winner_median', 'Loser_median', 'Median_holding_period']
    }

    formatted_data = []
    processed_metrics = set()
    
    for category, metrics in categories.items():
        for metric in metrics:
            if metric in raw_metrics:
                raw_val = raw_metrics[metric]
                fmt = format_mapping.get(metric, '{:.2f}')
                
                try:
                    clean_val = fmt.format(raw_val)
                except ValueError:
                    clean_val = str(raw_val)

                clean_label = label_mapping.get(metric, metric.replace('_', ' ').title())
                formatted_data.append({'Category': category, 'Metric': clean_label, 'Value': clean_val})
                processed_metrics.add(metric)

    for metric in raw_metrics.index:
        if metric not in processed_metrics:
            raw_val = raw_metrics[metric]
            clean_label = metric.replace('_', ' ').title()
            
            if isinstance(raw_val, (int, float)):
                 clean_val = '{:.4f}'.format(raw_val)
            else:
                 clean_val = str(raw_val)

            formatted_data.append({'Category': 'Autres / Détails', 'Metric': clean_label, 'Value': clean_val})

    df_clean = pd.DataFrame(formatted_data)
    return df_clean.set_index(['Category', 'Metric'])


def save_single_graph(fig, name, folder_name="results"):
    os.makedirs(folder_name, exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    img_path = os.path.join(folder_name, f"{safe_name}.png")
    fig.write_image(img_path, width=1200, height=600, scale=2)
import numpy as np
import pandas as pd
import os
import logging
import sys
from pathlib import Path
import plotly.express as px

def config_logger(log_filename="backtest.log"):
    log_path = Path(log_filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers[:]:
            root.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.WARNING, 
        handlers=[console_handler]
    )

    logger = logging.getLogger("MyStrategy")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    logger.addHandler(file_handler)

    return logger


def get_portfolio_returns(weights, stock_returns):
    return np.dot(stock_returns, weights)

def get_portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

def prepare_alpha_terms(stock_returns, benchmark_returns):
    """Function to do calc outside optimization function to save time."""
    R = stock_returns.to_numpy()          # (T, N)
    b = benchmark_returns.to_numpy()      # (T,)

    mu_R = R.mean(axis=0)                 # (N,)
    mu_b = b.mean()

    cov_Rb = ((R - mu_R) * (b - mu_b)[:, None]).mean(axis=0)  # (N,)
    var_b = np.var(b)

    return mu_R, mu_b, cov_Rb, var_b

def objective_maximize_alpha_fast(w, mu_R, mu_b, cov_Rb, var_b):
    port_mean = w @ mu_R
    beta = (w @ cov_Rb) / var_b
    alpha = port_mean - beta * mu_b
    return -alpha






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



def plot_calibration_results(sharpe_dict, robustness_range=None):
    df_res = pd.DataFrame(list(sharpe_dict.items()), columns=['Window', 'Sharpe'])
    df_res = df_res.sort_values('Window')

    fig = px.line(
        df_res, 
        x='Window', 
        y='Sharpe', 
        title='Sharpe Ratio Sensitivity vs Lookback Window',
        markers=True,
        labels={'Window': 'Window (Months)', 'Sharpe': 'Sharpe Ratio'}
    )

    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        template='plotly_white',
        hovermode='x unified'
    )

    if robustness_range:
        fig.add_vrect(
            x0=robustness_range[0], 
            x1=robustness_range[1], 
            fillcolor="green", opacity=0.15, 
            layer="below", line_width=0,
            annotation_text="Robustness Zone", 
            annotation_position="top left"
        )

    return fig


def obtenir_tickers_actifs(df, date_cible):
    date_cible = pd.to_datetime(date_cible)
    mask = (df['MbrStartDt'] <= date_cible) & (df['MbrEndDt'] >= date_cible)
    return df.loc[mask, 'PERMNO'].drop_duplicates().to_list()
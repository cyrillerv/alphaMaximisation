import pandas as pd
import numpy as np
from src.optimization import *

def compute_weights_at_rebal_date(df_stocks_ret, histo_compo_cac, logger, bench_returns, UNIT, rebal_date, WINDOW, MAX_VOL) :
    start_date_window = rebal_date - pd.DateOffset(**{UNIT: WINDOW})
    end_date_window = rebal_date    
    stock_ret_window = df_stocks_ret.loc[start_date_window:end_date_window].copy()

    try:
        # Retrieve the composition of the CAC40 at the begining of the optimisation period
        compo_at_date = histo_compo_cac.replace(0, np.nan).reindex([stock_ret_window.index.min()], method='ffill').T.dropna()
        valid_tickers = compo_at_date.index.intersection(df_stocks_ret.columns)
    except Exception:
        logger.debug(f"Skipping {rebal_date.date()}: Not enough valid tickers ({len(valid_tickers)})")
        return {}

    if len(valid_tickers) < 2:
        logger.debug(f"Skipping {rebal_date.date()}: Not enough valid tickers ({len(valid_tickers)})")
        return {}

    stock_ret_window = stock_ret_window[valid_tickers].dropna(axis=1)
    bench_ret_window = bench_returns.loc[start_date_window:end_date_window].copy()

    if stock_ret_window.empty:
        logger.warning(f"No data for window ending {rebal_date.date()}")
        return {}

    optimal_weights = optimize_single_period(
        stock_ret_window.values, 
        bench_ret_window.values, 
        max_volatility=MAX_VOL
    )

    if optimal_weights is not None:
        dic_weights = dict(zip(stock_ret_window.columns, optimal_weights))
        return dic_weights
    return {}
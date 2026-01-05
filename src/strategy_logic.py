import pandas as pd
import numpy as np
from src.optimization import *
from src.utils import obtenir_tickers_actifs

def compute_weights_at_rebal_date(stock_prices, histo_compo_cac, logger, bench_returns, UNIT, rebal_date, WINDOW, MAX_VOL) :
    start_date_window = rebal_date - pd.DateOffset(**{UNIT: WINDOW})
    end_date_window = rebal_date    

    try:
        # Retrieve the composition of the CAC40 at the begining of the optimisation period
        valid_tickers = obtenir_tickers_actifs(histo_compo_cac, rebal_date)
        valid_tickers.append('rf')
    except Exception:
        logger.debug(f"Skipping {rebal_date.date()}: Not enough valid tickers ({len(valid_tickers)})")
        return {}

    if len(valid_tickers) < 2:
        logger.debug(f"Skipping {rebal_date.date()}: Not enough valid tickers ({len(valid_tickers)})")
        return {}

    stock_ret_window_long = stock_prices.loc[
        (stock_prices['PERMNO'].isin(valid_tickers)) &
        (stock_prices['date'] >= start_date_window) &
        (stock_prices['date'] <= end_date_window)
    ]
    stock_ret_window = stock_ret_window_long.pivot_table(values="RET_calc", columns="PERMNO", index="date").fillna(0)
    bench_ret_window = bench_returns.loc[start_date_window:end_date_window].copy()

    if stock_ret_window.empty:
        logger.warning(f"No data for window ending {rebal_date.date()}")
        return {}

    optimal_weights = optimize_single_period_gpu(
        stock_ret_window, 
        bench_ret_window, 
        max_volatility=MAX_VOL
    )

    if optimal_weights is not None:
        dic_weights = dict(zip(stock_ret_window.columns, optimal_weights))
        return dic_weights
    return {}
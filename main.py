import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

from src.data import load_histo_cac, load_stock_prices
from src.optimization import optimize_single_period
from src.execution import generate_transaction_log
from src.utils import config_logger, format_financial_report, save_single_graph
from backtesting.core import BacktestEngine 

# TODO: continuer à clean le code
# TODO: passer les print en log
# TODO: dans le tableau comparatif mettre les data du benchmark
# TODO: mettre le formatting dans la librairie backtesting
# TODO: faire cross-validation pour ma strat
unit = "months"
WINDOW = 21 # in years
STEP = 1 # in years
MAX_VOL = 0.30
INITIAL_CAPITAL = 1_000_000
START_DATE = "2010-01-01"
BENCHMARK = "^FCHI"

logger = config_logger()
logger.info('Running strat')

if __name__ == "__main__":

    print("Step1: Data Loadings")
    histo_compo_cac = load_histo_cac()
    all_tickers = [BENCHMARK] + list(histo_compo_cac.columns)
    stock_returns, stock_prices = load_stock_prices(all_tickers)
    stock_returns.dropna(subset=['^FCHI'], inplace=True)

    # Checking whether the inputed start date has enough historical data to be the first rebal date
    min_required_start = stock_returns.index.min() + pd.DateOffset(**{unit: WINDOW})
    effective_start_date = max(pd.to_datetime(START_DATE), min_required_start)
    if effective_start_date > pd.to_datetime(START_DATE):
        print(f"Start date adjusted to {effective_start_date.date()} (insufficient history)")


    bench_returns = stock_returns['^FCHI'].copy()
    df_stocks_ret = stock_returns.drop(columns=['^FCHI']).copy()

    end_date = bench_returns.index.max()
    # Define a list of all rebalancing dates
    rebal_dates = list(pd.date_range(start=effective_start_date, end=end_date, freq=pd.DateOffset(**{unit: STEP})))

    dic_weights = {}
    for rebal_date in tqdm(rebal_dates) :
        start_date_window = rebal_date - pd.DateOffset(**{unit: WINDOW})
        end_date_window = rebal_date    
        stock_ret_window = df_stocks_ret.loc[start_date_window:end_date_window].copy()

        try:
            # Retrieve the composition of the CAC40 at the begining of the optimisation period
            compo_at_date = histo_compo_cac.replace(0, np.nan).reindex([stock_ret_window.index.min()], method='ffill').T.dropna()
            valid_tickers = compo_at_date.index.intersection(df_stocks_ret.columns)
        except Exception:
            logger.debug(f"Skipping {rebal_date.date()}: Not enough valid tickers ({len(valid_tickers)})")
            continue

        if len(valid_tickers) < 2:
            logger.debug(f"Skipping {rebal_date.date()}: Not enough valid tickers ({len(valid_tickers)})")
            continue

        stock_ret_window = stock_ret_window[valid_tickers].dropna(axis=1)
        bench_ret_window = bench_returns.loc[start_date_window:end_date_window].copy()

        if stock_ret_window.empty:
            logger.warning(f"No data for window ending {rebal_date.date()}")
            continue

        optimal_weights = optimize_single_period(
            stock_ret_window.values, 
            bench_ret_window.values, 
            max_volatility=MAX_VOL
        )

        if optimal_weights is not None:
            dic_weights[rebal_date] = dict(zip(stock_ret_window.columns, optimal_weights))


    print("Weights cleaning")
    df_weights = pd.DataFrame(dic_weights).T.sort_index()
    df_weights_filtered = df_weights.fillna(0).round(4)
    df_weights_filtered = df_weights_filtered.loc[(df_weights_filtered != 0).any(axis=1)]
    print(f"We generated weights for {len(df_weights_filtered)} rebalancing dates.")


    print("Generating transaction log...")
    df_transactions = generate_transaction_log(
        df_weights_filtered, 
        stock_prices, 
        initial_capital=INITIAL_CAPITAL
    )


    print("Starting Backtest")
    df_prices_bt = stock_prices[list(set(df_transactions['Symbol'])) + [BENCHMARK]]
    engine = BacktestEngine(
        df_transactions, 
        df_prices_bt, 
        bench_df_input=df_prices_bt['^FCHI'].to_frame(), 
        annual_discount_rate=0
    )
    engine.run()


    print("Saving results")
    metrics = pd.Series(engine.summary())
    df_report = format_financial_report(metrics)
    df_report.to_markdown(Path('results') / 'report.md', index=True)
    print(df_report)

    graphs_to_save = {
        "01_Cumulative_PnL": engine.cumulative_pnl_graph,
        "02_Drawdown": engine.drawdown_graph,
        "03_Returns_Dist": engine.returns_histogram,
        "04_Hit_Ratio": engine.hit_ratio_pie,
        "05_Volume_vs_Perf": engine.volume_vs_perf_scatter_plot
    }
    if not engine.bench_df.empty:
        graphs_to_save["06_Regression_Analysis"] = engine.fig_regression
    if engine.sector_mapping: 
        graphs_to_save["07_Sector_Exposure"] = engine.graph_sector_exposure
        graphs_to_save["08_Allocation_Long"] = engine.allocation_sector_long
        graphs_to_save["09_Allocation_Short"] = engine.allocation_sector_short

    for name, fig in graphs_to_save.items():
        save_single_graph(fig, name)
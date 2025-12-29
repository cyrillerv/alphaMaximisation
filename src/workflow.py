from tqdm import tqdm
import pandas as pd
from src.strategy_logic import *
from src.execution import generate_transaction_log
from backtesting.core import BacktestEngine 

def compute_rebal_date(WINDOW, stock_returns, bench_returns, START_DATE, UNIT, STEP, logger) :
    # TODO: mettre cette vérfi dans la qfonction car window dynamqieu
    # Checking whether the inputed start date has enough historical data to be the first rebal date
    min_required_start = stock_returns.index.min() + pd.DateOffset(**{UNIT: WINDOW})
    effective_start_date = max(pd.to_datetime(START_DATE), min_required_start)
    if effective_start_date > pd.to_datetime(START_DATE):
        logger.warning(f"Start date adjusted to {effective_start_date.date()} (insufficient history)")

    end_date = bench_returns.index.max()

    # Define a list of all rebalancing dates
    rebal_dates = list(pd.date_range(start=effective_start_date, end=end_date, freq=pd.DateOffset(**{UNIT: STEP})))

    return rebal_dates


def run_rolling_backtest(WINDOW_list, rebal_dates, stock_prices, df_stocks_ret, histo_compo_cac, logger, bench_returns, UNIT, INITIAL_CAPITAL, BENCHMARK, MAX_VOL) :

    dic_weights = {}
    for rebal_date in tqdm(rebal_dates) :
        dic_weights_window = {}
        for curr_window in WINDOW_list :
            dic_weights_window[curr_window] = compute_weights_at_rebal_date(df_stocks_ret, histo_compo_cac, logger, bench_returns, UNIT, rebal_date, curr_window, MAX_VOL)

        df_daily_weights = pd.DataFrame(dic_weights_window).fillna(0)
        dic_weights[rebal_date] = df_daily_weights.mean(axis=1).to_dict()

    logger.info("Weights cleaning")
    df_weights = pd.DataFrame(dic_weights).T.sort_index()
    df_weights_filtered = df_weights.fillna(0).round(4)
    df_weights_filtered = df_weights_filtered.loc[(df_weights_filtered != 0).any(axis=1)]
    logger.info(f"We generated weights for {len(df_weights_filtered)} rebalancing dates.")


    logger.info("Generating transaction log...")
    df_transactions = generate_transaction_log(
        df_weights_filtered, 
        stock_prices, 
        initial_capital=INITIAL_CAPITAL
    )


    logger.info("Starting Backtest")
    df_prices_bt = stock_prices[list(set(df_transactions['Symbol'])) + [BENCHMARK]]

    engine = BacktestEngine(
        df_transactions, 
        df_prices_bt, 
        bench_df_input=df_prices_bt[BENCHMARK].to_frame(), 
        annual_discount_rate=0
    )
    engine.run()
    return engine
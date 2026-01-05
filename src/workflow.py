from tqdm import tqdm
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.offsets import CBMonthEnd
from src.strategy_logic import *
from src.execution import generate_transaction_log
from backtesting.core import BacktestEngine 

def compute_rebal_date(WINDOW, stock_returns, bench_returns, START_DATE, UNIT, STEP, logger) :
    # TODO: mettre cette vérfi dans la qfonction car window dynamqieu
    # Checking whether the inputed start date has enough historical data to be the first rebal date
    min_required_start = stock_returns['date'].min() + pd.DateOffset(**{UNIT: WINDOW})
    effective_start_date = max(pd.to_datetime(START_DATE), min_required_start)
    if effective_start_date > pd.to_datetime(START_DATE):
        logger.warning(f"Start date adjusted to {effective_start_date.date()} (insufficient history)")

    end_date = bench_returns.index.max()

    # Define a list of all rebalancing dates
    # rebal_dates = list(pd.date_range(start=effective_start_date, end=end_date, freq=pd.DateOffset(**{UNIT: STEP})))
    # us_bday = CustomBusinessDay(calendar='NYSE')
    # Rebal à la fin du mois ouvré
    rebal_dates = list(
        pd.date_range(
            start=effective_start_date,
            end=end_date,
            freq=CBMonthEnd(calendar='NYSE')
        )
    )

    return rebal_dates


def run_rolling_backtest(WINDOW_list, rebal_dates, stock_prices, histo_compo_cac, logger, bench_returns, UNIT, INITIAL_CAPITAL, MAX_VOL, END_DATE_STRAT) :

    dic_weights = {}
    for rebal_date in tqdm(rebal_dates) :
        dic_weights_window = {}
        for curr_window in WINDOW_list :
            dic_weights_window[curr_window] = compute_weights_at_rebal_date(stock_prices, histo_compo_cac, logger, bench_returns, UNIT, rebal_date, curr_window, MAX_VOL)

        df_daily_weights = pd.DataFrame(dic_weights_window).fillna(0)
        dic_weights[rebal_date] = df_daily_weights.mean(axis=1).to_dict()

    logger.info("Weights cleaning")
    df_weights = pd.DataFrame(dic_weights).T.sort_index()
    df_weights_filtered = df_weights.fillna(0).round(4)
    df_weights_filtered = df_weights_filtered.loc[(df_weights_filtered != 0).any(axis=1)]
    logger.info(f"We generated weights for {len(df_weights_filtered)} rebalancing dates.")


    logger.info("Generating transaction log...")
    df_weights_filtered.to_csv(r'data\checkpoints\df_weights_filtered.csv')
    transaction_journal_bt = generate_transaction_log(
        df_weights_filtered, 
        stock_prices, 
        INITIAL_CAPITAL, 
        END_DATE_STRAT
        )
    transaction_journal_bt.to_csv(r"data\checkpoints\transaction_journal.csv")

    logger.info("Starting Backtest")
    tickers_traded = transaction_journal_bt['Symbol'].drop_duplicates().dropna().to_list()
    date_first_transac = transaction_journal_bt['Date'].min()
    date_last_transac = max(transaction_journal_bt['Date'].max(), transaction_journal_bt['ClosingDate'].max()) # TODO: mettre des vérif, plus de robustesse
    stock_prices_long_bt = stock_prices[
        (stock_prices["PERMNO"].isin(tickers_traded)) &
        (stock_prices["date"] >= date_first_transac) &
        (stock_prices["date"] <= date_last_transac)
    ].copy()
    stock_prices_bt = stock_prices_long_bt.pivot_table(values="PRC", columns="PERMNO", index="date")
    stock_prices_bt.to_csv(r'data\checkpoints\stock_prices.csv')

    # On reconstitue le prix d'un portfolio investit dans le benchmark
    bench_price = (bench_returns + 1).cumprod() - 1
    bench_price = bench_price[
        (bench_price.index >= date_first_transac) &
        (bench_price.index <= date_last_transac)
        ].copy()


    engine = BacktestEngine(
        transaction_journal_bt, 
        stock_prices_bt, 
        bench_df_input=bench_price, 
        annual_discount_rate=0,
        # stop_loss=-0.02
    )
    engine.run()
    return engine
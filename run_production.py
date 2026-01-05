import pandas as pd
from pathlib import Path

from src.data import load_compo_universe, load_stock_prices
from src.utils import config_logger, format_financial_report, save_single_graph
from src.workflow import run_rolling_backtest, compute_rebal_date
from config import Config

# TODO: continuer à clean le code
# TODO: passer les print en log
# TODO: dans le tableau comparatif mettre les data du benchmark
# TODO: mettre le formatting dans la librairie backtesting
# TODO: faire cross-validation pour ma strat
# TODO: faire du volatility time management pour essayer de réduire la vol en baissant moins les returns pour booster sharp ratio
# => cela pourrais définir notre date de rebal qui serait dynamique
# TODO: corriger survivor bias
# TODO: ajouter le rf asset
# TODO: faire alpha against momentum

# TODO: corriger les trous temporels dans nos df

if __name__ == "__main__" :
    conf = Config()

    logger = config_logger()
    logger.info('Running strat')

    logger.info("Step1: Data Loadings")
    compo_universe = pd.read_csv(r"data\raw\compo_sp500_final.csv", parse_dates=['MbrStartDt', 'MbrEndDt'])

    path = r"data\raw\stock_prices_final.csv"
    stock_prices = pd.read_csv(path, parse_dates=['date'])
    stock_prices.sort_values("date", inplace=True) # S'assurer que tout est trié avant
    stock_prices['RET_calc'] = stock_prices.groupby("PERMNO")['PRC'].pct_change()

    stock_prices = stock_prices[stock_prices['date'] < conf.END_DATE_STRAT].copy()
    stock_prices['PERMNO'] = stock_prices['PERMNO'].apply(str)
    compo_universe['PERMNO'] = compo_universe['PERMNO'].apply(str)

    df_fama = pd.read_csv(
        r"data\raw\F-F_Research_Data_5_Factors_2x3_daily.csv", 
        skiprows=3, 
        skipfooter=1, 
        sep=",", 
        index_col=0, 
        parse_dates=True, 
        engine='python'
        )
    df_mom = pd.read_csv(
        r"data\raw\F-F_Momentum_Factor_daily.csv", 
        skiprows=12, 
        skipfooter=1, 
        sep=",", 
        index_col=0, 
        parse_dates=True, 
        engine='python'
        )
    bench_returns = pd.concat([df_fama['Mkt-RF'], df_mom], axis=1)
    bench_returns.dropna(inplace=True)
    bench_returns = (bench_returns['Mkt-RF'] / 100).to_frame()
    bench_returns = bench_returns[bench_returns.index < conf.END_DATE_STRAT].copy()

    rebal_dates = compute_rebal_date(conf.WINDOW, stock_prices, bench_returns, conf.START_DATE, conf.UNIT, conf.STEP, logger)


    WINDOW_list = [20]
    engine = run_rolling_backtest(WINDOW_list, rebal_dates, stock_prices, compo_universe, logger, bench_returns, conf.UNIT, conf.INITIAL_CAPITAL, conf.MAX_VOL, conf.END_DATE_STRAT)



    logger.info("Saving results")
    metrics = pd.Series(engine.summary())
    df_report = format_financial_report(metrics)
    df_report.to_markdown(Path('results') / 'report.md', index=True)

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
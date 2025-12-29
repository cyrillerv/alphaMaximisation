import pandas as pd
from pathlib import Path

from src.data import load_histo_cac, load_stock_prices
from src.utils import config_logger, format_financial_report, save_single_graph
from src.workflow import run_rolling_backtest, compute_rebal_date
from config import Config

# TODO: continuer à clean le code
# TODO: passer les print en log
# TODO: dans le tableau comparatif mettre les data du benchmark
# TODO: mettre le formatting dans la librairie backtesting
# TODO: faire cross-validation pour ma strat


if __name__ == "__main__" :
    conf = Config()

    logger = config_logger()
    logger.info('Running strat')



    logger.info("Step1: Data Loadings")
    histo_compo_cac = load_histo_cac()
    all_tickers = [conf.BENCHMARK] + list(histo_compo_cac.columns)
    stock_returns, stock_prices = load_stock_prices(all_tickers)
    stock_returns.dropna(subset=[conf.BENCHMARK], inplace=True)




    bench_returns = stock_returns[conf.BENCHMARK].copy()
    df_stocks_ret = stock_returns.drop(columns=[conf.BENCHMARK]).copy()



    rebal_dates = compute_rebal_date(conf.WINDOW, stock_returns, bench_returns, conf.START_DATE, conf.UNIT, conf.STEP, logger)


    WINDOW_list = [17, 18, 19, 20, 21]
    engine = run_rolling_backtest(WINDOW_list, rebal_dates, stock_prices, df_stocks_ret, histo_compo_cac, logger, bench_returns, conf.UNIT, conf.INITIAL_CAPITAL, conf.BENCHMARK, conf.MAX_VOL)



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
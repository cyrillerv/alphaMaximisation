from tqdm import tqdm

from src.data import load_histo_cac, load_stock_prices
from src.utils import config_logger, save_single_graph, plot_calibration_results
from src.workflow import run_rolling_backtest, compute_rebal_date
from config import Config

# TODO: continuer à clean le code
# TODO: passer les print en log
# TODO: dans le tableau comparatif mettre les data du benchmark
# TODO: mettre le formatting dans la librairie backtesting
# TODO: faire cross-validation pour ma strat
# TODO: finir de trouver les tickers du CAC

if __name__ == "__main__" :
    conf = Config()

    logger = config_logger()
    logger.info('Running Calibration')



    logger.info("Step1: Data Loadings")
    histo_compo_cac = load_histo_cac()
    all_tickers = [conf.BENCHMARK] + list(histo_compo_cac.columns)
    stock_returns, stock_prices = load_stock_prices(all_tickers, univers=conf.BENCHMARK)
    stock_returns.dropna(subset=[conf.BENCHMARK], inplace=True)


    bench_returns = stock_returns[conf.BENCHMARK].copy()
    df_stocks_ret = stock_returns.drop(columns=[conf.BENCHMARK]).copy()


    sharps = {}
    for window_value in tqdm(range(1, 25)) :
        WINDOW_list = [window_value]
        rebal_dates = compute_rebal_date(window_value, stock_returns, bench_returns, conf.START_DATE, conf.UNIT, conf.STEP, logger)
        engine = run_rolling_backtest(WINDOW_list, rebal_dates, stock_prices, df_stocks_ret, histo_compo_cac, logger, bench_returns, conf.UNIT, conf.INITIAL_CAPITAL, conf.BENCHMARK, conf.MAX_VOL)
        sharps[window_value] = engine.summary()['sharpe_ratio']

    fig = plot_calibration_results(sharps, robustness_range=(19, 22))

    save_single_graph(fig, "Calibration")

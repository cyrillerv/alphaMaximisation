import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data import load_histo_cac, load_stock_prices
from src.optimization import optimize_single_period
from src.execution import generate_transaction_log
from backtesting.core import BacktestEngine 

WINDOW = 2 # in years
STEP = 1 # in years
MAX_VOL = 0.30
INITIAL_CAPITAL = 1_000_000
START_DATE = "2010-01-01"

print("Step1: Data Loadings")
histo_compo_cac = load_histo_cac()
all_tickers = ["^FCHI"] + list(histo_compo_cac.columns)
stock_returns, stock_prices = load_stock_prices(all_tickers)
stock_returns.dropna(subset=['^FCHI'], inplace=True)

# Regarder si la start date laisse suffisamment d'histo par rapport à la window choisie
cutoff = pd.to_datetime(START_DATE) - pd.DateOffset(years=WINDOW)
has_history = stock_returns.index.min() <= cutoff
if not has_history :
    START_DATE = stock_returns.index.min() + pd.DateOffset(years=WINDOW)

bench_returns = stock_returns['^FCHI'].copy()
df_stocks_ret = stock_returns.drop(columns=['^FCHI']).copy()

end_date = bench_returns.index.max()
# Define a list of all rebalancing dates
rebal_dates = list(pd.date_range(start=START_DATE, end=end_date, freq=pd.DateOffset(years=STEP)))

dic_weights = {}
for rebal_date in rebal_dates :
    start_date_window = rebal_date - pd.DateOffset(years=2)
    end_date_window = rebal_date
    # Récupérer les données de la window
    stock_ret_window = df_stocks_ret.loc[start_date_window:end_date_window].copy()
    try:
        # Récupérer la liste des tickers composant le CAC au début de la période
        compo_at_date = histo_compo_cac.replace(0, np.nan).reindex([stock_ret_window.index.min()], method='ffill').T.dropna()
        valid_tickers = compo_at_date.index.intersection(df_stocks_ret.columns)
    except Exception:
        continue

    if len(valid_tickers) < 2:
        continue

    stock_ret_window = stock_ret_window[valid_tickers].dropna(axis=1)
    bench_ret_window = bench_returns.loc[start_date_window:end_date_window].copy()

    if stock_ret_window.empty:
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
print(f"Poids générés pour {len(df_weights_filtered)} dates de rebalancement.")
print(df_weights_filtered.tail())

print("Génération du journal des transactions...")
df_transactions = generate_transaction_log(
    df_weights_filtered, 
    stock_prices, 
    initial_capital=INITIAL_CAPITAL
)

print("Lancement du moteur de backtest...")
df_prices_bt = stock_prices[list(set(df_transactions['Symbol'])) + ["^FCHI"]]
engine = BacktestEngine(
    df_transactions, 
    df_prices_bt, 
    bench_df_input=df_prices_bt['^FCHI'].to_frame(), 
    annual_discount_rate=0
)

engine.run()
engine.summary()
engine.plot_graphs()
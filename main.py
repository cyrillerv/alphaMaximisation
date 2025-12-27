import pandas as pd
import numpy as np
from tqdm import tqdm

# Imports de tes modules
from src.data import load_histo_cac, load_stock_prices
from src.optimization import optimize_single_period
from src.execution import generate_transaction_log
from backtesting.core import BacktestEngine 

WINDOW = 365 * 2
STEP = 365
MAX_VOL = 0.30
INITIAL_CAPITAL = 1_000_000

print("Chargement des données...")
histo_compo_cac = load_histo_cac()
all_tickers = ["^FCHI"] + list(histo_compo_cac.columns)
stock_returns, stock_prices = load_stock_prices(all_tickers) 
stock_returns.dropna(subset=['^FCHI'], inplace=True)

data_slice_ret = stock_returns.loc["2010-01-01":].copy()
s_bench_ret = data_slice_ret['^FCHI']
df_stocks_ret = data_slice_ret.drop(columns=['^FCHI'])

print("Démarrage de l'optimisation...")
indices = range(0, len(s_bench_ret) - WINDOW + 1, STEP)
dic_weights = {}

for start in tqdm(indices):
    end = start + WINDOW
    
    current_rebalance_date = s_bench_ret.index[end-1]
    
    idx_univers = max(0, end - STEP)
    date_univers = s_bench_ret.index[idx_univers]
    
    try:
        compo_at_date = histo_compo_cac.replace(0, np.nan).reindex([date_univers], method='ffill').T.dropna()
        valid_tickers = compo_at_date.index.intersection(df_stocks_ret.columns)
    except Exception:
        continue

    if len(valid_tickers) < 2:
        continue

    stock_returns_win = df_stocks_ret.iloc[start:end][valid_tickers].dropna(axis=1)
    bench_win = s_bench_ret.iloc[start:end]

    if stock_returns_win.empty:
        continue

    optimal_weights = optimize_single_period(
        stock_returns_win.values, 
        bench_win.values, 
        max_volatility=MAX_VOL
    )

    if optimal_weights is not None:
        dic_weights[current_rebalance_date] = dict(zip(stock_returns_win.columns, optimal_weights))

df_weights = pd.DataFrame(dic_weights).T.sort_index()

df_weights_filtered = df_weights.fillna(0).round(4)
df_weights_filtered = df_weights_filtered.loc[(df_weights_filtered != 0).any(axis=1)]

print(f"Poids générés pour {len(df_weights_filtered)} dates de rebalancement.")
print(df_weights_filtered.tail())

print("Génération du journal des transactions...")

df_prices_bt = stock_prices.loc["2010-01-01":].ffill()

df_transactions = generate_transaction_log(
    df_weights_filtered, 
    df_prices_bt, 
    initial_capital=INITIAL_CAPITAL
)

df_prices_bt = df_prices_bt[list(set(df_transactions['Symbol']))]

print("Lancement du moteur de backtest...")
engine = BacktestEngine(
    df_transactions, 
    df_prices_bt, 
    bench_df_input=df_prices_bt['^FCHI'].to_frame(), 
    annual_discount_rate=0
)

engine.run()
engine.summary()
engine.plot_graphs()
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_transaction_log(df_weights_filtered, stock_prices, initial_cash, end_date_strat) :
    cash = initial_cash
    nb_stocks = {}
    closing_dates = list(df_weights_filtered.index[1:]) + [end_date_strat]
    all_transac = pd.DataFrame()
    stock_prices.sort_values('date', inplace=True)

    for (_, rows), closing_date in zip(
            tqdm(df_weights_filtered.iterrows()),
            closing_dates
        ):
        date = rows.name
        prices_date = stock_prices[stock_prices['date'] <= date].set_index('PERMNO')['PRC'].copy()
        # D'abord on vend les positions qu'on a en portfolio
        tickers_in_ptf = [ticker for ticker, volume in nb_stocks.items() if volume != 0]
        price_date_stock = prices_date.loc[tickers_in_ptf].copy()
        price_date_stock = price_date_stock[~price_date_stock.index.duplicated(keep='last')].to_dict()
        result = sum([price_date_stock[k] * nb_stocks[k] for k in price_date_stock.keys() & nb_stocks.keys()])
        cash += result
        nb_stocks = {}
        
        # Puis on achète les nouvelles
        filtered_row = rows[rows.ne(0)].copy()
        price_date_stock = prices_date.loc[filtered_row.index].copy()
        price_date_stock = price_date_stock[~price_date_stock.index.duplicated(keep='last')]
        amt_per_stock = filtered_row * cash
        volume_per_stock = (amt_per_stock // price_date_stock).replace(0, np.nan).dropna()
        nb_stocks = volume_per_stock.to_dict()

        # On enlève au cash ce qu'on a dépensé
        used_cash = (volume_per_stock * price_date_stock).sum()
        cash -= used_cash

        # Et on construit le df de transactions
        transactions = volume_per_stock.rename('Volume').reset_index()
        transactions.rename(columns={"index": "Symbol"}, inplace=True)
        transactions['Date'] = date
        transactions['Type'] = 'Buy'
        transactions['ClosingDate'] = closing_date

        all_transac = pd.concat([all_transac, transactions])
    all_transac = all_transac[all_transac['Volume'].ne(0)].copy()
    return all_transac
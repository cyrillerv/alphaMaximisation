import pandas as pd
import numpy as np
import yfinance as yf
import os
import pandas as pd

# def load_histo_cac() :
#     df_histo_cac = pd.read_excel("cac_histo.xlsx", sheet_name='Sheet1')
#     df_histo_cac.dropna(subset=['Valeur'], inplace=True)
#     df_histo_cac['Date'] = pd.to_datetime(df_histo_cac['Date'])
#     df_histo_cac.set_index('Date', inplace=True)

#     df_map_ticker = pd.read_excel("cac_histo.xlsx", sheet_name='Sheet2')
#     df_map_ticker.dropna(inplace=True)
#     dic_map_ticker = df_map_ticker.set_index('Valeur')['Unnamed: 1'].to_dict()

#     df_histo_cac['Ticker'] = df_histo_cac['Valeur'].map(dic_map_ticker)
#     liste_in = ['Admission']
#     liste_out = ['Retrait']
#     liste_suspension = ['Suspension', 'Fin de suspension']
#     df_histo_cac['IndicSens'] = np.where(df_histo_cac['Sens'].isin(liste_in), 1, 
#                                         np.where(df_histo_cac['Sens'].isin(liste_suspension), 0, -1)
#                                         )

#     df_histo_cac.dropna(subset=['Ticker'], inplace=True)
#     df_indic_in_out_cac = df_histo_cac.reset_index().pivot_table(index='Date', columns='Ticker', values='IndicSens', aggfunc='sum', fill_value=np.nan)
#     df_compo_cac = df_indic_in_out_cac.cumsum()
#     df_compo_cac.ffill(inplace=True)
#     return df_compo_cac.fillna(0)

def load_compo_universe(path) :
    compo_universe = pd.read_csv(path, index_col=0)
    compo_universe.index = pd.to_datetime(compo_universe.index)
    date_range = pd.date_range(compo_universe.index.min(), compo_universe.index.max(), freq='D')
    compo_universe = compo_universe.reindex(date_range, method='ffill')
    return compo_universe
    

def load_stock_prices(tickers, univers, reload=False) :
    file_path = f'stock_prices_{univers}.csv'
    if (not os.path.exists(file_path)) or reload: 
        res = yf.download(tickers, period="max", interval="1d")
        stock_prices = res['Close']
        full_index = pd.date_range(
            start=stock_prices.index.min(),
            end=stock_prices.index.max(),
            freq="D"
        )
        stock_price_filtered = stock_prices.reindex(full_index, method='ffill').copy()
        stock_price_filtered.to_csv(f'stock_prices_{univers}.csv')
    else :
        stock_price_filtered = pd.read_csv(f'stock_prices_{univers}.csv', index_col=0)
        stock_price_filtered.index = pd.to_datetime(stock_price_filtered.index)
    stock_returns = stock_price_filtered.pct_change().copy()
    stock_returns = stock_returns.iloc[1:]
    return stock_returns, stock_price_filtered
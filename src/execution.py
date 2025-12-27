import pandas as pd
import numpy as np

def generate_transaction_log(df_weights, df_prices, initial_capital=1_000_000):

    prices_ffill = df_prices.ffill()
    
    valid_dates = df_weights.index.intersection(prices_ffill.index)
    df_weights = df_weights.loc[valid_dates]
    
    current_holdings = pd.Series(dtype=float)
    cash = initial_capital
    transactions = []

    for date in df_weights.index:
        current_prices = prices_ffill.loc[date]
        
        equity_value = 0
        if not current_holdings.empty:
            valid_holdings = current_holdings.index.intersection(current_prices.index)
            equity_value = (current_holdings[valid_holdings] * current_prices[valid_holdings]).sum()
        
        total_portfolio_value = cash + equity_value
        
        target_weights = df_weights.loc[date]
        target_weights = target_weights[target_weights > 0] 
        
        target_amounts = target_weights * total_portfolio_value
        
        current_prices_subset = current_prices.loc[target_weights.index]
        target_quantities = (target_amounts // current_prices_subset).fillna(0)
        
        all_tickers = current_holdings.index.union(target_quantities.index)
        
        current_qty_aligned = current_holdings.reindex(all_tickers, fill_value=0)
        target_qty_aligned = target_quantities.reindex(all_tickers, fill_value=0)
        
        order_quantities = target_qty_aligned - current_qty_aligned
        
        for ticker, qty in order_quantities[order_quantities != 0].items():
            direction = 'Buy' if qty > 0 else 'Sell'
            
            transactions.append({
                'Date': date,
                'Symbol': ticker,
                'Type': direction,
                'Volume': abs(qty)
            })

        current_holdings = target_quantities[target_quantities > 0].copy()
        
        invested_amount = (current_holdings * current_prices.loc[current_holdings.index]).sum()
        cash = total_portfolio_value - invested_amount

    return pd.DataFrame(transactions)
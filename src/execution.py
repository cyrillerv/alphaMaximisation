import pandas as pd
import numpy as np

def generate_transaction_log(df_weights, df_prices, initial_capital=1_000_000):
    # On s'assure que les prix sont remplis
    prices_ffill = df_prices.ffill()
    
    # Alignement des dates : on ne garde que les dates de rebalancement présentes dans les prix
    valid_dates = df_weights.index.intersection(prices_ffill.index)
    df_weights = df_weights.loc[valid_dates]
    
    current_holdings = pd.Series(dtype=float)
    cash = initial_capital
    transactions = []

    # On parcourt chaque date de rebalancement
    for date in df_weights.index:
        # 1. Valorisation du portefeuille actuel
        current_prices = prices_ffill.loc[date]
        
        # Valeur des actions détenues
        equity_value = 0
        if not current_holdings.empty:
            # On ne garde que les actifs qui ont un prix à cette date
            valid_holdings = current_holdings.index.intersection(current_prices.index)
            equity_value = (current_holdings[valid_holdings] * current_prices[valid_holdings]).sum()
        
        total_portfolio_value = cash + equity_value
        
        # 2. Calcul des nouvelles quantités cibles (Target)
        target_weights = df_weights.loc[date]
        target_weights = target_weights[target_weights > 0] # On ignore les poids nuls
        
        # Montant alloué par action
        target_amounts = target_weights * total_portfolio_value
        
        # Quantité cible (division entière pour simuler des lots complets)
        # On utilise .reindex pour gérer les actifs qui ne sont pas dans les prix (cas rare mais possible)
        current_prices_subset = current_prices.loc[target_weights.index]
        target_quantities = (target_amounts // current_prices_subset).fillna(0)
        
        # 3. Calcul des Deltas (Ordres à passer)
        # On combine l'ancien et le nouveau portefeuille pour avoir tous les tickers
        all_tickers = current_holdings.index.union(target_quantities.index)
        
        current_qty_aligned = current_holdings.reindex(all_tickers, fill_value=0)
        target_qty_aligned = target_quantities.reindex(all_tickers, fill_value=0)
        
        order_quantities = target_qty_aligned - current_qty_aligned
        
        # 4. Enregistrement des transactions
        for ticker, qty in order_quantities[order_quantities != 0].items():
            direction = 'Buy' if qty > 0 else 'Sell'
            
            transactions.append({
                'Date': date,
                'Symbol': ticker,
                'Type': direction,
                'Volume': abs(qty)
            })
        
        # 5. Mise à jour du portefeuille théorique pour la prochaine itération
        # Note : Dans un backtest simple sans gestion précise du cash résiduel, 
        # on considère que le portefeuille cible est atteint.
        current_holdings = target_quantities[target_quantities > 0].copy()
        
        # Mise à jour du cash (le reste non investi à cause de la division entière)
        invested_amount = (current_holdings * current_prices.loc[current_holdings.index]).sum()
        cash = total_portfolio_value - invested_amount

    return pd.DataFrame(transactions)
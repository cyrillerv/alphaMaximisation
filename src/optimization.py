import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.utils import *

import torch

def prepare_alpha_terms_torch(stock_returns, benchmark_returns, device="cuda"):
    R = torch.tensor(stock_returns.to_numpy(), dtype=torch.float32, device=device)
    b = torch.tensor(benchmark_returns.to_numpy(), dtype=torch.float32, device=device)

    mu_R = R.mean(dim=0)
    mu_b = b.mean()

    cov_Rb = ((R - mu_R) * (b - mu_b).unsqueeze(1)).mean(dim=0)
    var_b = b.var(unbiased=False)

    return mu_R, mu_b, cov_Rb, var_b


def portfolio_volatility_torch(w, cov):
    return torch.sqrt(w @ cov @ w)


def optimize_single_period_gpu(
    stock_returns,
    benchmark_returns,
    max_volatility=0.30,
    device="cuda",
    n_iter=100
):
    df = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()
    if len(df) < 2:
        return None

    stock_returns = df.iloc[:, :-1]
    benchmark_returns = df.iloc[:, -1]

    mu_R, mu_b, cov_Rb, var_b = prepare_alpha_terms_torch(
        stock_returns, benchmark_returns, device
    )

    cov = torch.tensor(
        np.cov(stock_returns.to_numpy(), rowvar=False),
        dtype=torch.float32,
        device=device
    )

    n = stock_returns.shape[1]
    w = torch.full((n,), 1 / n, device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS([w], max_iter=n_iter)

    def closure():
        optimizer.zero_grad()

        # alpha (à maximiser → on minimise -alpha)
        alpha = w @ (mu_R - cov_Rb / var_b)

        vol = portfolio_volatility_torch(w, cov)

        # pénalités contraintes
        penalty_sum = 1e3 * (w.sum() - 1) ** 2
        penalty_vol = 1e3 * torch.clamp(vol - max_volatility, min=0) ** 2
        penalty_bounds = 1e3 * torch.sum(torch.clamp(-w, min=0) ** 2)

        loss = -alpha + penalty_sum + penalty_vol + penalty_bounds
        loss.backward()
        return loss

    optimizer.step(closure)

    w = torch.clamp(w, min=0)
    w = w / w.sum()

    return w.detach().cpu().numpy()




# def optimize_single_period(stock_returns, benchmark_returns, max_volatility=0.30):

#     df = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner").dropna()
#     stock_returns = df.iloc[:, :-1]
#     benchmark_returns = df.iloc[:, -1]

#     if len(df) < 2:
#         return None

#     mu_R, mu_b, cov_Rb, var_b = prepare_alpha_terms(
#         stock_returns, benchmark_returns
#     )

#     cov_matrix = np.cov(stock_returns.to_numpy(), rowvar=False)

#     n = stock_returns.shape[1]
#     x0 = np.full(n, 1 / n)
#     bounds = [(0, 1)] * n

#     constraints = (
#         {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
#         {'type': 'ineq', 'fun': lambda w: max_volatility - get_portfolio_volatility(w, cov_matrix)}
#     )

#     result = minimize(
#         objective_maximize_alpha_fast,
#         x0,
#         args=(mu_R, mu_b, cov_Rb, var_b),
#         method="SLSQP",
#         bounds=bounds,
#         constraints=constraints
#     )

#     return result.x if result.success else None

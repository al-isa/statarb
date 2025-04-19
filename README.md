This is a statiscal arbitrage model which uses a cointegration-based pairs trading strategy.
Heres the overall workflow/pipeline:
1. Pull Data: We get the historcial daily closing prices for a list of bank stocks from Yahoo Finance
2. Normalize & Visualize: Normalize prices and show visual correlation
3. Cointegration Testing: For every pair of stocks, check if they are cointegrated-ie, they move together in a statically reliable way over time
4. Spread Aalysis: Compute the spread between the two stocks, test for mean-reversion using the Augmented Dickey-Fuller test and calculate the half life of the mean reversion
5. Z-Score Trading: Use the z-score threshold to simulate basic trading strategy-go long/short when spread deviates too far from the eman, and exit when it reverts
6. Backtest and Evaluate: Visualize positions, z-scores, and cumulative returns. Evalute using Sharpe ratio, drawdown, and return metrics

Statistics and Math Analysis:
Conitegration:
In time series analysis, cointegration is a relation ship between two non stationary series where a linear combination of them is stationary
Intuition - Stock prices are usually non-stationary, they trend over time thus their mean/variance change. But two stocks may move together over time, occasionally straying apart but always returning near each other. If their difference (spread) is stable around a mean, then they are cointegrated
Assume two asset price series: Yₜ and Xₜ
We can say they are conintegrated if there exists a linear combination: Zₜ = Yₜ − βXₜ − α, such that Zₜ is stationary (Zₜ∼I(0)) and Yₜ,Xₜ∼I(1),
Zₜ will act as the spread between two stocks, it behaves like a mean-reverting process: Zₜ∼N(μ,σ^2), this is what we trade on. If it drifts too far from the mean, we will predict it will return

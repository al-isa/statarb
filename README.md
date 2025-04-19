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
For each stock pair, the code pulls their historical daily closing prices and tests whether they are conintegrated.
We use the Engle-Granger two-step conitegration test.
Step 1 of Engle Granger: We run the OLS regression Zₜ = Yₜ − βXₜ − α, and we get the redisudals εₜ = Yₜ − βXₜ − α
Step 2: We apply the ADF test (Augmented Dickey-Fuller) onεₜ to check for stationarity.
H₀: No cointegration  →  residuals are non-stationary
H₁: Cointegration exists  →  residuals are stationary
So we reject the null if p<0.05 meaning cointegration exists
This threshold (0.05) means there is less than 5% probability that these two stocks are not cointegrated. It's a common threshold in econometrics. Alternatively we may use p<0.01 for very strong evidence, or p<0.10 for weak/moderate evidence. 
Conclusion (Why do we test for conitegartion and not correlation?): Correlation measures short-term co-movement, but not whether the relationship holds over time. So for example, Correlation might be high between AAPL and MSFT during a bull run, but if their spread keeps drifting-thecorrelation is useless for pairs trading, cointegration checks long-term equilibrium-far more robust for trading strategies

Z-Score Normalization and Trading Signals:
Once we find two stocks that are cointegrated, we construct a spread between them and monitor how far it deviates from the historical average. That deviation is expressed using a z-score to normalize data.
Given two conintegrated prices series: Yₜ and Xₜ
We regress them to find: β (the hedge ratio), α (intercept/drift)
Then compute the spread, Spreadₜ = Yₜ − β × Xₜ − α
The raw spread isn't standardized, so we nomrlaize it to get a z-score Zₜ = (Spreadₜ − μ) / σ
Where:
μ: historical mean of the spread
σ: standard deviation of the spread
Now we can interpret that:
Zₜ = 0 → Spread is at its mean
Zₜ = 2 → Spread is 2 standard deviations above mean (likely overextended)
Zₜ = -2 → Spread is 2 std dev below mean (likely underextended)
This is our rule-based system to enter and exit trades.
Trading Signal Logic:
Entry Conidtions - When the spread is significantly far from the mean, we expect mean reversion, so we trade against the deviation
If Zₜ > +1.5:
    SHORT the spread → Short Yₜ, Long Xₜ
If Zₜ < -1.5:
    LONG the spread → Long Yₜ, Short Xₜ
Exit Conditions - We exit when the spread reverts to the mean
If current position is active and Zₜ crosses 0:
    CLOSE the position
We think of the z-score as a sentiment gauge for how "streched" the relationship is, when the z-score is far from 0, the market is irrational-so we assume reversion, when it nears 0, the pair is in balance-no action needed.
This works because z-scores are under the assumption that the spread follows a normal distrubtion centered at 0: Z ∼ 𝒩(0, 1)
So, about 68% of the values are within ± 1 and about 95% are within ± 2. If we see |Z| > 1.5, this is a rare event-which is potentially a profitable opportunity

Half-Life of Mean Reversion:
The half life gives us a way to quantify the speed of mean reversion (How long does it take to revert halfway back, if the spread between two cointegrated stocks moves away from the average). This is important as if the mean revrsion is too slow (e.g, 200 days) your capital is tied up for too long, if its too fast (e.g, 1 day, its likely just noise or overfitting. We wants pairs with reasonable and predictable snapbacks-ideally within 2-60 trading days.
We model the spread using a simple first-order autoregressive model, or AR(1)
ΔSpreadₜ = λ × Spreadₜ₋₁ + εₜ
Where
ΔSpreadₜ = Spreadₜ − Spreadₜ₋₁ is the change in spread
Spreadₜ₋₁ is the lagged spread
λ is the mean reversion speed
εₜ is white noise
This is fitted using statsmodels.tsa.ar_model.AutoReg
Once we estimate λ, the half-life 𝜏 is derived by solving this decay equation: τ = −ln(2) / λ
We use ln(2) because the solution to an AR(1) mean-reverting process resembles exponential decay: Spreadₜ = Spread₀ × e^(−λt)
So, half-life 𝜏 is the value of t when:Spreadₜ = ½ × Spread₀
⇒ ½ = e^(−λτ)
⇒ ln(½) = −λτ
⇒ τ = −ln(2) / λ
This tells us the expected time it takes for a shock to decay by 50%
​In this code, pairs are filtered with:
if half_life < 2 or half_life > 60:
    continue
So we discard fast mean-reverting pairs (likely just noise or intraday fluvtuations) and slow mean-reverting pairs (not tradable over a swing timeframe). We keep pairs where the spread will revert within a realistic trading horizon
Edge Case:
If λ is 0, there is no mean reversion at all-spread behaves like a random walk. In this case half life = ∞, we handle this with:
if lambda_val == 0:
    return np.inf
Conclusion: This matters because while cointegration tells us whether the spread is mean-rverting, the half-life tells us how tradable it is. This is a classic example of statiscal validity and practical utility

Backtesting Logic and Performance Metrics:
We base positions entirely on the z-score of the spread.
Entry:
If Zₜ < −1.5:
    Long Spread → Buy Y, Sell X
If Zₜ > +1.5:
    Short Spread → Sell Y, Buy X
Exit:
If Zₜ crosses 0:
    Close Position (mean reversion hit)
Position Encoding:
0: No position
1: Long Spread (Long Y, Short X)
-1: Short Spread (SHort Y, Long X)
This is implemented using a forward-looking loop:
for i in range(1, len(pair_df)):
    z = pair_df['zscore'].iloc[i]
    prev_pos = pair_df['position'].iloc[i-1]
Every day we decide whether to enter, hold, or exit a trade
Calculating Strategy Returns:
Once trades are simulated, we calculate how profitable they are. We first compute daily percentage returns of each stock:
s1_returns = ΔYₜ / Yₜ₋₁
s2_returns = ΔXₜ / Xₜ₋₁
If we long the spread out position earns:
Return = +s1_returns − β × s2_returns
If we short the spread, we invert it:
Return = −(s1_returns − β × s2_returns)
To generalize:
strategy_returnsₜ = positionₜ₋₁ × (s1_returnsₜ − β × s2_returnsₜ)
To see how wealth compounds over time:
cum_returnₜ = (1 + r₁) × (1 + r₂) × ... × (1 + rₜ) − 1
This gives us the equity curve of the strategy
Peformance Metrics:
Total Return - Overall % gain/loss: Total_Return = cum_strategy_returnsₜ × 100
Annualized Return - Assuming 252 trading days: Annual_Return = ((1 + Total_Return/100) ** (252 / N)) − 1 (Where N = number of trading days)
Sharpe ratio - Evaluates return vs risk (High Shapre-High risk-adjusted return): Sharpe = √252 × (mean(daily returns) / std(daily returns))
Maximum Drawdown - Worst peak-to-trough decline: Max_Drawdown = min(cum_returns − cum_max(cum_returns)) × 100
Number of trades - How many times we switched positions: num_trades = (position != position.shift(1)).sum() - 1
![image](https://github.com/user-attachments/assets/305445a2-2b37-4f76-8435-1486f447db3a)
![image](https://github.com/user-attachments/assets/d88c28e2-21d8-4512-95ed-31bcd22bf504)

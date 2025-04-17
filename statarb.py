import pandas as pd
import numpy as np
import seaborn
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.ar_model import AutoReg
from itertools import combinations
import statistics

#Pairs trading stategy,
#We use mean-reverting stock pairs using cointegration tests and backtest their performance
#We download the data, normailze it, analyse the potential cointegration and use data visualization
#and back test with performance metrics

tickers = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'BK']
data = yf.download(tickers, start="2020-01-01", end="2024-12-31", group_by='ticker')

#organize the closing prices into a single dataframe for easy access
close_prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers})

#normalize prices to visually compare stocks
normalized = close_prices / close_prices.iloc[0]#Normalize prices

def plot_normalized_prices():
    #Plot normalized prices to visually see similarity
    normalized.plot(figsize=(14, 6), title="Normalized Stock Prices")
    plt.ylabel("Normalized Price")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap():
    corrmat = close_prices.corr()
    plt.figure(figsize=(8, 6))
    seaborn.heatmap(corrmat, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Stock Correlation Heatmap")
    plt.show()

coint_pairs = []
coint_stats = []


#defunct code for now, other cointegration method yields better results, should recheck math here for future implementations
'''
def find_pairs_regression():
    #Find pairs using OLS regression and filter based on spread statistics
    print("Using Regression Method")
    for stock1, stock2 in combinations(close_prices.columns, 2):
        series1 = close_prices[stock1].values
        series2 = close_prices[stock2].values

        #y=α+βx+ϵ
        #stack x and a column of 1s to solve for beta and alpha
        X = np.vstack([series2, np.ones(len(series2))]).T
        beta, alpha = np.linalg.lstsq(X, series1, rcond=None)[0]

        #Calculate residuals between actual y and predicted y_hat
        spread = series1 - (beta * series2 + alpha)

        #normalize spread
        zscore = (spread - spread.mean()) /spread.std()

        # Mean and std
        z_std = np.std(zscore)
        z_mean = np.mean(zscore)

        # Drift check
        slope = np.polyfit(np.arange(len(zscore)), zscore, 1)[0]

        # Rolling stability check
        z_rolling_std = pd.Series(zscore).rolling(window=100).std().dropna()
        z_std_stability = z_rolling_std.std()

        print(f"\nPair: {stock1}/{stock2}")
        print(f"Z-Mean: {z_mean:.4f}")
        print(f"Z-Std: {z_std:.4f}")
        print(f"Slope: {slope:.6f}")
        print(f"Z-Std Stability: {z_std_stability:.4f}")
        # Filter conditions
        # Tiered filter: strict first, relaxed backup
        if (
            0.95 < z_std < 1.05 and
            abs(z_mean) < 0.05 and
            abs(slope) < 0.0005 and
            z_std_stability < 0.11
        ):
            tier = "A"
        elif (
            0.95 < z_std < 1.05 and
            abs(z_mean) < 0.05 and
            abs(slope) < 0.002 and
            z_std_stability < 0.15
        ):
            tier = "B"
        else:
            tier = None

        if tier:
            regression_pairs.append((stock1, stock2))
            regression_stats.append({
                'beta': beta,
                'alpha': alpha,
                'z_mean': z_mean,
                'z_std': z_std,
                'slope': slope,
                'z_std_stability': z_std_stability,
                'tier': tier
            })
'''

#helper function to test wether a time series is stationary using teh Augmented Dickey-Fuller test
def test_stationarity(series, significance_level=0.05):
    #tests for stationarity using ADF test
    result = adfuller(series)
    pvalue = result[1]
    
    return pvalue < significance_level, pvalue

def calculate_half_life(spread):
    spread_lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    model = AutoReg(delta, lags=1, old_names=False)
    res = model.fit()
    lambda_val = res.params[1]
    if lambda_val == 0:
        return np.inf
    half_life = -np.log(2) /lambda_val
    return half_life

#Find all pairs of stocks that are cointegrated (ie. their price differences tend to revert to a mean)
def find_pairs_coint():
    print("Using CoIntegration Method")
    for stock1, stock2 in combinations(close_prices.columns, 2):
        series1 = close_prices[stock1]
        series2 = close_prices[stock2]

        #perform coint test
        _, pvalue, _ = coint(series1, series2)

        #if pvalue is less than threshold (changeable), pairs are liekly cointegrated
        if pvalue < 0.05:
            #Calculate hedge ratio (beta) using OLS regression
            X = np.vstack([series2, np.ones(len(series2))]).T
            beta, alpha = np.linalg.lstsq(X, series1, rcond=None)[0]

            #Calculate residuals between actual y and predicted y_hat
            spread = series1 - (beta * series2 + alpha)

            #Calculate half-life to check for mean revrsion speed
            half_life = calculate_half_life(spread)

            if half_life < 2 or half_life > 60:
                continue

            #normalize spread into a zscore
            zscore = (spread - spread.mean()) /spread.std()

            # Assessment of the quality of the mean reversion

            # Mean and std
            z_std = np.std(zscore)
            z_mean = np.mean(zscore)

            # Drift check
            slope = np.polyfit(np.arange(len(zscore)), zscore, 1)[0]

            # Rolling stability check
            z_rolling_std = pd.Series(zscore).rolling(window=100).std().dropna()
            z_std_stability = z_rolling_std.std()

            print(f"\nPair: {stock1}/{stock2}")
            print(f"Z-Mean: {z_mean:.4f}")
            print(f"Z-Std: {z_std:.4f}")
            print(f"Slope: {slope:.6f}")
            print(f"Z-Std Stability: {z_std_stability:.4f}")

        #tiering system, may be useless...
        if pvalue < 0.01:
            tier = 'A'
        elif pvalue < 0.05:
            tier = 'B'
        else:
            tier = None

        if tier:
            coint_pairs.append((stock1, stock2))
            coint_stats.append({
                'beta': beta,
                'alpha': alpha,
                'pvalue': pvalue,
                'z_mean': z_mean,
                'z_std': z_std,
                'slope': slope,
                'z_std_stability': z_std_stability,
                'tier': tier
            })

# Plot price and spread behaiour of a pair of stocks
def plot_spread(stock1, stock2, beta, alpha, title=None):
    """Plot the spread for a stock pair"""
    s1 = close_prices[stock1]
    s2 = close_prices[stock2]
    
    # Recalculate spread and z-score
    spread = s1.values - (beta * s2.values + alpha)
    zscore = (spread - spread.mean()) / spread.std()
    zscore_series = pd.Series(zscore, index=close_prices.index)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Plot prices
    ax1.plot(s1.index, s1, label=stock1, linewidth=1)
    ax1.plot(s2.index, s2, label=stock2, linewidth=1)
    ax1.set_title(f"Stock Prices: {stock1} vs {stock2}")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot z-score
    ax2.plot(zscore_series.index, zscore_series, label="Z-Score", color='purple')
    ax2.axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label="Entry/Exit Threshold")
    ax2.axhline(y=-1.5, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title("Z-Score with Trading Thresholds")
    ax2.set_ylabel("Z-Score")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if title:
        fig.suptitle(title, fontsize=16)
        fig.subplots_adjust(top=0.9)
    
    plt.show()


#Simulate a trading strategy based on z-score thresholds
def quick_backtest(stock1, stock2, beta, alpha, entry_threshold=1.5, exit_threshold=0):
    """Do a quick backtest for a stock pair"""
    s1 = close_prices[stock1]
    s2 = close_prices[stock2]
    
    # Create DataFrame for strategy
    pair_df = pd.DataFrame(index=close_prices.index)
    pair_df['s1'] = s1
    pair_df['s2'] = s2
    
    # Calculate spread and z-score
    spread = s1.values - (beta * s2.values + alpha)
    pair_df['spread'] = spread
    pair_df['zscore'] = (spread - spread.mean()) / spread.std()
    
    # Trading signals
    pair_df['position'] = 0
    
    # Trading logic, we generate trading signals based on z-score crossing thresholds
    for i in range(1, len(pair_df)):
        z = pair_df['zscore'].iloc[i]
        prev_pos = pair_df['position'].iloc[i-1]
        
        # No position
        if prev_pos == 0:
            if z < -entry_threshold:
                pair_df.loc[pair_df.index[i], 'position'] = 1  # Long the spread
            elif z > entry_threshold:
                pair_df.loc[pair_df.index[i], 'position'] = -1  # Short the spread
            else:
                pair_df.loc[pair_df.index[i], 'position'] = 0
        
        # Long position
        elif prev_pos == 1:
            if z >= exit_threshold:
                pair_df.loc[pair_df.index[i], 'position'] = 0  # Exit
            else:
                pair_df.loc[pair_df.index[i], 'position'] = 1  # Stay long
        
        # Short position
        elif prev_pos == -1:
            if z <= exit_threshold:
                pair_df.loc[pair_df.index[i], 'position'] = 0  # Exit
            else:
                pair_df.loc[pair_df.index[i], 'position'] = -1  # Stay short
    
    # Calculate returns
    pair_df['s1_returns'] = pair_df['s1'].pct_change()
    pair_df['s2_returns'] = pair_df['s2'].pct_change()
    
    pair_df['strategy_returns'] = pair_df['position'].shift(1) * (
        pair_df['s1_returns'] - beta * pair_df['s2_returns']
    )
    
    pair_df['cum_strategy_returns'] = (1 + pair_df['strategy_returns'].fillna(0)).cumprod() - 1
    
    # Calculate metrics
    total_return = pair_df['cum_strategy_returns'].iloc[-1] * 100
    annual_return = ((1 + total_return / 100) ** (252 / len(pair_df)) - 1) * 100
    
    daily_returns = pair_df['strategy_returns'].dropna()
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
    
    max_dd = (pair_df['cum_strategy_returns'] - pair_df['cum_strategy_returns'].cummax()).min() * 100
    
    # Plot results
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 15), sharex=True)
    
    # Plot z-score and positions
    ax1.plot(pair_df.index, pair_df['zscore'], label='Z-Score', color='gray')
    ax1.scatter(pair_df[pair_df['position'] == 1].index, 
               pair_df['zscore'][pair_df['position'] == 1], 
               color='green', label='Long', marker='^', s=50)
    ax1.scatter(pair_df[pair_df['position'] == -1].index, 
               pair_df['zscore'][pair_df['position'] == -1], 
               color='red', label='Short', marker='v', s=50)
    ax1.axhline(y=entry_threshold, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(y=-entry_threshold, color='g', linestyle='--', alpha=0.5)
    ax1.axhline(y=exit_threshold, color='k', linestyle='--', alpha=0.3)
    ax1.set_title('Z-Score and Trading Positions')
    ax1.set_ylabel('Z-Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot positions over time
    ax2.plot(pair_df.index, pair_df['position'], label='Position', 
            drawstyle='steps-post', color='purple')
    ax2.set_title('Position Over Time')
    ax2.set_ylabel('Position')
    ax2.grid(True, alpha=0.3)
    
    # Plot cumulative returns
    ax3.plot(pair_df.index, pair_df['cum_strategy_returns'] * 100, 
            label='Strategy Returns (%)', color='blue')
    ax3.set_title(f'Cumulative Returns (%) - Sharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}%')
    ax3.set_ylabel('Returns (%)')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    title = f"{stock1}/{stock2} Backtest - Annual Return: {annual_return:.2f}%, Sharpe: {sharpe:.2f}"
    fig.suptitle(title, fontsize=16)
    fig.subplots_adjust(top=0.95)
    
    plt.show()
    
    results = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'num_trades': (pair_df['position'] != pair_df['position'].shift(1)).sum() - 1  # Exclude first NaN
    }
    
    return results

#run
find_pairs_coint()

# Run backtest and plot the pairs
print(f"Cointegration method found {len(coint_pairs)} pairs")

if coint_pairs:
    print("\n=== Testing first cointegration pair ===")
    for i in range(len(coint_pairs)):
        stock1, stock2 = coint_pairs[i]
        stats = coint_stats[i]
    
        # Plot the spread
        plot_spread(stock1, stock2, stats['beta'], stats['alpha'], 
                f"Cointegration Method: {stock1}/{stock2}")
        
        # Quick backtest
        results = quick_backtest(stock1, stock2, stats['beta'], stats['alpha'])
        print(f"Backtest results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
#!/usr/bin/env python3
"""
Paper Trading Backtest with Real Market Data

Uses yfinance to fetch real market data and tests the 100k RL model
performance over a full year of historical trading (2024).
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our gym and training components
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

class PaperTradingBacktester:
    """Comprehensive paper trading backtest system."""

    def __init__(self, model_path="./models/qwen_final_model_100k"):
        self.model_path = model_path
        self.model = None
        self.results = {}

    def load_model(self):
        """Load the trained RL model."""
        if not os.path.exists(self.model_path + ".zip"):
            raise FileNotFoundError(f"Model not found at {self.model_path}.zip")

        print(f"[LOAD] Loading RL model from {self.model_path}")
        self.model = PPO.load(self.model_path)
        print(f"[OK] Model loaded successfully")
        return True

    def fetch_market_data(self, symbol, start_date, end_date):
        """Fetch real market data using yfinance."""
        print(f"[DATA] Fetching {symbol} data from {start_date} to {end_date}")

        try:
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data found for {symbol}")

            print(f"[OK] Downloaded {len(data)} trading days of data")
            print(f"  - Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
            print(f"  - Average volume: {data['Volume'].mean():,.0f}")

            return data

        except Exception as e:
            print(f"[ERROR] Failed to fetch data: {e}")
            # Fallback to synthetic data for testing
            print(f"[FALLBACK] Using synthetic data for {symbol}")
            return self.generate_synthetic_data(symbol, start_date, end_date)

    def generate_synthetic_data(self, symbol, start_date, end_date):
        """Generate synthetic data when real data is unavailable."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')

        # Filter to trading days (remove weekends)
        trading_days = dates[dates.weekday < 5]

        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)
        initial_price = 150.0
        prices = [initial_price]

        for i in range(1, len(trading_days)):
            daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily return, 2% volatility
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1.0))

        # Create synthetic OHLCV data
        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.lognormal(15, 1, len(prices)).astype(int)
        }, index=trading_days)

        return data

    def create_realistic_environment(self, data, symbol, initial_cash=100000):
        """Create trading environment with real market data."""

        # Calculate real volatility
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility

        # Create asset configuration with real data
        asset = AssetConfig(
            symbol=symbol,
            name=self._get_company_name(symbol),
            sector=self._get_sector(symbol),
            initial_price=data['Close'].iloc[0],
            volatility=volatility
        )

        print(f"[ENV] Creating environment for {symbol}")
        print(f"  - Initial price: ${asset.initial_price:.2f}")
        print(f"  - Annualized volatility: {volatility:.2%}")

        # Create custom environment that uses real data
        env = RealMarketEnvironment(
            asset=asset,
            market_data=data,
            initial_cash=initial_cash,
            lookback_window=30
        )

        return env

    def _get_company_name(self, symbol):
        """Get company name for symbol."""
        names = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'NVDA': 'NVIDIA Corporation',
            'META': 'Meta Platforms Inc.'
        }
        return names.get(symbol, f'{symbol} Corporation')

    def _get_sector(self, symbol):
        """Get sector for symbol."""
        sectors = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'NVDA': 'Technology',
            'META': 'Technology'
        }
        return sectors.get(symbol, 'Technology')

    def run_backtest(self, symbol, start_date="2024-01-01", end_date="2024-12-31"):
        """Run comprehensive backtest on a single symbol."""

        print(f"\n{'='*80}")
        print(f"PAPER TRADING BACKTEST: {symbol}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Model: {self.model_path}")
        print(f"{'='*80}")

        # Fetch market data
        market_data = self.fetch_market_data(symbol, start_date, end_date)

        # Create environment with real data
        env = self.create_realistic_environment(market_data, symbol)

        # Run backtest
        print(f"\n[BACKTEST] Running paper trading simulation...")

        obs = env.reset()
        done = False
        day = 0

        portfolio_history = []
        action_history = []
        reward_history = []
        position_history = []
        price_history = []

        while not done and day < len(market_data) - 30:  # Leave room for lookback
            # Get model prediction
            if isinstance(obs, tuple):
                obs_array = obs[0]
            else:
                obs_array = obs

            if len(obs_array.shape) == 1:
                obs_array = obs_array.reshape(1, -1)

            action, _ = self.model.predict(obs_array, deterministic=True)
            action = action[0]  # Extract single action

            # Execute trade
            step_result = env.step(action)

            if len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                obs, reward, done = step_result[0], step_result[1], step_result[2]
                info = {}

            # Record results
            portfolio_history.append(env.portfolio_value)
            action_history.append(action)
            reward_history.append(reward)
            position_history.append(env.positions.get(symbol, 0))
            price_history.append(market_data['Close'].iloc[day + 30])  # Current price

            day += 1

            # Progress update
            if day % 50 == 0:
                current_return = (env.portfolio_value - 100000) / 100000
                print(f"  Day {day:4d}: Portfolio ${env.portfolio_value:,.2f} ({current_return:+.2%}) - Action: {['SELL', 'HOLD', 'BUY'][action]}")

        # Calculate final results
        final_portfolio = env.portfolio_value
        total_return = (final_portfolio - 100000) / 100000

        # Calculate performance metrics
        returns = pd.Series(portfolio_history).pct_change().dropna()

        results = {
            'symbol': symbol,
            'period': f"{start_date} to {end_date}",
            'trading_days': len(portfolio_history),
            'initial_portfolio': 100000,
            'final_portfolio': final_portfolio,
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(portfolio_history)) - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(portfolio_history),
            'portfolio_history': portfolio_history,
            'action_history': action_history,
            'price_history': price_history,
            'position_history': position_history,
            'benchmark_return': (market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0],
            'alpha': total_return - ((market_data['Close'].iloc[-1] - market_data['Close'].iloc[0]) / market_data['Close'].iloc[0])
        }

        self.results[symbol] = results
        return results

    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown."""
        portfolio_array = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        return np.max(drawdown)

    def generate_performance_report(self, symbol):
        """Generate comprehensive performance report."""

        if symbol not in self.results:
            print(f"[ERROR] No results found for {symbol}")
            return

        results = self.results[symbol]

        print(f"\n{'='*80}")
        print(f"PERFORMANCE REPORT: {symbol}")
        print(f"{'='*80}")

        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"  Initial Value:     ${results['initial_portfolio']:,.2f}")
        print(f"  Final Value:       ${results['final_portfolio']:,.2f}")
        print(f"  Total Return:      {results['total_return']:+.2%}")
        print(f"  Annualized Return: {results['annualized_return']:+.2%}")
        print(f"  Trading Days:      {results['trading_days']}")

        print(f"\nRISK METRICS:")
        print(f"  Volatility:        {results['volatility']:.2%}")
        print(f"  Sharpe Ratio:      {results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:      {results['max_drawdown']:.2%}")

        print(f"\nBENCHMARK COMPARISON:")
        print(f"  Buy & Hold Return: {results['benchmark_return']:+.2%}")
        print(f"  Alpha:             {results['alpha']:+.2%}")

        if results['alpha'] > 0:
            print(f"  [OUTPERFORMED] RL model beat buy-and-hold by {results['alpha']:.2%}")
        else:
            print(f"  [UNDERPERFORMED] Buy-and-hold beat RL model by {abs(results['alpha']):.2%}")

        print(f"\nTRADING ACTIVITY:")
        action_counts = pd.Series(results['action_history']).value_counts()
        for action, count in action_counts.items():
            action_name = ['SELL', 'HOLD', 'BUY'][action]
            percentage = count / len(results['action_history']) * 100
            print(f"  {action_name:4}: {count:4d} times ({percentage:5.1f}%)")

        # Create performance chart
        self._create_performance_chart(symbol, results)

        return results

    def _create_performance_chart(self, symbol, results):
        """Create performance visualization chart."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

            days = range(len(results['portfolio_history']))

            # Portfolio value over time
            ax1.plot(days, results['portfolio_history'], 'b-', linewidth=2, label='RL Model Portfolio')
            benchmark_values = [100000 * (1 + results['benchmark_return']) ** (d / len(days)) for d in days]
            ax1.plot(days, benchmark_values, 'r--', linewidth=2, label='Buy & Hold')
            ax1.set_title(f'{symbol} - Portfolio Performance Over Time')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Price and position
            ax2_twin = ax2.twinx()
            ax2.plot(days, results['price_history'], 'g-', linewidth=1, label='Stock Price')
            ax2_twin.plot(days, results['position_history'], 'b-', linewidth=1, label='Position', alpha=0.7)
            ax2.set_ylabel('Stock Price ($)', color='g')
            ax2_twin.set_ylabel('Position (Shares)', color='b')
            ax2.set_title('Stock Price and Position')
            ax2.grid(True, alpha=0.3)

            # Actions over time
            action_colors = {0: 'red', 1: 'gray', 2: 'green'}
            action_labels = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

            for i, action in enumerate(results['action_history']):
                ax3.scatter(i, action, c=action_colors[action], s=10, alpha=0.6)

            ax3.set_ylabel('Trading Action')
            ax3.set_xlabel('Trading Days')
            ax3.set_yticks([0, 1, 2])
            ax3.set_yticklabels(['SELL', 'HOLD', 'BUY'])
            ax3.set_title('Trading Actions Over Time')
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save chart
            chart_path = f"./{symbol}_backtest_performance.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\n[CHART] Performance chart saved to {chart_path}")

        except ImportError:
            print(f"\n[WARNING] Matplotlib not available - skipping chart generation")
        except Exception as e:
            print(f"\n[ERROR] Failed to create chart: {e}")

    def run_multiple_symbol_backtest(self, symbols=None):
        """Run backtest on multiple symbols."""

        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

        print(f"\n{'='*80}")
        print(f"MULTIPLE SYMBOL PAPER TRADING BACKTEST")
        print(f"Testing RL model on {len(symbols)} symbols for 2024")
        print(f"{'='*80}")

        all_results = {}

        for symbol in symbols:
            try:
                print(f"\n{'-'*60}")
                results = self.run_backtest(symbol)
                report = self.generate_performance_report(symbol)
                all_results[symbol] = results

            except Exception as e:
                print(f"[ERROR] Failed to backtest {symbol}: {e}")
                all_results[symbol] = None

        # Generate summary report
        self.generate_summary_report(all_results)

        return all_results

    def generate_summary_report(self, all_results):
        """Generate summary report across all symbols."""

        print(f"\n{'='*80}")
        print(f"SUMMARY REPORT - ALL SYMBOLS")
        print(f"{'='*80}")

        successful_results = {k: v for k, v in all_results.items() if v is not None}

        if not successful_results:
            print("[ERROR] No successful backtests to summarize")
            return

        # Create comparison table
        print(f"\nPERFORMANCE COMPARISON:")
        print(f"{'Symbol':<8} {'Return':<10} {'Sharpe':<8} {'Max DD':<10} {'Alpha':<10} {'Days':<8}")
        print("-" * 70)

        total_return = 0
        total_alpha = 0
        outperformed_count = 0

        for symbol, results in successful_results.items():
            return_str = f"{results['total_return']:+.2%}"
            sharpe_str = f"{results['sharpe_ratio']:.2f}"
            dd_str = f"{results['max_drawdown']:.2%}"
            alpha_str = f"{results['alpha']:+.2%}"
            days_str = str(results['trading_days'])

            print(f"{symbol:<8} {return_str:<10} {sharpe_str:<8} {dd_str:<10} {alpha_str:<10} {days_str:<8}")

            total_return += results['total_return']
            total_alpha += results['alpha']
            if results['alpha'] > 0:
                outperformed_count += 1

        # Summary statistics
        num_symbols = len(successful_results)
        avg_return = total_return / num_symbols
        avg_alpha = total_alpha / num_symbols
        win_rate = outperformed_count / num_symbols * 100

        print(f"\nSUMMARY STATISTICS:")
        print(f"  Symbols Tested:     {num_symbols}")
        print(f"  Average Return:     {avg_return:+.2%}")
        print(f"  Average Alpha:      {avg_alpha:+.2%}")
        print(f"  Win Rate (vs B&H):  {win_rate:.1f}%")
        print(f"  Outperformed:       {outperformed_count}/{num_symbols}")

        # Overall assessment
        if avg_alpha > 0.02:  # 2% average alpha
            print(f"\n[EXCELLENT] RL model shows strong outperformance (+{avg_alpha:.2%} avg alpha)")
        elif avg_alpha > 0:
            print(f"\n[GOOD] RL model shows modest outperformance (+{avg_alpha:.2%} avg alpha)")
        else:
            print(f"\n[NEEDS WORK] RL model underperforms buy-and-hold ({avg_alpha:.2%} avg alpha)")

class RealMarketEnvironment:
    """Custom environment that uses real market data."""

    def __init__(self, asset, market_data, initial_cash=100000, lookback_window=30):
        self.asset = asset
        self.market_data = market_data
        self.initial_cash = initial_cash
        self.lookback_window = lookback_window

        # Trading state
        self.current_step = 0
        self.portfolio_value = initial_cash
        self.cash_balance = initial_cash
        self.position = 0.0
        self.positions = {asset.symbol: 0.0}

        # History tracking
        self.portfolio_history = []
        self.price_history = market_data['Close'].tolist()

    def reset(self):
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.portfolio_value = self.initial_cash
        self.cash_balance = self.initial_cash
        self.position = 0.0
        self.positions = {self.asset.symbol: 0.0}
        self.portfolio_history = []

        return self._get_observation(), {}

    def step(self, action):
        """Execute one trading step."""

        current_price = self.price_history[self.current_step]

        # Execute action
        if action == 0:  # SELL
            if self.position > 0:
                shares_to_sell = min(self.position, 100)  # Sell in blocks of 100
                self.cash_balance += shares_to_sell * current_price
                self.position -= shares_to_sell

        elif action == 2:  # BUY
            max_shares = self.cash_balance / (current_price * 1.001)  # Account for commission
            shares_to_buy = min(int(max_shares / 100) * 100, 100)  # Buy in blocks of 100
            if shares_to_buy > 0:
                self.cash_balance -= shares_to_buy * current_price * 1.001
                self.position += shares_to_buy

        # Update positions
        self.positions[self.asset.symbol] = self.position

        # Update portfolio value
        position_value = self.position * current_price
        self.portfolio_value = self.cash_balance + position_value

        # Calculate reward (portfolio return)
        reward = (self.portfolio_value - self.initial_cash) / self.initial_cash * 0.01  # Scaled reward

        # Advance time
        self.current_step += 1

        # Check termination
        done = self.current_step >= len(self.price_history) - 1 or self.portfolio_value < self.initial_cash * 0.3

        # Get observation
        obs = self._get_observation()

        # Info
        info = {
            'portfolio_value': self.portfolio_value,
            'cash_balance': self.cash_balance,
            'position': self.position,
            'current_price': current_price,
            'total_return': (self.portfolio_value - self.initial_cash) / self.initial_cash
        }

        return obs, reward, done, False, info

    def _get_observation(self):
        """Get current observation with exactly 49 features to match trained model."""

        # Get price history (30 features)
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        price_history = self.price_history[start_idx:end_idx]

        # Normalize price history
        if len(price_history) > 0:
            current_price = price_history[-1]
            normalized_prices = [(p / current_price - 1) for p in price_history]
        else:
            normalized_prices = [0.0] * self.lookback_window

        # Pad if necessary to get exactly 30 price features
        while len(normalized_prices) < self.lookback_window:
            normalized_prices.insert(0, 0.0)

        features = normalized_prices.copy()  # 30 features

        # Add technical indicators (10 features)
        returns = np.diff(price_history) if len(price_history) > 1 else [0.0]

        # SMA_10
        if len(price_history) >= 10:
            sma_10 = np.mean(price_history[-10:])
            features.append((current_price / sma_10 - 1) if sma_10 > 0 else 0)
        else:
            features.append(0.0)

        # SMA_30
        if len(price_history) >= 30:
            sma_30 = np.mean(price_history[-30:])
            features.append((current_price / sma_30 - 1) if sma_30 > 0 else 0)
        else:
            features.append(0.0)

        # RSI_14
        if len(returns) >= 14:
            gains = np.where(returns > 0, returns, 0)
            losses = np.where(returns < 0, -returns, 0)
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            features.append((rsi - 50) / 50)  # Normalized to [-1, 1]
        else:
            features.append(0.0)

        # MACD (simplified)
        if len(price_history) >= 26:
            ema_12 = np.mean(price_history[-12:])
            ema_26 = np.mean(price_history[-26:])
            macd = (ema_12 - ema_26) / ema_26 if ema_26 > 0 else 0
            features.append(macd)
        else:
            features.append(0.0)

        # Bollinger Bands (simplified)
        if len(price_history) >= 20:
            bb_mean = np.mean(price_history[-20:])
            bb_std = np.std(price_history[-20:])
            bb_upper = bb_mean + 2 * bb_std
            bb_lower = bb_mean - 2 * bb_std
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            features.append(bb_position - 0.5)  # Normalized to [-0.5, 0.5]
            features.append((bb_upper - bb_lower) / current_price if current_price > 0 else 0)  # Band width
        else:
            features.append(0.0)  # BB position
            features.append(0.0)  # BB width

        # ATR_14 (simplified)
        if len(returns) >= 14:
            atr = np.std(returns[-14:]) * current_price
            features.append(atr / current_price if current_price > 0 else 0)
        else:
            features.append(0.0)

        # Volume ratio (placeholder)
        features.append(1.0)  # Normalized volume

        # Price momentum 5
        if len(returns) >= 5:
            momentum_5 = np.mean(returns[-5:])
            features.append(momentum_5)
        else:
            features.append(0.0)

        # Price momentum 20
        if len(returns) >= 20:
            momentum_20 = np.mean(returns[-20:])
            features.append(momentum_20)
        else:
            features.append(0.0)

        # Portfolio state (5 features)
        position_ratio = (self.position * current_price) / self.portfolio_value if self.portfolio_value > 0 else 0
        cash_ratio = self.cash_balance / self.portfolio_value if self.portfolio_value > 0 else 0

        # PnL calculation
        total_pnl = (self.portfolio_value - 100000) / 100000

        features.append(cash_ratio)  # Cash ratio
        features.append(position_ratio)  # Position ratio
        features.append(total_pnl)  # Total PnL
        features.append(self.position / 1000 if self.position != 0 else 0)  # Position size (normalized)
        features.append(np.log(self.portfolio_value / 100000))  # Log portfolio value

        # Market state (4 features)
        if len(returns) > 0:
            volatility = np.std(returns) if len(returns) > 1 else 0
            trend = np.mean(returns[-min(10, len(returns)):]) if len(returns) >= 3 else 0
            price_change = returns[-1] if len(returns) > 0 else 0
            volume_proxy = abs(returns[-1]) if len(returns) > 0 else 0
        else:
            volatility = trend = price_change = volume_proxy = 0

        features.append(volatility)  # Market volatility
        features.append(trend)  # Market trend
        features.append(price_change)  # Last price change
        features.append(volume_proxy)  # Volume proxy

        # Ensure we have exactly 49 features
        features_array = np.array(features, dtype=np.float32)
        if len(features_array) != 49:
            print(f"Warning: Got {len(features_array)} features, expected 49")
            # Pad or truncate to 49
            if len(features_array) < 49:
                features_array = np.pad(features_array, (0, 49 - len(features_array)), 'constant')
            else:
                features_array = features_array[:49]

        return features_array

def main():
    """Main paper trading backtest function."""

    print("PAPER TRADING BACKTEST WITH REAL MARKET DATA")
    print("Testing 100k RL model on 2024 market performance")
    print("=" * 80)

    try:
        # Initialize backtester
        backtester = PaperTradingBacktester()

        # Load model
        backtester.load_model()

        # Run single symbol test (AAPL for 2024)
        print(f"\n{'='*80}")
        print("SINGLE SYMBOL TEST: AAPL (2024)")
        print(f"{'='*80}")

        aapl_results = backtester.run_backtest('AAPL', '2024-01-01', '2024-12-31')
        backtester.generate_performance_report('AAPL')

        # Run multiple symbol tests
        print(f"\n{'='*80}")
        print("MULTIPLE SYMBOL TEST")
        print(f"{'='*80}")

        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        multi_results = backtester.run_multiple_symbol_backtest(symbols)

        print(f"\nPaper trading backtest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return 0

    except Exception as e:
        print(f"[ERROR] Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
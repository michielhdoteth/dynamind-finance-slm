#!/usr/bin/env python3
"""
Baseline Trading Strategies Comparison

Comprehensive comparison of RL model against traditional trading strategies
including buy & hold, moving averages, RSI, MACD, and random trading.
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

class BaselineStrategy:
    """Base class for trading strategies."""

    def __init__(self, name):
        self.name = name
        self.position = 0
        self.cash = 100000
        self.portfolio_value = 100000
        self.portfolio_history = []
        self.trade_history = []

    def initialize(self, data):
        """Initialize strategy with market data."""
        self.data = data
        self.current_step = 0

    def act(self, data_slice):
        """Generate trading action. Returns 0 (SELL), 1 (HOLD), or 2 (BUY)."""
        raise NotImplementedError

    def execute_trade(self, action, price):
        """Execute trade with given action and price."""

        # Commission and slippage
        commission_rate = 0.001
        slippage_rate = 0.0005

        if action == 0 and self.position > 0:  # SELL
            # Sell all shares
            sell_value = self.position * price * (1 - commission_rate - slippage_rate)
            self.cash += sell_value
            self.position = 0

        elif action == 2 and self.cash > 0:  # BUY
            # Buy with available cash
            buy_price = price * (1 + commission_rate + slippage_rate)
            max_shares = int(self.cash / buy_price / 100) * 100  # Round to nearest 100
            if max_shares > 0:
                self.cash -= max_shares * buy_price
                self.position += max_shares

        # Update portfolio value
        self.portfolio_value = self.cash + self.position * price
        self.portfolio_history.append(self.portfolio_value)
        self.trade_history.append({
            'step': self.current_step,
            'action': action,
            'price': price,
            'position': self.position,
            'portfolio_value': self.portfolio_value
        })

        self.current_step += 1

class BuyAndHoldStrategy(BaselineStrategy):
    """Simple buy and hold strategy."""

    def __init__(self):
        super().__init__("Buy & Hold")
        self.bought = False

    def act(self, data_slice):
        """Buy on first day, then hold."""
        if not self.bought and len(data_slice) > 0:
            self.bought = True
            return 2  # BUY
        return 1  # HOLD

class MovingAverageStrategy(BaselineStrategy):
    """Moving average crossover strategy."""

    def __init__(self, short_window=10, long_window=30):
        super().__init__(f"MA Crossover ({short_window}/{long_window})")
        self.short_window = short_window
        self.long_window = long_window

    def act(self, data_slice):
        """Generate signals based on MA crossover."""
        if len(data_slice) < self.long_window:
            return 1  # HOLD until we have enough data

        prices = data_slice['Close'].values
        short_ma = np.mean(prices[-self.short_window:])
        long_ma = np.mean(prices[-self.long_window:])

        current_position = self.position

        # Buy signal: short MA crosses above long MA
        if short_ma > long_ma and current_position == 0:
            return 2  # BUY

        # Sell signal: short MA crosses below long MA
        elif short_ma < long_ma and current_position > 0:
            return 0  # SELL

        return 1  # HOLD

class RSIStrategy(BaselineStrategy):
    """RSI-based trading strategy."""

    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        super().__init__(f"RSI ({oversold}/{overbought})")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought

    def calculate_rsi(self, prices):
        """Calculate RSI indicator."""
        if len(prices) < self.rsi_period + 1:
            return 50  # Neutral

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def act(self, data_slice):
        """Generate signals based on RSI."""
        if len(data_slice) < self.rsi_period + 1:
            return 1  # HOLD

        prices = data_slice['Close'].values
        rsi = self.calculate_rsi(prices)

        current_position = self.position

        # Buy signal: RSI oversold
        if rsi < self.oversold and current_position == 0:
            return 2  # BUY

        # Sell signal: RSI overbought
        elif rsi > self.overbought and current_position > 0:
            return 0  # SELL

        return 1  # HOLD

class MACDStrategy(BaselineStrategy):
    """MACD-based trading strategy."""

    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(f"MACD ({fast}/{slow}/{signal})")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def calculate_macd(self, prices):
        """Calculate MACD indicator."""
        if len(prices) < self.slow + self.signal:
            return 0, 0  # Neutral values

        # Calculate EMAs
        exp_fast = prices.ewm(span=self.fast).mean()
        exp_slow = prices.ewm(span=self.slow).mean()
        macd_line = exp_fast - exp_slow
        signal_line = macd_line.ewm(span=self.signal).mean()

        return macd_line.iloc[-1], signal_line.iloc[-1]

    def act(self, data_slice):
        """Generate signals based on MACD crossover."""
        if len(data_slice) < self.slow + self.signal:
            return 1  # HOLD

        prices = data_slice['Close']
        macd, signal = self.calculate_macd(prices)

        current_position = self.position

        # Buy signal: MACD crosses above signal
        if macd > signal and current_position == 0:
            return 2  # BUY

        # Sell signal: MACD crosses below signal
        elif macd < signal and current_position > 0:
            return 0  # SELL

        return 1  # HOLD

class RandomStrategy(BaselineStrategy):
    """Random trading strategy for comparison."""

    def __init__(self, buy_prob=0.3, sell_prob=0.3):
        super().__init__("Random Strategy")
        self.buy_prob = buy_prob
        self.sell_prob = sell_prob

    def act(self, data_slice):
        """Generate random trading actions."""
        rand = np.random.random()

        current_position = self.position

        if rand < self.buy_prob and current_position == 0:
            return 2  # BUY
        elif rand < self.buy_prob + self.sell_prob and current_position > 0:
            return 0  # SELL
        else:
            return 1  # HOLD

class MomentumStrategy(BaselineStrategy):
    """Momentum-based trading strategy."""

    def __init__(self, lookback=20, threshold=0.02):
        super().__init__(f"Momentum ({lookback}d)")
        self.lookback = lookback
        self.threshold = threshold

    def act(self, data_slice):
        """Generate signals based on price momentum."""
        if len(data_slice) < self.lookback + 1:
            return 1  # HOLD

        prices = data_slice['Close'].values
        current_price = prices[-1]
        past_price = prices[-self.lookback-1]

        momentum = (current_price - past_price) / past_price

        current_position = self.position

        # Buy signal: Positive momentum above threshold
        if momentum > self.threshold and current_position == 0:
            return 2  # BUY

        # Sell signal: Negative momentum below negative threshold
        elif momentum < -self.threshold and current_position > 0:
            return 0  # SELL

        return 1  # HOLD

class BaselineComparator:
    """Compare RL model against baseline strategies."""

    def __init__(self, rl_model_path="./models/qwen_final_model_100k"):
        self.rl_model_path = rl_model_path
        self.rl_model = None
        self.results = {}

    def load_rl_model(self):
        """Load the trained RL model."""
        if not os.path.exists(self.rl_model_path + ".zip"):
            raise FileNotFoundError(f"RL model not found at {self.rl_model_path}")

        print(f"[LOAD] Loading RL model from {self.rl_model_path}")
        self.rl_model = PPO.load(self.rl_model_path)
        print(f"[OK] RL model loaded successfully")

    def fetch_market_data(self, symbol, start_date, end_date):
        """Fetch market data using yfinance."""
        print(f"[DATA] Fetching {symbol} data from {start_date} to {end_date}")

        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if data.empty:
                raise ValueError(f"No data found for {symbol}")

            print(f"[OK] Downloaded {len(data)} trading days")
            return data

        except Exception as e:
            print(f"[ERROR] Failed to fetch data: {e}")
            # Use synthetic data for testing
            return self.generate_synthetic_data(symbol, start_date, end_date)

    def generate_synthetic_data(self, symbol, start_date, end_date):
        """Generate synthetic market data."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start=start, end=end, freq='D')
        trading_days = dates[dates.weekday < 5]

        np.random.seed(hash(symbol) % 2**32)
        initial_price = 150.0
        prices = [initial_price]

        for i in range(1, len(trading_days)):
            daily_return = np.random.normal(0.0005, 0.02)
            new_price = prices[-1] * (1 + daily_return)
            prices.append(max(new_price, 1.0))

        data = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.lognormal(15, 1, len(prices)).astype(int)
        }, index=trading_days)

        return data

    def test_strategy(self, strategy, data, symbol):
        """Test a strategy on given data."""
        print(f"\n[TEST] Testing {strategy.name} on {symbol}")

        # Initialize strategy
        strategy.initialize(data)

        # Run strategy on historical data
        for i in range(30, len(data)):  # Start from day 30 for indicators
            data_slice = data.iloc[:i+1]
            action = strategy.act(data_slice)
            current_price = data.iloc[i]['Close']
            strategy.execute_trade(action, current_price)

        # Calculate performance metrics
        final_value = strategy.portfolio_value
        total_return = (final_value - 100000) / 100000

        # Calculate risk metrics
        returns = pd.Series(strategy.portfolio_history).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Calculate max drawdown
        portfolio_array = np.array(strategy.portfolio_history)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        # Calculate benchmark return
        benchmark_return = (data.iloc[-1]['Close'] - data.iloc[30]['Close']) / data.iloc[30]['Close']

        results = {
            'strategy_name': strategy.name,
            'symbol': symbol,
            'final_value': final_value,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_return,
            'alpha': total_return - benchmark_return,
            'portfolio_history': strategy.portfolio_history,
            'trade_history': strategy.trade_history
        }

        print(f"  Return: {total_return:+.2%}, Sharpe: {sharpe_ratio:.3f}, Alpha: {results['alpha']:+.2%}")

        return results

    def test_rl_model(self, data, symbol):
        """Test RL model on market data."""
        print(f"\n[TEST] Testing RL model on {symbol}")

        # Create environment for RL model
        from paper_trading_backtest import RealMarketEnvironment

        asset = AssetConfig(
            symbol=symbol,
            name=f"{symbol} Corp",
            sector="Technology",
            initial_price=data.iloc[0]['Close'],
            volatility=data['Close'].pct_change().std() * np.sqrt(252)
        )

        env = RealMarketEnvironment(
            asset=asset,
            market_data=data,
            initial_cash=100000,
            lookback_window=30
        )

        # Run RL model
        obs = env.reset()
        done = False

        portfolio_history = []
        action_history = []

        while not done and env.current_step < len(data) - 30:
            # Get RL model prediction
            if isinstance(obs, tuple):
                obs_array = obs[0]
            else:
                obs_array = obs

            if len(obs_array.shape) == 1:
                obs_array = obs_array.reshape(1, -1)

            action, _ = self.rl_model.predict(obs_array, deterministic=True)
            action = action[0]

            # Execute action
            step_result = env.step(action)

            if len(step_result) == 4:
                obs, reward, done, info = step_result
            elif len(step_result) == 5:
                obs, reward, done, truncated, info = step_result
                done = done or truncated
            else:
                obs, reward, done = step_result[0], step_result[1], step_result[2]
                info = {}

            portfolio_history.append(env.portfolio_value)
            action_history.append(action)

        # Calculate performance metrics
        final_value = env.portfolio_value
        total_return = (final_value - 100000) / 100000

        returns = pd.Series(portfolio_history).pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        portfolio_array = np.array(portfolio_history)
        peak = np.maximum.accumulate(portfolio_array)
        drawdown = (peak - portfolio_array) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

        benchmark_return = (data.iloc[-1]['Close'] - data.iloc[30]['Close']) / data.iloc[30]['Close']

        results = {
            'strategy_name': 'RL Model (100k)',
            'symbol': symbol,
            'final_value': final_value,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_return,
            'alpha': total_return - benchmark_return,
            'portfolio_history': portfolio_history,
            'action_history': action_history
        }

        print(f"  Return: {total_return:+.2%}, Sharpe: {sharpe_ratio:.3f}, Alpha: {results['alpha']:+.2%}")

        return results

    def run_comprehensive_comparison(self, symbols=None):
        """Run comparison across multiple symbols and strategies."""

        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

        print(f"COMPREHENSIVE BASELINE COMPARISON")
        print(f"Testing RL model vs traditional strategies")
        print(f"Symbols: {', '.join(symbols)}")
        print("=" * 80)

        # Initialize strategies
        strategies = [
            BuyAndHoldStrategy(),
            MovingAverageStrategy(10, 30),
            RSIStrategy(14, 30, 70),
            MACDStrategy(12, 26, 9),
            MomentumStrategy(20, 0.02),
            RandomStrategy(0.3, 0.3)
        ]

        all_results = {}

        # Load RL model
        self.load_rl_model()

        for symbol in symbols:
            print(f"\n{'='*60}")
            print(f"TESTING SYMBOL: {symbol}")
            print(f"{'='*60}")

            # Fetch market data
            data = self.fetch_market_data(symbol, '2024-01-01', '2024-12-31')

            symbol_results = {}

            # Test all baseline strategies
            for strategy in strategies:
                try:
                    result = self.test_strategy(strategy, data, symbol)
                    symbol_results[strategy.name] = result
                except Exception as e:
                    print(f"[ERROR] {strategy.name} failed: {e}")

            # Test RL model
            try:
                rl_result = self.test_rl_model(data, symbol)
                symbol_results['RL Model (100k)'] = rl_result
            except Exception as e:
                print(f"[ERROR] RL model failed: {e}")

            all_results[symbol] = symbol_results

        # Generate comparison report
        self.generate_comparison_report(all_results)

        return all_results

    def generate_comparison_report(self, all_results):
        """Generate comprehensive comparison report."""

        print(f"\n{'='*80}")
        print("BASELINE COMPARISON REPORT")
        print(f"{'='*80}")

        # Aggregate results across symbols
        strategy_performance = {}

        for symbol, symbol_results in all_results.items():
            for strategy_name, result in symbol_results.items():
                if strategy_name not in strategy_performance:
                    strategy_performance[strategy_name] = {
                        'returns': [],
                        'sharpe_ratios': [],
                        'max_drawdowns': [],
                        'alphas': []
                    }

                strategy_performance[strategy_name]['returns'].append(result['total_return'])
                strategy_performance[strategy_name]['sharpe_ratios'].append(result['sharpe_ratio'])
                strategy_performance[strategy_name]['max_drawdowns'].append(result['max_drawdown'])
                strategy_performance[strategy_name]['alphas'].append(result['alpha'])

        # Calculate averages
        summary_stats = {}
        for strategy_name, metrics in strategy_performance.items():
            summary_stats[strategy_name] = {
                'avg_return': np.mean(metrics['returns']),
                'avg_sharpe': np.mean(metrics['sharpe_ratios']),
                'avg_drawdown': np.mean(metrics['max_drawdowns']),
                'avg_alpha': np.mean(metrics['alphas']),
                'win_rate': sum(1 for a in metrics['alphas'] if a > 0) / len(metrics['alphas'])
            }

        # Sort by average return
        sorted_strategies = sorted(summary_stats.items(), key=lambda x: x[1]['avg_return'], reverse=True)

        print(f"\nRANKING BY AVERAGE RETURN:")
        print("-" * 70)
        print(f"{'Rank':<5} {'Strategy':<25} {'Return':<10} {'Sharpe':<8} {'Alpha':<10} {'Win Rate':<10}")
        print("-" * 70)

        for rank, (strategy_name, stats) in enumerate(sorted_strategies, 1):
            print(f"{rank:<5} {strategy_name:<25} {stats['avg_return']:+9.2%} "
                  f"{stats['avg_sharpe']:7.3f} {stats['avg_alpha']:+9.2%} "
                  f"{stats['win_rate']*100:8.1f}%")

        # Detailed symbol breakdown
        print(f"\nDETAILED SYMBOL BREAKDOWN:")
        for symbol, symbol_results in all_results.items():
            print(f"\n{symbol}:")
            print("-" * 50)

            sorted_symbol = sorted(symbol_results.items(),
                                 key=lambda x: x[1]['total_return'], reverse=True)

            for strategy_name, result in sorted_symbol:
                print(f"  {strategy_name:<25}: {result['total_return']:+7.2%} "
                      f"(Sharpe: {result['sharpe_ratio']:.2f}, Alpha: {result['alpha']:+.2%})")

        # RL model analysis
        if 'RL Model (100k)' in summary_stats:
            rl_stats = summary_stats['RL Model (100k)']

            print(f"\nRL MODEL ANALYSIS:")
            print("-" * 40)
            print(f"Average Return: {rl_stats['avg_return']:+.2%}")
            print(f"Average Sharpe: {rl_stats['avg_sharpe']:.3f}")
            print(f"Average Alpha: {rl_stats['avg_alpha']:+.2%}")
            print(f"Win Rate: {rl_stats['win_rate']*100:.1f}%")

            # Find best baseline
            baseline_rank = [i for i, (name, _) in enumerate(sorted_strategies, 1)
                           if name == 'RL Model (100k)'][0]

            if baseline_rank == 1:
                print(f"🏆 RL Model RANKS #1 out of {len(sorted_strategies)} strategies!")
            else:
                print(f"RL Model ranks #{baseline_rank} out of {len(sorted_strategies)} strategies")

            # Compare against best baseline
            best_strategy = sorted_strategies[0]
            if best_strategy[0] != 'RL Model (100k)':
                diff = rl_stats['avg_return'] - best_strategy[1]['avg_return']
                print(f"Gap vs best: {diff:+.2%} ({best_strategy[0]})")

        # Risk analysis
        print(f"\nRISK ANALYSIS:")
        print("-" * 30)
        sorted_by_sharpe = sorted(summary_stats.items(),
                                key=lambda x: x[1]['avg_sharpe'], reverse=True)

        print("Top 3 by Sharpe Ratio:")
        for i, (strategy_name, stats) in enumerate(sorted_by_sharpe[:3], 1):
            print(f"  {i}. {strategy_name}: {stats['avg_sharpe']:.3f}")

        # Create comparison chart
        self.create_comparison_chart(summary_stats)

        return summary_stats

    def create_comparison_chart(self, summary_stats):
        """Create comparison visualization."""
        try:
            strategies = list(summary_stats.keys())
            returns = [summary_stats[s]['avg_return'] * 100 for s in strategies]
            sharpe_ratios = [summary_stats[s]['avg_sharpe'] for s in strategies]
            alphas = [summary_stats[s]['avg_alpha'] * 100 for s in strategies]

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

            # Returns chart
            bars1 = ax1.bar(strategies, returns, color='skyblue')
            ax1.set_title('Average Return by Strategy')
            ax1.set_ylabel('Return (%)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)

            # Highlight RL model
            for i, strategy in enumerate(strategies):
                if 'RL Model' in strategy:
                    bars1[i].set_color('orange')
                    bars1[i].set_edgecolor('red')
                    bars1[i].set_linewidth(2)

            # Sharpe ratio chart
            bars2 = ax2.bar(strategies, sharpe_ratios, color='lightgreen')
            ax2.set_title('Average Sharpe Ratio by Strategy')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, alpha=0.3)

            # Highlight RL model
            for i, strategy in enumerate(strategies):
                if 'RL Model' in strategy:
                    bars2[i].set_color('orange')
                    bars2[i].set_edgecolor('red')
                    bars2[i].set_linewidth(2)

            # Alpha chart
            colors = ['green' if a > 0 else 'red' for a in alphas]
            bars3 = ax3.bar(strategies, alphas, color=colors, alpha=0.7)
            ax3.set_title('Average Alpha vs Buy & Hold')
            ax3.set_ylabel('Alpha (%)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)

            # Highlight RL model
            for i, strategy in enumerate(strategies):
                if 'RL Model' in strategy:
                    bars3[i].set_edgecolor('red')
                    bars3[i].set_linewidth(3)

            plt.tight_layout()

            # Save chart
            chart_path = "./baseline_comparison_chart.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\n[CHART] Comparison chart saved to {chart_path}")

        except Exception as e:
            print(f"[ERROR] Failed to create chart: {e}")

def main():
    """Main comparison function."""

    print("BASELINE TRADING STRATEGIES COMPARISON")
    print("Comparing RL model against traditional trading strategies")
    print("=" * 80)

    try:
        comparator = BaselineComparator()
        results = comparator.run_comprehensive_comparison()

        print(f"\nBaseline comparison completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
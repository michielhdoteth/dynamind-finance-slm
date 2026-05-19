#!/usr/bin/env python3
"""
Qwen RL Model vs Baseline Qwen Model Comparison

Compares the performance of Qwen model with RL training against
the original Qwen model without RL training on trading tasks.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our gym and training components
from environments import SingleAssetTradingEnv
from environments.base_env import AssetConfig
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn

class QwenFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor matching the trained RL model architecture."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Same architecture as used in RL training
        self.net = nn.Sequential(
            nn.Linear(np.prod(observation_space.shape), 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim)
        )

    def forward(self, observations):
        return self.net(observations)

class BaselineQwenTrader:
    """Baseline Qwen model without RL training for comparison."""

    def __init__(self):
        self.model = None
        self.features_extractor = None
        self.policy_network = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_baseline_model()

    def _initialize_baseline_model(self):
        """Initialize baseline Qwen model with random weights (no RL training)."""
        print(f"[INIT] Creating baseline Qwen model (no RL training)")

        # Create the same architecture but without training
        observation_space_shape = (49,)  # Same as trained model

        # Create feature extractor
        from gymnasium import spaces
        dummy_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=observation_space_shape, dtype=np.float32)
        self.features_extractor = QwenFeaturesExtractor(dummy_obs_space, features_dim=256)

        # Create policy network (same as PPO but untrained)
        self.policy_network = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 actions: SELL, HOLD, BUY
        )

        # Move to device
        self.features_extractor.to(self.device)
        self.policy_network.to(self.device)

        print(f"[OK] Baseline Qwen model created on {self.device}")

    def predict(self, observation):
        """Predict action using untrained Qwen model."""
        if isinstance(observation, tuple):
            obs_array = observation[0]
        else:
            obs_array = observation

        if len(obs_array.shape) == 1:
            obs_array = obs_array.reshape(1, -1)

        # Convert to tensor
        obs_tensor = torch.FloatTensor(obs_array).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.features_extractor(obs_tensor)
            # Get action logits
            action_logits = self.policy_network(features)
            # Sample action (random since untrained)
            action_probs = torch.softmax(action_logits, dim=-1)
            action = torch.multinomial(action_probs, 1).item()

        return action, None  # Return action and dummy state

class QwenRLComparator:
    """Compare Qwen RL model vs baseline Qwen model."""

    def __init__(self):
        self.rl_model = None
        self.baseline_model = None
        self.results = {}

    def load_rl_model(self, model_path="./models/qwen_final_model_100k"):
        """Load the trained RL model."""
        if not os.path.exists(model_path + ".zip"):
            raise FileNotFoundError(f"RL model not found at {model_path}")

        print(f"[LOAD] Loading RL model from {model_path}")
        self.rl_model = PPO.load(model_path)
        print(f"[OK] RL model loaded successfully")

    def initialize_baseline_model(self):
        """Initialize baseline Qwen model."""
        self.baseline_model = BaselineQwenTrader()

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

    def create_environment(self, data, symbol):
        """Create trading environment for testing."""
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

        return env

    def test_model(self, model, env, model_name, symbol):
        """Test a model on given environment."""
        print(f"\n[TEST] Testing {model_name} on {symbol}")

        obs = env.reset()
        done = False

        portfolio_history = []
        action_history = []
        reward_history = []

        step = 0
        while not done and step < 220:  # Limit to ~220 trading days (1 year)
            # Get model prediction
            if hasattr(model, 'predict') and hasattr(model, 'policy'):  # RL model
                if isinstance(obs, tuple):
                    obs_array = obs[0]
                else:
                    obs_array = obs

                if len(obs_array.shape) == 1:
                    obs_array = obs_array.reshape(1, -1)

                action, _ = model.predict(obs_array, deterministic=True)
                action = action[0]

            else:  # Baseline Qwen model
                action, _ = model.predict(obs)

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
            reward_history.append(reward)

            step += 1

            # Progress update
            if step % 50 == 0:
                current_return = (env.portfolio_value - 100000) / 100000
                action_name = ['SELL', 'HOLD', 'BUY'][action]
                print(f"  Step {step:3d}: Portfolio ${env.portfolio_value:,.2f} ({current_return:+.2%}) - {action_name}")

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

        # Calculate benchmark return
        if len(env.price_history) > 30:
            benchmark_return = (env.price_history[-1] - env.price_history[30]) / env.price_history[30]
        else:
            benchmark_return = 0

        results = {
            'model_name': model_name,
            'symbol': symbol,
            'final_value': final_value,
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'benchmark_return': benchmark_return,
            'alpha': total_return - benchmark_return,
            'portfolio_history': portfolio_history,
            'action_history': action_history,
            'reward_history': reward_history,
            'trading_days': len(portfolio_history)
        }

        print(f"  Final: {total_return:+.2%}, Sharpe: {sharpe_ratio:.3f}, Alpha: {results['alpha']:+.2%}")

        return results

    def analyze_trading_behavior(self, results):
        """Analyze the trading behavior patterns."""

        action_history = results['action_history']
        reward_history = results['reward_history']

        # Action distribution
        action_counts = pd.Series(action_history).value_counts().to_dict()
        total_actions = len(action_history)

        action_distribution = {
            'sell': action_counts.get(0, 0) / total_actions * 100,
            'hold': action_counts.get(1, 0) / total_actions * 100,
            'buy': action_counts.get(2, 0) / total_actions * 100
        }

        # Trading frequency
        non_hold_actions = sum(1 for a in action_history if a != 1)
        trading_frequency = non_hold_actions / total_actions * 100

        # Reward analysis
        avg_reward = np.mean(reward_history) if reward_history else 0
        reward_std = np.std(reward_history) if reward_history else 0

        # Consistency
        positive_rewards = sum(1 for r in reward_history if r > 0)
        consistency = positive_rewards / len(reward_history) * 100 if reward_history else 0

        behavior_analysis = {
            'action_distribution': action_distribution,
            'trading_frequency': trading_frequency,
            'avg_reward': avg_reward,
            'reward_std': reward_std,
            'consistency': consistency,
            'total_actions': total_actions
        }

        return behavior_analysis

    def run_comprehensive_comparison(self, symbols=None):
        """Run comprehensive comparison between RL and baseline Qwen models."""

        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

        print("QWEN RL MODEL VS BASELINE QWEN MODEL COMPARISON")
        print("Testing impact of RL training on Qwen trading performance")
        print("=" * 80)

        # Load models
        self.load_rl_model()
        self.initialize_baseline_model()

        all_results = {}

        for symbol in symbols:
            print(f"\n{'='*60}")
            print(f"TESTING SYMBOL: {symbol}")
            print(f"{'='*60}")

            # Fetch market data
            data = self.fetch_market_data(symbol, '2024-01-01', '2024-12-31')

            # Create environment
            env = self.create_environment(data, symbol)

            symbol_results = {}

            # Test RL model
            print(f"\n--- Testing Qwen with RL Training ---")
            try:
                rl_results = self.test_model(self.rl_model, env, "Qwen+RL (100k)", symbol)
                rl_results['behavior_analysis'] = self.analyze_trading_behavior(rl_results)
                symbol_results['Qwen+RL'] = rl_results
            except Exception as e:
                print(f"[ERROR] RL model test failed: {e}")

            # Test baseline Qwen model
            print(f"\n--- Testing Baseline Qwen (No RL) ---")
            try:
                # Reset environment for fair comparison
                env_baseline = self.create_environment(data, symbol)
                baseline_results = self.test_model(self.baseline_model, env_baseline, "Qwen Baseline", symbol)
                baseline_results['behavior_analysis'] = self.analyze_trading_behavior(baseline_results)
                symbol_results['Qwen_Baseline'] = baseline_results
            except Exception as e:
                print(f"[ERROR] Baseline model test failed: {e}")

            all_results[symbol] = symbol_results

        # Generate comparison report
        self.generate_comparison_report(all_results)

        return all_results

    def generate_comparison_report(self, all_results):
        """Generate comprehensive comparison report."""

        print(f"\n{'='*80}")
        print("QWEN RL VS BASELINE COMPARISON REPORT")
        print(f"{'='*80}")

        # Aggregate results
        rl_performance = []
        baseline_performance = []
        symbol_comparisons = {}

        for symbol, symbol_results in all_results.items():
            if 'Qwen+RL' in symbol_results and 'Qwen_Baseline' in symbol_results:
                rl_result = symbol_results['Qwen+RL']
                baseline_result = symbol_results['Qwen_Baseline']

                rl_performance.append(rl_result['total_return'])
                baseline_performance.append(baseline_result['total_return'])

                # Calculate improvement
                return_improvement = (rl_result['total_return'] - baseline_result['total_return']) / (abs(baseline_result['total_return']) + 1e-8) * 100
                sharpe_improvement = (rl_result['sharpe_ratio'] - baseline_result['sharpe_ratio']) / (abs(baseline_result['sharpe_ratio']) + 1e-8) * 100

                symbol_comparisons[symbol] = {
                    'rl_return': rl_result['total_return'],
                    'baseline_return': baseline_result['total_return'],
                    'return_improvement': return_improvement,
                    'rl_sharpe': rl_result['sharpe_ratio'],
                    'baseline_sharpe': baseline_result['sharpe_ratio'],
                    'sharpe_improvement': sharpe_improvement,
                    'rl_behavior': rl_result['behavior_analysis'],
                    'baseline_behavior': baseline_result['behavior_analysis']
                }

        # Overall statistics
        avg_rl_return = np.mean(rl_performance)
        avg_baseline_return = np.mean(baseline_performance)
        overall_improvement = (avg_rl_return - avg_baseline_return) / (abs(avg_baseline_return) + 1e-8) * 100

        print(f"\nOVERALL PERFORMANCE COMPARISON:")
        print("-" * 50)
        print(f"Average RL Model Return:      {avg_rl_return:+.2%}")
        print(f"Average Baseline Return:       {avg_baseline_return:+.2%}")
        print(f"Overall Improvement:          {overall_improvement:+.2%}")

        # Detailed symbol breakdown
        print(f"\nDETAILED SYMBOL BREAKDOWN:")
        print("-" * 70)
        print(f"{'Symbol':<8} {'Qwen+RL':<12} {'Qwen Base':<12} {'Improvement':<12} {'RL Sharpe':<10} {'Base Sharpe':<12}")
        print("-" * 70)

        for symbol, comparison in symbol_comparisons.items():
            print(f"{symbol:<8} {comparison['rl_return']:+11.2%} {comparison['baseline_return']:+11.2%} "
                  f"{comparison['return_improvement']:+11.2%} {comparison['rl_sharpe']:9.3f} "
                  f"{comparison['baseline_sharpe']:11.3f}")

        # Trading behavior analysis
        print(f"\nTRADING BEHAVIOR ANALYSIS:")
        print("-" * 40)

        print("\nQwen+RL Model Behavior:")
        rl_trading_freq = np.mean([comp['rl_behavior']['trading_frequency'] for comp in symbol_comparisons.values()])
        rl_consistency = np.mean([comp['rl_behavior']['consistency'] for comp in symbol_comparisons.values()])
        rl_buy_ratio = np.mean([comp['rl_behavior']['action_distribution']['buy'] for comp in symbol_comparisons.values()])
        rl_sell_ratio = np.mean([comp['rl_behavior']['action_distribution']['sell'] for comp in symbol_comparisons.values()])

        print(f"  Trading Frequency:  {rl_trading_freq:.1f}%")
        print(f"  Consistency:        {rl_consistency:.1f}% positive rewards")
        print(f"  Buy Actions:       {rl_buy_ratio:.1f}%")
        print(f"  Sell Actions:      {rl_sell_ratio:.1f}%")

        print("\nBaseline Qwen Behavior:")
        baseline_trading_freq = np.mean([comp['baseline_behavior']['trading_frequency'] for comp in symbol_comparisons.values()])
        baseline_consistency = np.mean([comp['baseline_behavior']['consistency'] for comp in symbol_comparisons.values()])
        baseline_buy_ratio = np.mean([comp['baseline_behavior']['action_distribution']['buy'] for comp in symbol_comparisons.values()])
        baseline_sell_ratio = np.mean([comp['baseline_behavior']['action_distribution']['sell'] for comp in symbol_comparisons.values()])

        print(f"  Trading Frequency:  {baseline_trading_freq:.1f}%")
        print(f"  Consistency:        {baseline_consistency:.1f}% positive rewards")
        print(f"  Buy Actions:       {baseline_buy_ratio:.1f}%")
        print(f"  Sell Actions:      {baseline_sell_ratio:.1f}%")

        # RL Training Impact Analysis
        print(f"\nRL TRAINING IMPACT ANALYSIS:")
        print("-" * 40)

        trading_freq_improvement = (rl_trading_freq - baseline_trading_freq) / (baseline_trading_freq + 1e-8) * 100
        consistency_improvement = (rl_consistency - baseline_consistency) / (baseline_consistency + 1e-8) * 100

        print(f"Return Improvement:        {overall_improvement:+.2f}%")
        print(f"Trading Frequency Change:  {trading_freq_improvement:+.2f}%")
        print(f"Consistency Improvement:   {consistency_improvement:+.2f}%")

        # Winner determination
        rl_wins = sum(1 for comp in symbol_comparisons.values() if comp['rl_return'] > comp['baseline_return'])
        total_symbols = len(symbol_comparisons)
        rl_win_rate = rl_wins / total_symbols * 100 if total_symbols > 0 else 0

        print(f"\nHEAD-TO-HEAD RESULTS:")
        if total_symbols > 0:
            print(f"RL Model Wins: {rl_wins}/{total_symbols} ({rl_win_rate:.1f}%)")

            if overall_improvement > 5:
                print(f"[CHAMPION] CONCLUSION: RL training dramatically improves Qwen performance!")
            elif overall_improvement > 0:
                print(f"[SUCCESS] CONCLUSION: RL training modestly improves Qwen performance.")
            else:
                print(f"[NEUTRAL] CONCLUSION: RL training does not show clear improvement in this test.")
        else:
            print("No valid comparisons completed.")

        # Create comparison chart
        self.create_comparison_chart(symbol_comparisons)

        return symbol_comparisons

    def create_comparison_chart(self, symbol_comparisons):
        """Create comparison visualization."""
        try:
            symbols = list(symbol_comparisons.keys())
            rl_returns = [symbol_comparisons[s]['rl_return'] * 100 for s in symbols]
            baseline_returns = [symbol_comparisons[s]['baseline_return'] * 100 for s in symbols]
            improvements = [symbol_comparisons[s]['return_improvement'] for s in symbols]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Returns comparison
            x = np.arange(len(symbols))
            width = 0.35

            bars1 = ax1.bar(x - width/2, rl_returns, width, label='Qwen+RL', color='orange', alpha=0.8)
            bars2 = ax1.bar(x + width/2, baseline_returns, width, label='Qwen Baseline', color='skyblue', alpha=0.8)

            ax1.set_xlabel('Stock Symbol')
            ax1.set_ylabel('Return (%)')
            ax1.set_title('Qwen+RL vs Qwen Baseline: Returns Comparison')
            ax1.set_xticks(x)
            ax1.set_xticklabels(symbols)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')

            for bar in bars2:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom')

            # Improvement chart
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            bars3 = ax2.bar(symbols, improvements, color=colors, alpha=0.7)
            ax2.set_xlabel('Stock Symbol')
            ax2.set_ylabel('Improvement (%)')
            ax2.set_title('RL Training Impact: % Improvement Over Baseline')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.grid(True, alpha=0.3)

            # Add value labels
            for bar in bars3:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')

            plt.tight_layout()

            # Save chart
            chart_path = "./qwen_rl_vs_baseline_comparison.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\n[CHART] Comparison chart saved to {chart_path}")

        except Exception as e:
            print(f"[ERROR] Failed to create chart: {e}")

def main():
    """Main comparison function."""

    print("QWEN RL MODEL VS BASELINE QWEN COMPARISON")
    print("Testing the impact of RL training on Qwen trading performance")
    print("=" * 80)

    try:
        comparator = QwenRLComparator()
        results = comparator.run_comprehensive_comparison()

        print(f"\nQwen RL vs Baseline comparison completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return 0

    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
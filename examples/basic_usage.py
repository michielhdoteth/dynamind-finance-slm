"""
Basic Usage Examples for Financial Trading Gym

This script demonstrates how to use the various environments in the
financial trading gym package with different configurations and algorithms.
"""

import warnings
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

import os

# Import our gym environments
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic import create_synthetic_data
from environments import (
    MarketMakingEnv,
    PortfolioOptimizationEnv,
    RegimeDetectionEnv,
    SingleAssetTradingEnv,
)
from environments.base_env import AssetConfig, RiskConstraints, TransactionCosts


def create_sample_assets() -> List[AssetConfig]:
    """Create sample asset configurations"""
    return [
        AssetConfig(
            symbol="AAPL",
            name="Apple Inc.",
            sector="Technology",
            initial_price=150.0,
            volatility=0.025,
            drift=0.0002,
            market_cap=3e12,
            avg_daily_volume=50e6,
        ),
        AssetConfig(
            symbol="MSFT",
            name="Microsoft Corp.",
            sector="Technology",
            initial_price=300.0,
            volatility=0.022,
            drift=0.00015,
            market_cap=2.5e12,
            avg_daily_volume=30e6,
        ),
        AssetConfig(
            symbol="JPM",
            name="JPMorgan Chase",
            sector="Financial",
            initial_price=150.0,
            volatility=0.020,
            drift=0.0001,
            market_cap=500e9,
            avg_daily_volume=15e6,
        ),
        AssetConfig(
            symbol="JNJ",
            name="Johnson & Johnson",
            sector="Healthcare",
            initial_price=160.0,
            volatility=0.015,
            drift=0.00008,
            market_cap=450e9,
            avg_daily_volume=10e6,
        ),
        AssetConfig(
            symbol="XOM",
            name="Exxon Mobil",
            sector="Energy",
            initial_price=100.0,
            volatility=0.028,
            drift=0.00005,
            market_cap=400e9,
            avg_daily_volume=20e6,
        ),
    ]


def demo_single_asset_trading():
    """Demonstrate single asset trading environment"""
    print("\n" + "=" * 60)
    print("SINGLE ASSET TRADING ENVIRONMENT DEMO")
    print("=" * 60)

    # Create single asset
    asset = create_sample_assets()[0]  # AAPL

    # Create environment
    env = SingleAssetTradingEnv(
        asset=asset,
        initial_cash=1_000_000,
        max_episode_length=252,
        lookback_window=30,
        action_space_type="discrete",  # or "continuous"
        seed=42,
    )

    print(f"Environment created for {asset.symbol}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action meanings: {env.get_action_meanings()}")

    # Run a random episode
    obs, info = env.reset(seed=42)
    done = False
    step = 0
    portfolio_values = []
    actions_taken = []
    rewards = []

    print(f"\nStarting episode with initial portfolio: ${info['portfolio_value']:,.2f}")

    while not done and step < 100:  # Limit steps for demo
        # Random action
        action = env.action_space.sample()
        actions_taken.append(action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        rewards.append(reward)

        step += 1
        done = terminated or truncated

        if step % 20 == 0:
            print(
                f"Step {step}: Portfolio=${info['portfolio_value']:,.2f}, "
                f"Return={info['total_return']:+.2%}, Action={action}"
            )

    # Print final statistics
    final_stats = env.get_portfolio_stats()
    print(f"\nEpisode completed after {step} steps")
    print(f"Final portfolio value: ${info['portfolio_value']:,.2f}")
    print(f"Total return: {info['total_return']:+.2%}")
    if final_stats:
        print(f"Sharpe ratio: {final_stats.get('sharpe_ratio', 0):.3f}")
        print(f"Max drawdown: {final_stats.get('max_drawdown', 0):.2%}")

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(rewards)
    plt.title("Rewards Over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    env.close()


def demo_portfolio_optimization():
    """Demonstrate portfolio optimization environment"""
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION ENVIRONMENT DEMO")
    print("=" * 60)

    # Create multiple assets
    assets = create_sample_assets()

    # Create environment
    env = PortfolioOptimizationEnv(
        assets=assets,
        initial_cash=1_000_000,
        max_episode_length=252,
        lookback_window=60,
        rebalance_mode="discrete",
        seed=42,
    )

    print(f"Environment created with {len(assets)} assets")
    print(f"Assets: {[asset.symbol for asset in assets]}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space shape: {env.observation_space.shape}")

    # Run a random episode
    obs, info = env.reset(seed=42)
    done = False
    step = 0
    portfolio_values = []
    weight_history = []

    print("\nStarting portfolio optimization episode")

    while not done and step < 100:  # Limit steps for demo
        # Random portfolio weights
        action = np.random.random(len(assets))
        action = action / np.sum(action)  # Normalize

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

        # Get current weights
        current_weights = [env.current_weights.get(asset.symbol, 0) for asset in assets]
        weight_history.append(current_weights)

        step += 1
        done = terminated or truncated

        if step % 20 == 0:
            top_holdings = sorted(
                [(assets[i].symbol, current_weights[i]) for i in range(len(assets))],
                key=lambda x: x[1],
                reverse=True,
            )[:3]
            print(
                f"Step {step}: Portfolio=${info['portfolio_value']:,.2f}, "
                f"Top holdings: {top_holdings}"
            )

    # Final portfolio composition
    final_weights = [env.current_weights.get(asset.symbol, 0) for asset in assets]

    print("\nFinal portfolio composition:")
    for i, asset in enumerate(assets):
        if final_weights[i] > 0.01:  # Only show >1% positions
            print(f"  {asset.symbol}: {final_weights[i]:.2%}")

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    weight_matrix = np.array(weight_history).T
    plt.stackplot(
        range(len(weight_history)),
        weight_matrix,
        labels=[asset.symbol for asset in assets],
    )
    plt.title("Portfolio Weights Over Time")
    plt.xlabel("Step")
    plt.ylabel("Weight")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    final_weights_sorted = sorted(
        [(assets[i].symbol, final_weights[i]) for i in range(len(assets))],
        key=lambda x: x[1],
        reverse=True,
    )
    symbols, weights = zip(*final_weights_sorted)
    plt.bar(symbols, weights)
    plt.title("Final Portfolio Weights")
    plt.xlabel("Asset")
    plt.ylabel("Weight")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    env.close()


def demo_regime_detection():
    """Demonstrate regime detection environment"""
    print("\n" + "=" * 60)
    print("REGIME DETECTION ENVIRONMENT DEMO")
    print("=" * 60)

    # Create assets
    assets = create_sample_assets()[:3]  # Use subset for demo

    # Create environment
    env = RegimeDetectionEnv(
        assets=assets,
        initial_cash=1_000_000,
        max_episode_length=504,  # 2 years for regime detection
        lookback_window=60,
        action_mode="combined",  # Predict regime + trade
        seed=42,
    )

    print("Environment created for regime detection")
    print(f"Number of regimes: {len(env.config.regime_types)}")
    print(f"Regime types: {[regime.name for regime in env.config.regime_types]}")
    print(f"Action space: {env.action_space}")

    # Run a random episode
    obs, info = env.reset(seed=42)
    done = False
    step = 0
    portfolio_values = []
    predictions = []
    true_regimes = []

    print("\nStarting regime detection episode")

    while not done and step < 150:  # Limit steps for demo
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

        # Store regime information
        if (
            hasattr(env, "predicted_regime_history")
            and len(env.predicted_regime_history) > step
        ):
            predictions.append(env.predicted_regime_history[step])
            true_regimes.append(env.market_data["true_regime"].iloc[step])

        step += 1
        done = terminated or truncated

        if step % 30 == 0:
            print(
                f"Step {step}: Portfolio=${info['portfolio_value']:,.2f}, "
                f"Return={info['total_return']:+.2%}"
            )

    # Calculate prediction accuracy
    if predictions and true_regimes:
        accuracy = sum(1 for p, t in zip(predictions, true_regimes) if p == t) / len(
            predictions
        )
        print(f"\nRegime prediction accuracy: {accuracy:.2%}")

    # Get regime statistics
    regime_stats = env.get_regime_statistics()
    print(f"Overall prediction accuracy: {regime_stats['overall_accuracy']:.2%}")
    print("Performance by regime:")
    for regime_name, perf in regime_stats["performance_by_regime"].items():
        print(
            f"  {regime_name}: Sharpe={perf['sharpe_ratio']:.3f}, "
            f"Samples={perf['samples']}"
        )

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    if predictions and true_regimes:
        plt.plot(true_regimes[: len(predictions)], label="True Regime", linewidth=2)
        plt.plot(predictions, label="Predicted Regime", linestyle="--", alpha=0.7)
        plt.title("Regime Detection")
        plt.xlabel("Step")
        plt.ylabel("Regime")
        plt.legend()
        plt.grid(True)

    plt.subplot(1, 3, 3)
    regime_names = list(regime_stats["regime_counts"].keys())
    regime_counts = list(regime_stats["regime_counts"].values())
    plt.bar(regime_names, regime_counts)
    plt.title("Regime Distribution")
    plt.xlabel("Regime")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    env.close()


def demo_market_making():
    """Demonstrate market making environment"""
    print("\n" + "=" * 60)
    print("MARKET MAKING ENVIRONMENT DEMO")
    print("=" * 60)

    # Create single asset for market making
    asset = create_sample_assets()[0]  # AAPL

    # Create environment
    env = MarketMakingEnv(
        asset=asset,
        initial_cash=100_000,
        max_episode_length=1000,
        lookback_window=50,
        order_book_depth=5,
        seed=42,
    )

    print(f"Environment created for market making {asset.symbol}")
    print(f"Action space: {env.action_space}")
    print(f"Action meanings: {env.get_action_meanings()}")

    # Run a random episode
    obs, info = env.reset(seed=42)
    done = False
    step = 0
    portfolio_values = []
    spreads = []
    inventory_levels = []

    print("\nStarting market making episode")

    while not done and step < 200:  # Limit steps for demo
        # Random action
        action = env.action_space.sample()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        inventory_levels.append(env.inventory)

        step += 1
        done = terminated or truncated

        if step % 50 == 0:
            print(
                f"Step {step}: Portfolio=${info['portfolio_value']:,.2f}, "
                f"Inventory={env.inventory}, Return={info['total_return']:+.2%}"
            )

    # Get market making statistics
    mm_stats = env.get_market_making_stats()
    print("\nMarket Making Statistics:")
    print(f"Total trades: {mm_stats['total_trades']}")
    print(f"Fill rate: {mm_stats['fill_rate']:.2%}")
    print(f"Average spread: {mm_stats['average_spread']:.4f}")
    print(f"Inventory utilization: {mm_stats['inventory_utilization']:.2%}")

    if "sharpe_ratio" in mm_stats:
        print(f"Sharpe ratio: {mm_stats['sharpe_ratio']:.3f}")

    # Plot results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(portfolio_values)
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Step")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(inventory_levels)
    plt.title("Inventory Level Over Time")
    plt.xlabel("Step")
    plt.ylabel("Inventory (shares)")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    if "pnl_breakdown" in mm_stats:
        pnl_components = mm_stats["pnl_breakdown"]
        components = list(pnl_components.keys())
        values = list(pnl_components.values())
        plt.bar(components, values)
        plt.title("P&L Components")
        plt.xlabel("Component")
        plt.ylabel("P&L ($)")
        plt.xticks(rotation=45)
        plt.grid(True)

    plt.tight_layout()
    plt.show()

    env.close()


def demo_environment_comparison():
    """Compare performance across different environments"""
    print("\n" + "=" * 60)
    print("ENVIRONMENT COMPARISON DEMO")
    print("=" * 60)

    # Create assets
    assets = create_sample_assets()

    # Environment configurations
    environments = {
        "Single Asset": SingleAssetTradingEnv(
            asset=assets[0], initial_cash=1_000_000, max_episode_length=100, seed=42
        ),
        "Portfolio": PortfolioOptimizationEnv(
            assets=assets[:3], initial_cash=1_000_000, max_episode_length=100, seed=42
        ),
        "Regime Detection": RegimeDetectionEnv(
            assets=assets[:2], initial_cash=1_000_000, max_episode_length=100, seed=42
        ),
        "Market Making": MarketMakingEnv(
            asset=assets[0], initial_cash=100_000, max_episode_length=100, seed=42
        ),
    }

    # Run episodes and collect results
    results = {}
    for env_name, env in environments.items():
        print(f"\nTesting {env_name} environment...")

        obs, info = env.reset(seed=42)
        done = False
        portfolio_values = []
        rewards = []

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            portfolio_values.append(info["portfolio_value"])
            rewards.append(reward)
            done = terminated or truncated

        # Calculate statistics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[
            0
        ]
        volatility = np.std(np.diff(portfolio_values)) / np.mean(portfolio_values)
        sharpe_ratio = np.mean(rewards) / (np.std(rewards) + 1e-8)

        results[env_name] = {
            "total_return": total_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "final_value": portfolio_values[-1],
            "portfolio_values": portfolio_values,
        }

        print(f"  Total return: {total_return:+.2%}")
        print(f"  Sharpe ratio: {sharpe_ratio:.3f}")
        print(f"  Final value: ${portfolio_values[-1]:,.2f}")

        env.close()

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Environment Performance Comparison", fontsize=16)

    # Portfolio values over time
    ax = axes[0, 0]
    for env_name, result in results.items():
        ax.plot(result["portfolio_values"], label=env_name, linewidth=2)
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Step")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    ax.grid(True)

    # Total returns
    ax = axes[0, 1]
    env_names = list(results.keys())
    returns = [results[name]["total_return"] for name in env_names]
    colors = ["green" if r > 0 else "red" for r in returns]
    ax.bar(env_names, returns, color=colors, alpha=0.7)
    ax.set_title("Total Returns")
    ax.set_ylabel("Return")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True)

    # Sharpe ratios
    ax = axes[1, 0]
    sharpe_ratios = [results[name]["sharpe_ratio"] for name in env_names]
    colors = ["green" if s > 0 else "red" for s in sharpe_ratios]
    ax.bar(env_names, sharpe_ratios, color=colors, alpha=0.7)
    ax.set_title("Sharpe Ratios")
    ax.set_ylabel("Sharpe Ratio")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True)

    # Risk vs Return scatter
    ax = axes[1, 1]
    returns = [results[name]["total_return"] for name in env_names]
    volatilities = [results[name]["volatility"] for name in env_names]
    ax.scatter(
        volatilities, returns, s=100, alpha=0.7, c=range(len(env_names)), cmap="viridis"
    )
    for i, name in enumerate(env_names):
        ax.annotate(
            name,
            (volatilities[i], returns[i]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Total Return")
    ax.set_title("Risk vs Return Profile")
    ax.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """Run all demonstrations"""
    print("FINANCIAL TRADING GYM - COMPREHENSIVE DEMO")
    print("=" * 60)

    try:
        # Run individual demos
        demo_single_asset_trading()
        demo_portfolio_optimization()
        demo_regime_detection()
        demo_market_making()
        demo_environment_comparison()

        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

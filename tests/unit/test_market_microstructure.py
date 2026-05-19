"""
Test Market Microstructure Environments

This script tests the new market microstructure environments including
LOB simulation, execution optimization, and market-making strategies.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_lob_simulator():
    """Test the Limit Order Book simulator."""
    print("\n1. Testing Limit Order Book Simulator...")

    try:
        from environments.market_microstructure import (
            LimitOrderBook,
            Order,
            OrderSide,
            OrderType,
        )

        # Create LOB
        lob = LimitOrderBook(tick_size=0.01, base_spread=0.02)

        # Add initial liquidity
        lob.add_liquidity(num_orders=10, price_range=0.5)

        print("   Initial LOB state:")
        state = lob.get_order_book_state()
        print(f"     Best bid: {state['best_bid']:.2f}")
        print(f"     Best ask: {state['best_ask']:.2f}")
        print(f"     Mid price: {state['mid_price']:.2f}")
        print(f"     Spread: {state['spread']:.2f}")
        print(f"     Bid volume: {state['bid_volume']}")
        print(f"     Ask volume: {state['ask_volume']}")

        # Test market order execution
        market_order = Order(
            order_id="test_market_buy",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=500,
            trader_id="test_trader",
        )

        trades = lob.place_order(market_order)
        print(f"   Market order executed: {len(trades)} trades")

        if trades:
            trade = trades[0]
            print(f"     Trade price: {trade.price:.2f}")
            print(f"     Trade quantity: {trade.quantity}")

        # Test limit order placement
        limit_order = Order(
            order_id="test_limit_sell",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=300,
            price=101.0,
            trader_id="test_trader",
        )

        trades = lob.place_order(limit_order)
        print(f"   Limit order placed: {len(trades)} immediate trades")

        # Get market depth
        depth = lob.get_market_depth(levels=3)
        print("   Market depth:")
        print(f"     Bid levels: {len(depth['bids'])}")
        print(f"     Ask levels: {len(depth['asks'])}")

        if depth["bids"]:
            print(f"     Best bid level: {depth['bids'][0]}")
        if depth["asks"]:
            print(f"     Best ask level: {depth['asks'][0]}")

        print("   LOB simulator test passed!")
        return True

    except Exception as e:
        print(f"   LOB simulator test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_execution_environment():
    """Test the Execution environment."""
    print("\n2. Testing Execution Environment...")

    try:
        from environments.base_env import AssetConfig
        from environments.market_microstructure import (
            ExecutionEnvironment,
            ExecutionOrder,
            OrderSide,
        )

        # Create asset
        asset = AssetConfig(
            symbol="AAPL", name="Apple Inc.", sector="Technology", initial_price=150.0
        )

        # Create execution environment
        exec_env = ExecutionEnvironment(
            assets=[asset],
            initial_cash=1_000_000,
            max_episode_length=50,
            market_impact_factor=0.001,
            seed=42,
        )

        print("   Execution environment created!")
        print(f"     Action space: {exec_env.action_space}")
        print(f"     Observation space: {exec_env.observation_space.shape}")

        # Add execution order
        execution_order = ExecutionOrder(
            symbol="AAPL",
            side=OrderSide.BUY,
            total_quantity=1000,
            urgency=0.5,
            max_participation_rate=0.2,
            time_horizon=10,
        )

        exec_env.add_execution_order(execution_order)
        print("   Added execution order: Buy 1000 AAPL")

        # Reset environment
        obs, info = exec_env.reset(seed=42)
        print("   Environment reset successful")
        print(f"     Initial observation shape: {obs.shape}")
        print(f"     Portfolio value: ${info['portfolio_value']:,.2f}")

        # Test a few steps
        total_reward = 0
        for step in range(5):
            action = exec_env.action_space.sample()
            obs, reward, terminated, truncated, info = exec_env.step(action)
            total_reward += reward

            print(
                f"   Step {step+1}: Reward={reward:.4f}, Portfolio=${info['portfolio_value']:,.2f}"
            )

            if terminated or truncated:
                break

        print(f"   Total reward: {total_reward:.4f}")

        # Get execution summary
        summary = exec_env.get_execution_summary()
        print("   Execution summary:")
        print(f"     Completed orders: {len(summary['completed_orders'])}")
        print(f"     Pending orders: {len(summary['pending_orders'])}")
        print(f"     Execution metrics: {summary['execution_metrics']}")

        exec_env.close()
        print("   Execution environment test passed!")
        return True

    except Exception as e:
        print(f"   Execution environment test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_market_making_environment():
    """Test the Market Making environment."""
    print("\n3. Testing Market Making Environment...")

    try:
        from environments.base_env import AssetConfig
        from environments.market_microstructure import (
            MarketMakingConfig,
            MarketMakingEnvironment,
        )

        # Create assets
        assets = [
            AssetConfig(
                symbol="AAPL",
                name="Apple Inc.",
                sector="Technology",
                initial_price=150.0,
            ),
            AssetConfig(
                symbol="MSFT",
                name="Microsoft",
                sector="Technology",
                initial_price=250.0,
            ),
        ]

        # Create market-making configuration
        config = MarketMakingConfig(
            base_spread=0.02, max_position=500, quote_size=100, max_quotes=3
        )

        # Create market-making environment
        mm_env = MarketMakingEnvironment(
            assets=assets,
            initial_cash=2_000_000,
            max_episode_length=100,
            config=config,
            enable_risk_management=True,
            seed=42,
        )

        print("   Market making environment created!")
        print(f"     Action space: {mm_env.action_space}")
        print(f"     Observation space: {mm_env.observation_space.shape}")
        print(f"     Number of assets: {len(assets)}")
        print(f"     Base spread: {config.base_spread}")
        print(f"     Max position: {config.max_position}")

        # Reset environment
        obs, info = mm_env.reset(seed=42)
        print("   Environment reset successful")
        print(f"     Initial observation shape: {obs.shape}")
        print(f"     Portfolio value: ${info['portfolio_value']:,.2f}")

        # Test a few steps
        total_reward = 0
        for step in range(10):
            # Use a more intelligent action for market making
            action = np.array([0.1, 0.0, 0.0, 1.0])  # Tighten spread, fixed strategy
            obs, reward, terminated, truncated, info = mm_env.step(action)
            total_reward += reward

            if step % 3 == 0:  # Print every 3 steps
                print(
                    f"   Step {step+1}: Reward={reward:.4f}, Portfolio=${info['portfolio_value']:,.2f}"
                )
                print(f"     Active quotes: {info['active_quotes']}")
                print(f"     Total trades: {info['total_trades']}")

            if terminated or truncated:
                break

        print(f"   Total reward: {total_reward:.4f}")

        # Get market making summary
        summary = mm_env.get_market_making_summary()
        print("   Market making summary:")
        print(f"     Total P&L: ${summary['performance']['total_pnl']:.2f}")
        print(f"     Sharpe ratio: {summary['performance']['sharpe_ratio']:.4f}")
        print(f"     Max drawdown: {summary['performance']['max_drawdown']:.4f}")
        print(f"     Current positions: {summary['inventory']['current_positions']}")
        print(f"     Total trades: {summary['trading']['total_trades']}")

        mm_env.close()
        print("   Market making environment test passed!")
        return True

    except Exception as e:
        print(f"   Market making environment test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_integration():
    """Test integration between different market microstructure components."""
    print("\n4. Testing Integration...")

    try:
        from environments.base_env import AssetConfig
        from environments.market_microstructure import (
            ExecutionEnvironment,
            ExecutionOrder,
            LimitOrderBook,
            MarketMakingConfig,
            MarketMakingEnvironment,
            Order,
            OrderSide,
            OrderType,
        )

        # Create a simple market scenario
        asset = AssetConfig(
            symbol="TEST", name="Test Asset", sector="Technology", initial_price=100.0
        )

        # Create LOB
        lob = LimitOrderBook(tick_size=0.01)
        lob.add_liquidity(num_orders=5, price_range=0.2)

        print("   Created LOB with initial liquidity")

        # Simulate some market activity
        for i in range(5):
            side = OrderSide.BUY if i % 2 == 0 else OrderSide.SELL
            market_order = Order(
                order_id=f"market_{i}",
                side=side,
                order_type=OrderType.MARKET,
                quantity=np.random.randint(50, 200),
                trader_id="external",
            )
            trades = lob.place_order(market_order)
            if trades:
                print(f"     Trade {i+1}: {trades[0].quantity} @ {trades[0].price:.2f}")

        # Create market maker
        mm_config = MarketMakingConfig(quote_size=50, max_position=200)

        # Place some market-making orders
        mid_price = lob.get_mid_price()
        spread = 0.02

        bid_order = Order(
            order_id="mm_bid",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=mid_price - spread / 2,
            trader_id="market_maker",
        )

        ask_order = Order(
            order_id="mm_ask",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=50,
            price=mid_price + spread / 2,
            trader_id="market_maker",
        )

        bid_trades = lob.place_order(bid_order)
        ask_trades = lob.place_order(ask_order)

        print("   Market maker quotes placed:")
        print(f"     Bid trades: {len(bid_trades)}")
        print(f"     Ask trades: {len(ask_trades)}")

        # Final LOB state
        final_state = lob.get_order_book_state()
        print("   Final LOB state:")
        print(f"     Mid price: {final_state['mid_price']:.2f}")
        print(f"     Spread: {final_state['spread']:.2f}")
        print(f"     Total volume: {final_state['total_volume']}")
        print(f"     Total trades: {final_state['total_trades']}")

        print("   Integration test passed!")
        return True

    except Exception as e:
        print(f"   Integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("TESTING MARKET MICROSTRUCTURE ENVIRONMENTS")
    print("=" * 70)

    all_tests_passed = True

    # Run all tests
    tests = [
        test_lob_simulator,
        test_execution_environment,
        test_market_making_environment,
        test_integration,
    ]

    for test_func in tests:
        if not test_func():
            all_tests_passed = False

    print("\n" + "=" * 70)
    if all_tests_passed:
        print("ALL MARKET MICROSTRUCTURE TESTS PASSED!")
        print("=" * 70)

        print("\nMarket Microstructure Environments Ready!")
        print("\nFeatures implemented:")
        print("1. Limit Order Book (LOB) simulator")
        print("2. Order execution with market impact")
        print("3. Execution optimization (TWAP, VWAP, adaptive)")
        print("4. Market making with inventory management")
        print("5. Risk management and position limits")
        print("6. Realistic market dynamics")

        print("\nReady for:")
        print("• Advanced execution algorithm training")
        print("• Market making strategy optimization")
        print("• High-frequency trading research")
        print("• Market microstructure analysis")

    else:
        print("SOME TESTS FAILED!")
        print("=" * 70)

    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

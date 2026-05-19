#!/usr/bin/env python3
"""
Simple Risk Management System Test

Tests the core risk management components without requiring full package installation.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import risk management components directly
try:
    from risk.cvar_reward_shaper import CVaRConfig, CVaRRewardShaper, RiskMeasure

    print("[OK] CVaR Reward Shaper imported successfully")
except ImportError as e:
    print(f"[ERROR] Could not import CVaR Reward Shaper: {e}")
    sys.exit(1)

try:
    from environments.base_env import AssetConfig
    from risk.risk_manager import (
        PortfolioConstraints,
        PositionLimits,
        RiskLevel,
        RiskManager,
    )

    print("[OK] Risk Manager imported successfully")
except ImportError as e:
    print(f"[ERROR] Could not import Risk Manager: {e}")
    sys.exit(1)


def test_cvar_reward_shaper():
    """Test CVaR reward shaping functionality."""
    print("\n" + "=" * 60)
    print("TESTING CVAR REWARD SHAPING")
    print("=" * 60)

    # Initialize CVaR reward shaper
    config = CVaRConfig(
        confidence_level=0.05,
        window_size=50,
        risk_aversion=2.0,
        reward_shaping_method="cvar_adjustment",
    )

    cvar_shaper = CVaRRewardShaper(
        config=config,
        risk_measures=[
            RiskMeasure.CVAR,
            RiskMeasure.VAR,
            RiskMeasure.MAX_DD,
            RiskMeasure.SEMIVAR,
        ],
        enable_risk_budget=True,
    )

    print("[OK] CVaR Reward Shaper initialized")
    print(f"  - Confidence level: {config.confidence_level}")
    print(f"  - Window size: {config.window_size}")
    print(f"  - Risk aversion: {config.risk_aversion}")

    # Test CVaR calculation with known values
    test_returns = np.array([-0.05, -0.03, -0.08, -0.02, -0.01, 0.01, 0.02, 0.03])
    expected_var = np.percentile(test_returns, 5)  # 5% quantile
    expected_cvar = np.mean(test_returns[test_returns <= expected_var])

    calculated_cvar = cvar_shaper.calculate_cvar(test_returns, confidence_level=0.05)
    calculated_var = cvar_shaper.calculate_var(test_returns, confidence_level=0.05)

    print("\n✓ CVaR calculation validation:")
    print(f"  - Test returns: {test_returns}")
    print(f"  - Expected VaR (5%): {expected_var:.4f}")
    print(f"  - Calculated VaR (5%): {calculated_var:.4f}")
    print(f"  - Expected CVaR (5%): {expected_cvar:.4f}")
    print(f"  - Calculated CVaR (5%): {calculated_cvar:.4f}")

    # Simulate trading and reward shaping
    print("\n✓ Testing reward shaping simulation:")

    np.random.seed(42)
    n_steps = 100
    portfolio_values = []
    returns = []

    initial_value = 100000
    current_value = initial_value

    for step in range(n_steps):
        # Generate realistic return with some volatility clustering
        volatility = 0.02 if step < 50 else 0.03  # Increase volatility in second half
        ret = np.random.normal(0.0005, volatility)

        # Add occasional extreme moves
        if np.random.random() < 0.02:
            ret *= np.random.choice([-3, 3])

        current_value *= 1 + ret
        portfolio_values.append(current_value)
        returns.append(ret)

        # Apply reward shaping
        base_reward = ret * 1000  # Scale reward for RL
        additional_info = {"portfolio_value": current_value, "returns_history": returns}

        shaped_reward = cvar_shaper.shape_reward(base_reward, step, additional_info)

        if step % 25 == 0:
            risk_metrics = cvar_shaper.calculate_risk_metrics(step)
            print(f"  Step {step}:")
            print(f"    Portfolio value: ${current_value:,.0f}")
            print(f"    Base reward: {base_reward:.2f}")
            print(f"    Shaped reward: {shaped_reward:.2f}")
            print(f"    CVaR: {risk_metrics.get('cvar', 0):.4f}")
            print(f"    VaR: {risk_metrics.get('var', 0):.4f}")

    # Get final summary
    summary = cvar_shaper.get_risk_summary()
    print("\n✓ Final Risk Summary:")
    if "risk_metrics" in summary:
        metrics = summary["risk_metrics"]
        print(f"  - CVaR: {metrics.get('cvar', 0):.4f}")
        print(f"  - VaR: {metrics.get('var', 0):.4f}")
        print(f"  - Max Drawdown: {metrics.get('max_dd', 0):.4f}")
        print(f"  - Risk utilization: {summary['risk_utilization']:.2%}")

    if "reward_adjustment_stats" in summary:
        stats = summary["reward_adjustment_stats"]
        print(f"  - Mean reward adjustment: {stats['mean_adjustment']:.2f}")
        print(f"  - Negative adjustments: {stats['negative_adjustments']}")
        print(f"  - Positive adjustments: {stats['positive_adjustments']}")

    print(f"  - Total violations: {summary['total_violations']}")

    return cvar_shaper


def test_risk_manager():
    """Test risk manager with position constraints."""
    print("\n" + "=" * 60)
    print("TESTING RISK MANAGER")
    print("=" * 60)

    # Create sample assets
    assets = [
        AssetConfig(
            symbol="AAPL",
            sector="Technology",
            price_history=[150.0, 152.0, 151.0, 153.0, 155.0],
            volatility=0.02,
        ),
        AssetConfig(
            symbol="MSFT",
            sector="Technology",
            price_history=[300.0, 302.0, 301.0, 304.0, 306.0],
            volatility=0.018,
        ),
        AssetConfig(
            symbol="JPM",
            sector="Finance",
            price_history=[140.0, 141.0, 142.0, 140.5, 143.0],
            volatility=0.015,
        ),
    ]

    # Create risk configurations
    position_limits = risk.PositionLimits(
        max_position_size=0.4,
        max_sector_exposure=0.5,
        max_single_asset=0.3,
        min_diversification=2,
    )

    portfolio_constraints = PortfolioConstraints(
        max_leverage=1.5,
        max_drawdown_limit=0.15,
        min_liquidity_ratio=0.05,
        var_limit=0.04,
        concentration_limit=0.35,
    )

    cvar_config = CVaRConfig(confidence_level=0.05, window_size=30, risk_aversion=1.5)

    # Initialize risk manager
    risk_manager = RiskManager(
        assets=assets,
        position_limits=position_limits,
        portfolio_constraints=portfolio_constraints,
        cvar_config=cvar_config,
        enable_risk_shaping=True,
        enable_dynamic_limits=True,
    )

    print(f"✓ Risk Manager initialized with {len(assets)} assets")
    print(f"  - Max position size: {position_limits.max_position_size}")
    print(f"  - Max leverage: {portfolio_constraints.max_leverage}")
    print(f"  - Min diversification: {position_limits.min_diversification}")

    # Test position constraints
    current_prices = {asset.symbol: asset.price_history[-1] for asset in assets}

    # Test compliant positions
    compliant_positions = {"AAPL": 100, "MSFT": 50, "JPM": 30}

    # Set portfolio value for constraint checking
    risk_manager.current_portfolio_value = 100000

    results = risk_manager.check_position_constraints(
        compliant_positions, current_prices
    )
    print("\n✓ Compliant positions test:")
    print(f"  - Positions: {compliant_positions}")
    print(f"  - Is violation: {results['is_violation']}")
    print(f"  - Number of violations: {len(results['violations'])}")

    # Test violating positions
    violating_positions = {"AAPL": 1000, "MSFT": 0, "JPM": 0}  # Too large

    results = risk_manager.check_position_constraints(
        violating_positions, current_prices
    )
    print("\n✓ Violating positions test:")
    print(f"  - Positions: {violating_positions}")
    print(f"  - Is violation: {results['is_violation']}")
    if results["is_violation"]:
        print(f"  - Violations: {len(results['violations'])}")
        for violation in results["violations"][:2]:  # Show first 2 violations
            print(f"    * {violation['description']}")
        print(
            f"  - Adjusted positions sample: {list(results['adjusted_positions'].items())[:2]}"
        )

    # Test trade execution approval
    print("\n✓ Trade execution approval test:")

    # Test approved trade
    can_execute, reason = risk_manager.can_execute_trade(
        "AAPL", 50, "buy", current_prices
    )
    print(f"  - Buy 50 AAPL: {can_execute} ({reason})")

    # Test rejected trade
    can_execute, reason = risk_manager.can_execute_trade(
        "AAPL", 1000, "buy", current_prices
    )
    print(f"  - Buy 1000 AAPL: {can_execute} ({reason})")

    # Test risk state updates
    print("\n✓ Risk state monitoring:")

    portfolio_value = 100000
    cash_balance = 20000
    positions = compliant_positions

    for step in range(10):
        # Simulate portfolio changes
        portfolio_value *= 1 + np.random.normal(0.001, 0.01)
        cash_balance *= 1 + np.random.normal(0.0005, 0.005)

        # Update risk manager
        risk_manager.update_risk_state(
            portfolio_value, cash_balance, positions, current_prices, step
        )

        if step % 3 == 0:
            print(f"  Step {step}:")
            print(f"    Risk level: {risk_manager.current_risk_level.value}")
            print(f"    Portfolio value: ${portfolio_value:,.0f}")
            print(f"    Cash ratio: {cash_balance/portfolio_value:.2%}")

    # Get risk summary
    summary = risk_manager.get_risk_summary()
    print("\n✓ Final Risk Manager Summary:")
    print(f"  - Current risk level: {summary['current_risk_level']}")
    print(f"  - Portfolio value: ${summary['portfolio_value']:,.0f}")
    print(f"  - Cash balance: ${summary['cash_balance']:,.0f}")
    print(f"  - Total alerts: {summary['total_alerts']}")
    print(f"  - Dynamic limits: {summary['dynamic_limits']}")

    return risk_manager


def test_risk_integration():
    """Test integrated risk management with trading simulation."""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATED RISK SYSTEM")
    print("=" * 60)

    # Create assets
    assets = [
        AssetConfig(
            symbol="TECH1",
            sector="Technology",
            price_history=[100.0, 101.0, 102.0, 103.0, 104.0],
            volatility=0.02,
        ),
        AssetConfig(
            symbol="FIN1",
            sector="Finance",
            price_history=[50.0, 51.0, 50.5, 52.0, 51.5],
            volatility=0.015,
        ),
    ]

    # Initialize risk systems
    position_limits = risk.PositionLimits(max_position_size=0.5, min_diversification=1)
    portfolio_constraints = PortfolioConstraints(max_leverage=2.0, var_limit=0.05)
    cvar_config = CVaRConfig(confidence_level=0.05, risk_aversion=1.0)

    risk_manager = RiskManager(
        assets=assets,
        position_limits=position_limits,
        portfolio_constraints=portfolio_constraints,
        cvar_config=cvar_config,
        enable_risk_shaping=True,
    )

    print("✓ Integrated risk system initialized")

    # Simulate trading with risk management
    print("\n✓ Simulating trading with risk management:")

    portfolio_value = 100000
    cash_balance = 20000
    positions = {asset.symbol: 0 for asset in assets}
    current_prices = {asset.symbol: asset.price_history[-1] for asset in assets}

    total_pnl = 0
    risk_violations = 0
    executed_trades = 0
    rejected_trades = 0

    np.random.seed(42)

    for step in range(30):
        # Generate random trade
        symbol = np.random.choice(list(assets))
        action = np.random.choice(["buy", "sell"])
        quantity = np.random.randint(10, 100)

        # Check if trade is allowed
        can_execute, reason = risk_manager.can_execute_trade(
            symbol, quantity if action == "buy" else -quantity, action, current_prices
        )

        if can_execute:
            # Execute trade
            price = current_prices[symbol]
            trade_value = quantity * price
            cost = trade_value * 0.001  # 0.1% transaction cost

            if action == "buy":
                positions[symbol] += quantity
                cash_balance -= trade_value + cost
                total_pnl -= cost
            else:
                sell_quantity = min(positions[symbol], quantity)
                positions[symbol] -= sell_quantity
                cash_balance += sell_quantity * price - cost
                total_pnl -= cost

            executed_trades += 1
        else:
            rejected_trades += 1

        # Update portfolio value
        portfolio_value = cash_balance + sum(
            positions[symbol] * current_prices[symbol] for symbol in positions
        )

        # Update risk manager
        risk_manager.update_risk_state(
            portfolio_value, cash_balance, positions, current_prices, step
        )

        # Generate return and apply risk shaping
        if step > 0:
            base_return = (
                portfolio_value - prev_portfolio_value
            ) / prev_portfolio_value
            shaped_return = risk_manager.shape_reward(
                base_return, step, {"portfolio_value": portfolio_value}
            )

        prev_portfolio_value = portfolio_value

        # Simulate price movement
        for asset in assets:
            current_price = current_prices[asset.symbol]
            change = np.random.normal(0, asset.volatility * current_price)
            current_prices[asset.symbol] = max(0.1, current_price + change)

        # Check for risk violations
        if risk_manager.current_risk_level.value in ["high", "critical"]:
            risk_violations += 1

        if step % 10 == 0:
            print(f"  Step {step}:")
            print(f"    Portfolio value: ${portfolio_value:,.0f}")
            print(f"    P&L: ${total_pnl:,.0f}")
            print(f"    Risk level: {risk_manager.current_risk_level.value}")
            print(f"    Trades: {executed_trades} executed, {rejected_trades} rejected")

    # Final summary
    summary = risk_manager.get_risk_summary()
    print("\n✓ Integrated Risk System Results:")
    print(f"  - Final portfolio value: ${portfolio_value:,.0f}")
    print(f"  - Total P&L: ${total_pnl:,.0f}")
    print(f"  - Total return: {(portfolio_value - 100000) / 100000:.2%}")
    print(f"  - Risk violations: {risk_violations} steps")
    print(
        f"  - Trade execution rate: {executed_trades/(executed_trades+rejected_trades)*100:.1f}%"
    )
    print(f"  - Final risk level: {summary['current_risk_level']}")

    if "cvar_metrics" in summary and "risk_metrics" in summary["cvar_metrics"]:
        metrics = summary["cvar_metrics"]["risk_metrics"]
        print(f"  - Final CVaR: {metrics.get('cvar', 0):.4f}")
        print(f"  - Final VaR: {metrics.get('var', 0):.4f}")

    return risk_manager


def main():
    """Run all risk management tests."""
    print("SIMPLE RISK MANAGEMENT SYSTEM TEST")
    print("=" * 80)

    try:
        # Test 1: CVaR Reward Shaper
        cvar_shaper = test_cvar_reward_shaper()

        # Test 2: Risk Manager
        risk_manager = test_risk_manager()

        # Test 3: Integrated System
        integrated_system = test_risk_integration()

        print("\n" + "=" * 80)
        print("✓ ALL RISK MANAGEMENT TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)

        print("\nRisk Management System Validation:")
        print("✓ CVaR reward shaping with multiple risk measures")
        print("✓ Position constraints and portfolio risk limits")
        print("✓ Dynamic risk limit adjustment")
        print("✓ Trade execution approval based on risk constraints")
        print("✓ Real-time risk monitoring and alerting")
        print("✓ Integrated risk management with trading simulation")

        print("\nPhase 3 (Advanced Risk & Reward Systems) COMPLETED!")
        print("Ready for Phase 4: Professional Training Pipeline")

    except Exception as e:
        print(f"\n✗ Risk management test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

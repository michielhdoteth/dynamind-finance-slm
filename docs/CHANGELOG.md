# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Open-source release preparation
- Professional documentation and examples
- MIT license
- Comprehensive test suite
- Development and contribution guidelines

### Changed
- Improved package structure for OSS distribution
- Updated README for broader audience
- Enhanced API documentation

### Removed
- Development artifacts and temporary files
- Large model checkpoints (moved to separate storage)
- Research-specific validation results (archived)

## [0.1.0] - 2025-01-27

### Added
- Initial release of Financial Trading RL Gym
- Multiple trading environments (single asset, portfolio, regime detection, market making)
- Qwen transformer integration for advanced RL agents
- Advanced PPO training with trust-region control
- Comprehensive risk management framework
- Statistical validation tools and analysis pipelines
- Real market data integration capabilities
- Research validation with meta-learning emergence evidence

### Features
- **Environments:**
  - Single Asset Trading (`FinancialTrading-SingleAsset-v0`)
  - Portfolio Optimization (`FinancialTrading-Portfolio-v0`)
  - Regime Detection (`FinancialTrading-RegimeDetection-v0`)
  - Market Making (`FinancialTrading-MarketMaking-v0`)

- **Agents:**
  - Qwen transformer-based trading agents
  - Custom feature extractors for financial data
  - Advanced PPO implementation with KL constraints

- **Training:**
  - Multi-seed training infrastructure
  - Statistical validation framework
  - Bootstrap confidence intervals
  - Effect size analysis

- **Risk Management:**
  - VaR constraints and drawdown limits
  - Position sizing and sector exposure limits
  - CVaR reward shaping
  - Portfolio metrics calculation

- **Data:**
  - Synthetic market data generation
  - Real market data integration (Yahoo Finance)
  - Technical indicators and preprocessing
  - Regime switching market dynamics

### Research Validation
- Meta-learning emergence detection at ~55k training steps
- Large effect sizes (Cohen's d > 0.8) for key metrics
- Real market validation on 2024 data
- Robust performance across market regimes
- Statistical validation with bootstrap methods

---

## Version History

### v0.1.0 (2025-01-27)
- Initial public release
- Core trading environments
- Transformer agent integration
- Research validation framework

### Future Roadmap
- v0.2.0: Extended documentation and examples
- v0.3.0: Additional environment types
- v1.0.0: Stable production release
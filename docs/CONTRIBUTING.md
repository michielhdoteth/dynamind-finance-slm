# Contributing to Financial Trading RL Gym

Thank you for your interest in contributing to Financial Trading RL Gym! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with reinforcement learning and financial markets

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/financial-trading-rl-gym.git
cd financial-trading-rl-gym

# Create development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## 📋 Development Guidelines

### Code Style

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **isort**: Import sorting

```bash
# Format code
black financial_trading_gym/ tests/ examples/

# Sort imports
isort financial_trading_gym/ tests/ examples/

# Lint code
flake8 financial_trading_gym/ tests/ examples/

# Type checking
mypy financial_trading_gym/
```

### Testing

Before submitting a pull request, please ensure:

1. All tests pass:
   ```bash
   pytest tests/
   ```

2. Code coverage is maintained:
   ```bash
   pytest tests/ --cov=financial_trading_gym --cov-report=term-missing
   ```

3. Your changes don't break existing functionality:
   ```bash
   pytest tests/integration/
   ```

### Documentation

- Add docstrings to all public functions and classes
- Use Google-style docstrings
- Update relevant documentation in `docs/`
- Add examples for new features

## 🏗️ Adding New Environments

### Step 1: Create Environment Class

```python
from financial_trading_gym.environments.base_env import FinancialTradingBase
from gymnasium import spaces

class YourCustomEnv(FinancialTradingBase):
    """
    Description of your custom trading environment.

    This environment implements [specific features].
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your environment-specific parameters

    def _get_observation_space(self):
        """Return the observation space for your environment."""
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,), dtype=np.float32)

    def _get_action_space(self):
        """Return the action space for your environment."""
        return spaces.Discrete(3)  # Example: Sell, Hold, Buy

    def _get_info(self):
        """Return additional information dictionary."""
        return {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'cash': self.cash,
        }

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observation."""
        super().reset(seed=seed)
        # Implement reset logic
        return self._get_observation(), self._get_info()

    def step(self, action):
        """Execute one step in the environment."""
        # Implement step logic
        reward = self._calculate_reward()
        terminated = self._check_termination()
        truncated = self._check_truncation()
        info = self._get_info()

        return self._get_observation(), reward, terminated, truncated, info
```

### Step 2: Add Tests

Create tests in `tests/test_your_env.py`:

```python
import pytest
import gymnasium as gym
import financial_trading_gym

class TestYourCustomEnv:
    """Test suite for your custom environment."""

    def test_environment_creation(self):
        """Test that environment can be created successfully."""
        env = gym.make('financial_trading_gym/YourCustomEnv-v0')
        assert env is not None

    def test_observation_space(self):
        """Test observation space specifications."""
        env = gym.make('financial_trading_gym/YourCustomEnv-v0')
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_action_space(self):
        """Test action space specifications."""
        env = gym.make('financial_trading_gym/YourCustomEnv-v0')
        action = env.action_space.sample()
        assert env.action_space.contains(action)

    def test_step_functionality(self):
        """Test that step works correctly."""
        env = gym.make('financial_trading_gym/YourCustomEnv-v0')
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
```

### Step 3: Add Documentation

- Add environment description to `README.md`
- Create usage example in `examples/your_env_example.py`
- Add API documentation to `docs/api/your_env.md`

### Step 4: Register Environment

Register your environment in `financial_trading_gym/__init__.py`:

```python
from gymnasium import register

register(
    id='financial_trading_gym/YourCustomEnv-v0',
    entry_point='financial_trading_gym.environments.your_module:YourCustomEnv',
)
```

## 🐛 Bug Reports

When filing bug reports, please include:

1. **Environment Information:**
   - Python version
   - OS version
   - Package version

2. **Minimal Reproducible Example:**
   ```python
   import gymnasium as gym
   import financial_trading_gym

   env = gym.make('financial_trading_gym/YourEnv-v0')
   # Your code that reproduces the issue
   ```

3. **Expected vs Actual Behavior**
4. **Error Messages and Stack Traces**

## 💡 Feature Requests

When requesting features, please:

1. **Check existing issues** to avoid duplicates
2. **Provide clear use cases** for the feature
3. **Consider implementation complexity**
4. **Offer to contribute** if possible

## 📝 Pull Request Process

### Before Submitting

1. **Fork the repository** and create a feature branch
2. **Write tests** for new functionality
3. **Ensure all tests pass**
4. **Update documentation** as needed
5. **Run the full test suite**

### Submitting PRs

1. **Use descriptive titles** for your pull requests
2. **Link to relevant issues** in the description
3. **Provide clear descriptions** of changes
4. **Include screenshots** for UI changes
5. **Request code reviews** from maintainers

### PR Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

## 🔍 Code Review Guidelines

### For Reviewers

1. **Check functionality** - Does the code work as intended?
2. **Check style** - Does it follow project conventions?
3. **Check tests** - Are tests comprehensive?
4. **Check documentation** - Is it clear and accurate?
5. **Check performance** - Any obvious performance issues?

### For Authors

1. **Respond to feedback** promptly
2. **Make requested changes** in a timely manner
3. **Explain complex code** when necessary
4. **Update documentation** for API changes

## 📚 Documentation Guidelines

### Code Documentation

- Use Google-style docstrings
- Document all public methods and classes
- Include parameter types and return types
- Add examples for complex functionality

```python
def calculate_reward(self, action: np.ndarray, market_data: Dict[str, Any]) -> float:
    """Calculate reward for the given action and market conditions.

    Args:
        action: Trading action taken by the agent
        market_data: Dictionary containing market information

    Returns:
        Calculated reward value

    Example:
        >>> env = FinancialTradingEnv()
        >>> reward = env.calculate_reward(action, market_data)
        >>> print(reward)
        0.123
    """
```

### README Documentation

- Keep examples simple and runnable
- Include installation instructions
- Provide troubleshooting tips
- Update version information

## 🏆 Recognition

Contributors will be recognized in:

- README.md contributors section
- CHANGELOG.md for significant contributions
- Release notes for major features
- Project website (when implemented)

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🤝 Community

Join our discussions:
- GitHub Issues for bug reports and feature requests
- GitHub Discussions for general questions
- Email: michiel.horstman@4mlabs.io for private inquiries

Thank you for contributing to Financial Trading RL Gym! 🎉
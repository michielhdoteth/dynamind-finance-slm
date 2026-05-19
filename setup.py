"""
Setup script for RL Financial Markets Gym
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="financial-trading-rl-gym",
    version="0.1.0",
    author="Michiel Horstman",
    author_email="michiel.horstman@4mlabs.io",
    maintainer="Michiel Horstman",
    maintainer_email="michiel.horstman@4mlabs.io",
    description="Professional Reinforcement Learning Financial Trading Gym Environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/4mlabs/financial-trading-rl-gym",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "gymnasium>=0.26.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
        "yfinance>=0.1.70",
        "requests>=2.25.0",
        "stable-baselines3>=1.6.0",
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "protobuf>=3.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "jupyter>=1.0",
        ],
        "data": [
            "alpha_vantage>=2.3.0",
            "sqlalchemy>=1.4.0",
            "psycopg2-binary>=2.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "financial_trading_gym=financial_trading_gym.cli:main",
        ],
    },
)
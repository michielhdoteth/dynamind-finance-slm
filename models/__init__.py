"""
Model Integration for Financial Trading Gym

This module provides integration with various language models including
open-source models like Qwen, GPT, and other LLMs for trading decisions.
"""

from .llm_trader import LLMTradingAgent
from .qwen_integration import QwenTradingModel

__all__ = [
    'QwenTradingModel',
    'LLMTradingAgent'
]
"""
Data models for the Financial Market Environment.

Defines typed Action, Observation, and State used by both the server and client.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class MarketAction(Action):
    """Action for the Financial Market environment."""

    symbol: str = Field(..., description="Stock symbol to trade (e.g. 'RELIANCE.NS')")
    action_type: Literal["buy", "sell", "hold"] = Field(
        ..., description="Action type: buy, sell, or hold"
    )
    quantity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Fraction of available cash (buy) or holdings (sell) to trade",
    )


class MarketObservation(Observation):
    """Observation from the Financial Market environment."""

    market_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Market data per symbol: open/high/low/close/volume/rsi/macd",
    )
    portfolio: Dict[str, float] = Field(
        default_factory=dict,
        description="Shares held per symbol",
    )
    cash_balance: float = Field(
        default=10000.0,
        description="Available cash in portfolio",
    )
    portfolio_value: float = Field(
        default=10000.0,
        description="Total portfolio value (cash + holdings at current prices)",
    )
    task: str = Field(
        default="easy",
        description="Task difficulty: easy, medium, or hard",
    )
    step_num: int = Field(
        default=0,
        description="Current step number within episode",
    )
    feedback: str = Field(
        default="",
        description="Natural language feedback about the last action taken",
    )


class MarketState(State):
    """State of the Financial Market environment session."""

    task: str = Field(default="easy", description="Active task difficulty")
    max_steps: int = Field(default=5, description="Maximum steps in this episode")
    symbols: List[str] = Field(
        default_factory=list, description="Stock symbols available in this episode"
    )
    initial_value: float = Field(
        default=10000.0, description="Starting portfolio value for this episode"
    )

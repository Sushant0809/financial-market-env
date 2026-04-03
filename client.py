"""
Financial Market Environment Client.

Maintains a persistent WebSocket connection to the environment server
for efficient multi-step interaction (lower latency than HTTP).
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import MarketAction, MarketActions, MarketObservation, MarketState
except ImportError:
    from models import MarketAction, MarketActions, MarketObservation, MarketState


class MarketEnv(EnvClient[MarketActions, MarketObservation, MarketState]):
    """
    Async client for the Financial Market Environment.

    Example (async):
        >>> async with MarketEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task="easy")
        ...     actions = MarketActions(actions=[
        ...         MarketAction(symbol="RELIANCE.NS", action_type="buy", quantity=0.5),
        ...         MarketAction(symbol="TCS.NS", action_type="buy", quantity=0.3),
        ...     ])
        ...     result = await env.step(actions)
        ...     print(result.reward)
    """

    def _step_payload(self, action: MarketActions) -> Dict:
        return {
            "actions": [
                {"symbol": a.symbol, "action_type": a.action_type, "quantity": a.quantity}
                for a in action.actions
            ]
        }

    def _parse_result(self, payload: Dict) -> StepResult[MarketObservation]:
        obs_data = payload.get("observation", {})
        observation = MarketObservation(
            market_data=obs_data.get("market_data", {}),
            portfolio=obs_data.get("portfolio", {}),
            cash_balance=obs_data.get("cash_balance", 0.0),
            portfolio_value=obs_data.get("portfolio_value", 0.0),
            task=obs_data.get("task", "easy"),
            step_num=obs_data.get("step_num", 0),
            feedback=obs_data.get("feedback", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> MarketState:
        return MarketState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "easy"),
            max_steps=payload.get("max_steps", 5),
            symbols=payload.get("symbols", []),
            initial_value=payload.get("initial_value", 10000.0),
        )

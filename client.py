"""
Financial Market Environment Client.

Maintains a persistent WebSocket connection to the environment server
for efficient multi-step interaction (lower latency than HTTP).
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import MarketAction, MarketObservation, MarketState
except ImportError:
    from models import MarketAction, MarketObservation, MarketState


class MarketEnv(EnvClient[MarketAction, MarketObservation, MarketState]):
    """
    Async client for the Financial Market Environment.

    Example (async):
        >>> async with MarketEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task="easy")
        ...     result = await env.step(MarketAction(symbol="RELIANCE.NS", action_type="buy", quantity=0.5))
        ...     print(result.reward)

    Example (from Docker):
        >>> env = await MarketEnv.from_docker_image("financial-market-env:latest")
        >>> try:
        ...     result = await env.reset(task="medium")
        ...     result = await env.step(MarketAction(symbol="TCS.NS", action_type="hold", quantity=0.0))
        ... finally:
        ...     await env.close()
    """

    def _step_payload(self, action: MarketAction) -> Dict:
        return {
            "symbol": action.symbol,
            "action_type": action.action_type,
            "quantity": action.quantity,
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

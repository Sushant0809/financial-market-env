"""
Financial Market Environment Implementation.

Simulates stock market trading with three difficulty-tiered tasks:
  - easy:   1 stock, 5 steps,  ₹10,000 initial cash
  - medium: 3 stocks, 10 steps, ₹30,000 initial cash
  - hard:   5 stocks, 20 steps, ₹50,000 initial cash

The environment is OpenEnv-compliant: reset() / step() / state property.
"""

from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.rubrics import Rubric

try:
    from ..models import MarketAction, MarketActions, MarketObservation, MarketState
    from ..simulator import MarketDataSimulator, TASK_INITIAL_CASH, TASK_STEPS, TASK_SYMBOLS
    from ..grader import compute_step_reward, compute_final_reward
except ImportError:
    from models import MarketAction, MarketActions, MarketObservation, MarketState
    from simulator import MarketDataSimulator, TASK_INITIAL_CASH, TASK_STEPS, TASK_SYMBOLS
    from grader import compute_step_reward, compute_final_reward


class EasyGrader(Rubric):
    """Easy task: normalised simple return. 10% return = score 1.0."""
    def forward(self, action: Any, observation: Any) -> float:
        return float(observation.reward) if observation.reward is not None else 0.0


class MediumGrader(Rubric):
    """Medium task: sigmoid of annualised Sharpe ratio."""
    def forward(self, action: Any, observation: Any) -> float:
        return float(observation.reward) if observation.reward is not None else 0.0


class HardGrader(Rubric):
    """Hard/nifty50 task: return minus drawdown penalty."""
    def forward(self, action: Any, observation: Any) -> float:
        return float(observation.reward) if observation.reward is not None else 0.0


TASK_RUBRICS = {
    "easy":    EasyGrader,
    "medium":  MediumGrader,
    "hard":    HardGrader,
    "nifty50": HardGrader,
}


# Shared simulator instance (data loaded once per process)
_simulator = MarketDataSimulator()


class MarketEnvironment(Environment):
    """
    Financial Market trading environment.

    Each session maintains its own portfolio, episode window, and history,
    making the environment safe for concurrent use.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = MarketState()
        # Episode state (reset on every reset())
        self._task: str = "easy"
        self._symbols: list = []
        self._max_steps: int = 5
        self._initial_cash: float = 10000.0
        self._cash: float = 10000.0
        self._portfolio: Dict[str, float] = {}  # symbol -> shares
        self._window: dict = {}                  # symbol -> DataFrame
        self._current_step: int = 0
        self._history: list = []
        self._initial_value: float = 10000.0
        self._done: bool = False

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "easy",
        **kwargs: Any,
    ) -> MarketObservation:
        """Reset the environment for a new episode."""
        # Set rubric for this task so the framework recognises it as graded
        self.rubric = TASK_RUBRICS.get(task, EasyGrader)()

        self._task = task if task in TASK_STEPS else "easy"
        self._symbols = TASK_SYMBOLS[self._task]
        self._max_steps = TASK_STEPS[self._task]
        self._initial_cash = TASK_INITIAL_CASH[self._task]
        self._cash = self._initial_cash
        self._portfolio = {sym: 0.0 for sym in self._symbols}
        self._current_step = 0
        self._done = False
        self._history = []

        # Sample a random episode window
        self._window = _simulator.get_episode_window(
            self._symbols, self._max_steps, seed=seed
        )

        prices = _simulator.get_prices(self._window, 0)
        portfolio_value = self._portfolio_value(prices)
        self._initial_value = portfolio_value

        self._state = MarketState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task=self._task,
            max_steps=self._max_steps,
            symbols=self._symbols,
            initial_value=self._initial_value,
        )

        self._history.append({"portfolio_value": portfolio_value})

        market_data = _simulator.get_market_data(self._window, 0)
        return MarketObservation(
            market_data=market_data,
            portfolio=dict(self._portfolio),
            cash_balance=round(self._cash, 2),
            portfolio_value=round(portfolio_value, 2),
            task=self._task,
            step_num=0,
            feedback=(
                f"Episode started. Task: {self._task}. "
                f"Available symbols: {', '.join(self._symbols)}. "
                f"Initial cash: ₹{self._cash:,.2f}. "
                f"You have {self._max_steps} steps."
            ),
            done=False,
            reward=None,
        )

    def step(
        self,
        action: MarketActions,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MarketObservation:
        """Execute multiple trading actions and advance the market by one day."""
        if self._done:
            return self._terminal_obs("Episode already finished.")

        self._current_step += 1
        self._state.step_count = self._current_step

        prices_before = _simulator.get_prices(self._window, self._current_step - 1)
        prices_after = _simulator.get_prices(self._window, self._current_step)

        prev_value = self._portfolio_value(prices_before)

        # Execute all actions for this step
        feedbacks = []
        for act in action.actions:
            feedbacks.append(self._execute_action(act, prices_before))
        feedback = " | ".join(feedbacks) if feedbacks else "No actions submitted."

        curr_value = self._portfolio_value(prices_after)

        self._history.append({"portfolio_value": curr_value})

        done = self._current_step >= self._max_steps
        self._done = done

        # Compute reward
        if done:
            reward = compute_final_reward(self._task, self._history, self._initial_value)
        else:
            reward = compute_step_reward(prev_value, curr_value, self._initial_value)

        market_data = _simulator.get_market_data(self._window, self._current_step)
        return MarketObservation(
            market_data=market_data,
            portfolio=dict(self._portfolio),
            cash_balance=round(self._cash, 2),
            portfolio_value=round(curr_value, 2),
            task=self._task,
            step_num=self._current_step,
            feedback=feedback,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> MarketState:
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _portfolio_value(self, prices: Dict[str, float]) -> float:
        holdings_value = sum(
            self._portfolio.get(sym, 0.0) * price for sym, price in prices.items()
        )
        return self._cash + holdings_value

    def _execute_action(self, action: MarketAction, prices: Dict[str, float]) -> str:
        sym = action.symbol
        act = action.action_type
        qty = float(action.quantity)

        if sym not in self._symbols:
            return (
                f"Invalid symbol '{sym}'. Valid: {', '.join(self._symbols)}. "
                f"No trade executed."
            )

        price = prices[sym]

        if act == "buy":
            spend = self._cash * qty
            if spend < price:
                return f"Insufficient cash to buy {sym}. Cash: ₹{self._cash:.2f}, Price: ₹{price:.2f}."
            shares = spend / price
            self._cash -= spend
            self._portfolio[sym] = self._portfolio.get(sym, 0.0) + shares
            return (
                f"Bought {shares:.4f} shares of {sym} at ₹{price:.2f} "
                f"(spent ₹{spend:.2f}). Cash remaining: ₹{self._cash:.2f}."
            )

        elif act == "sell":
            held = self._portfolio.get(sym, 0.0)
            if held <= 0:
                return f"No shares of {sym} to sell."
            sell_shares = held * qty
            proceeds = sell_shares * price
            self._portfolio[sym] = held - sell_shares
            self._cash += proceeds
            return (
                f"Sold {sell_shares:.4f} shares of {sym} at ₹{price:.2f} "
                f"(received ₹{proceeds:.2f}). Cash: ₹{self._cash:.2f}."
            )

        else:  # hold
            return f"Held position. {sym} at ₹{price:.2f}. Cash: ₹{self._cash:.2f}."

    def _terminal_obs(self, msg: str) -> MarketObservation:
        prices = _simulator.get_prices(self._window, min(self._current_step, self._max_steps))
        portfolio_value = self._portfolio_value(prices)
        market_data = _simulator.get_market_data(self._window, min(self._current_step, self._max_steps))
        return MarketObservation(
            market_data=market_data,
            portfolio=dict(self._portfolio),
            cash_balance=round(self._cash, 2),
            portfolio_value=round(portfolio_value, 2),
            task=self._task,
            step_num=self._current_step,
            feedback=msg,
            done=True,
            reward=0.0,
        )

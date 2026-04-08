"""
Reward (grader) functions for each task difficulty.

All functions return a float in [0.0, 1.0].

Per-step partial rewards are computed by `compute_step_reward`.
Final episode rewards are computed by `compute_final_reward`.
"""

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# Per-step partial reward
# ---------------------------------------------------------------------------

def compute_step_reward(prev_value: float, curr_value: float, initial_value: float) -> float:
    """
    Partial progress reward emitted after every step.

    Returns the portfolio improvement this step as a fraction of initial value.
    Only positive improvements are rewarded (no penalty for losses — the final
    reward handles the overall performance).
    """
    improvement = (curr_value - prev_value) / max(initial_value, 1e-9)
    return float(max(0.0, improvement))


# ---------------------------------------------------------------------------
# Final episode reward
# ---------------------------------------------------------------------------

def compute_final_reward(task: str, history: List[dict], initial_value: float) -> float:
    """Dispatch to the appropriate grader based on task difficulty."""
    if task == "easy":
        return _easy_reward(history, initial_value)
    elif task == "medium":
        return _medium_reward(history, initial_value)
    elif task in ("hard", "nifty50"):
        return _hard_reward(history, initial_value)
    return 1e-4


def _easy_reward(history: List[dict], initial_value: float) -> float:
    """
    Easy task: normalized simple return.

    Mapping: 0% return → 0.0, 10% return → 1.0 (capped).
    The agent is rewarded for any positive return vs the starting portfolio value.
    """
    if not history:
        return 0.0
    final_value = history[-1]["portfolio_value"]
    ret = (final_value - initial_value) / max(initial_value, 1e-9)
    # Scale so that a 10% total return scores 1.0; clamp to open interval (0, 1)
    return float(min(max(ret * 10.0, 1e-4), 1.0 - 1e-4))


def _medium_reward(history: List[dict], initial_value: float) -> float:
    """
    Medium task: Sharpe ratio (annualised) mapped to [0, 1] via sigmoid.

    Sharpe > 0 → score > 0.5; Sharpe = 2 → score ≈ 0.88.
    Ensures partial credit even for volatile but profitable strategies.
    """
    if len(history) < 2:
        return 0.0
    values = [h["portfolio_value"] for h in history]
    step_returns = np.diff(values) / np.maximum(values[:-1], 1e-9)
    std = float(np.std(step_returns))
    if std < 1e-10:
        return 0.5 if float(np.mean(step_returns)) >= 0 else 1e-4
    # Annualise assuming ~252 trading days; episodes are short so scale lightly
    sharpe = float(np.mean(step_returns) / std) * np.sqrt(252)
    # Sigmoid centred at sharpe=0: score ∈ (0, 1), clamp to open interval
    score = float(1.0 / (1.0 + np.exp(-sharpe / 2.0)))
    return float(min(max(score, 1e-4), 1.0 - 1e-4))


def _hard_reward(history: List[dict], initial_value: float) -> float:
    """
    Hard task: total return minus 0.5× max drawdown, normalised to [0, 1].

    Penalises strategies that achieve returns via high drawdown (risky bets).
    A strategy with 10% return and 0% drawdown scores 1.0.
    """
    if len(history) < 2:
        return 0.0
    values = [h["portfolio_value"] for h in history]
    total_ret = (values[-1] - values[0]) / max(values[0], 1e-9)

    # Maximum drawdown
    peak = float(values[0])
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    # Score: centre at 0.5, penalise drawdown; clamp to open interval (0, 1)
    score = (total_ret + 0.5) - 0.5 * max_dd
    return float(min(max(score, 1e-4), 1.0 - 1e-4))

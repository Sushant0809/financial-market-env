"""
FastAPI application for the Financial Market Environment.

Endpoints:
    POST /reset    — Reset the environment for a new episode
    POST /step     — Execute a trading action
    GET  /state    — Get current session state
    GET  /schema   — Action / observation schemas
    GET  /tasks    — List all available tasks with metadata
    POST /grader   — Grade a submission for a given task
    WS   /ws       — WebSocket endpoint for persistent sessions (used by EnvClient)

Usage:
    uvicorn server.app:app --host 0.0.0.0 --port 8000
"""

from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core is required. Install with: pip install 'openenv-core[core]'") from e

try:
    from ..models import MarketActions, MarketObservation
    from .market_environment import MarketEnvironment
except ImportError:
    from models import MarketActions, MarketObservation
    from server.market_environment import MarketEnvironment


app = create_app(
    MarketEnvironment,
    MarketActions,
    MarketObservation,
    env_name="financial_market_env",
    max_concurrent_envs=16,
)

# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS = {
    "easy": {
        "id": "easy",
        "difficulty": "easy",
        "description": "Single stock (RELIANCE.NS), 10 steps, INR 10,000 starting cash. Score = return * 10, capped at 1.0.",
        "max_steps": 10,
        "reward_range": [0.0, 1.0],
        "symbols": ["RELIANCE.NS"],
        "initial_cash": 10000.0,
    },
    "medium": {
        "id": "medium",
        "difficulty": "medium",
        "description": "3 stocks (RELIANCE, TCS, HDFCBANK), 30 steps, INR 30,000. Score = sigmoid(annualised Sharpe ratio).",
        "max_steps": 30,
        "reward_range": [0.0, 1.0],
        "symbols": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
        "initial_cash": 30000.0,
    },
    "hard": {
        "id": "hard",
        "difficulty": "hard",
        "description": "5 stocks, 40 steps, INR 50,000. Score = total_return - 0.5 * max_drawdown, normalised to [0,1].",
        "max_steps": 40,
        "reward_range": [0.0, 1.0],
        "symbols": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"],
        "initial_cash": 50000.0,
    },
    "nifty50": {
        "id": "nifty50",
        "difficulty": "hard",
        "description": "All 50 NIFTY stocks, 30 steps, INR 500,000. Same scoring as hard task.",
        "max_steps": 30,
        "reward_range": [0.0, 1.0],
        "symbols": ["all 50 NIFTY constituents"],
        "initial_cash": 500000.0,
    },
}


# ---------------------------------------------------------------------------
# /tasks endpoint
# ---------------------------------------------------------------------------

@app.get("/tasks", tags=["Competition"])
def list_tasks():
    """List all available tasks with metadata and grading info."""
    return {
        "tasks": list(TASKS.values()),
        "total": len(TASKS),
        "grading": {
            "type": "deterministic_programmatic",
            "score_range": [0.0, 1.0],
            "llm_judging": False,
        },
    }


# ---------------------------------------------------------------------------
# /grader endpoint
# ---------------------------------------------------------------------------

class GraderRequest(BaseModel):
    task_id: str
    score: Optional[float] = None
    history: Optional[list] = None
    initial_value: Optional[float] = None


@app.post("/grader", tags=["Competition"])
def run_grader(request: GraderRequest):
    """Grade a submission for a given task."""
    task_id = request.task_id
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail=f"Unknown task: {task_id}. Valid: {list(TASKS.keys())}")

    try:
        from grader import compute_final_reward
    except ImportError:
        from grader import compute_final_reward

    score = request.score
    if score is None and request.history and request.initial_value:
        score = compute_final_reward(task_id, request.history, request.initial_value)

    score = float(score) if score is not None else 1e-4
    score = min(max(score, 1e-4), 1.0 - 1e-4)

    return {
        "task_id": task_id,
        "score": score,
        "passed": score >= 0.3,
        "feedback": f"Score {score:.3f} for task '{task_id}'. Threshold for pass: 0.3.",
        "grader": "deterministic_programmatic",
    }


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

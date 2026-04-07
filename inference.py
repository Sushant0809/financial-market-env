"""
Inference Script — Financial Market Environment
================================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL        The API endpoint for the LLM.
    MODEL_NAME          The model identifier to use for inference.
    HF_TOKEN            Your Hugging Face / API key.
    LOCAL_IMAGE_NAME    Docker image name (if using from_docker_image).

STDOUT FORMAT (evaluated automatically):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import AsyncOpenAI

from client import MarketEnv, MarketAction, MarketActions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MARKET_TASK", "easy")          # easy | medium | hard | all
BENCHMARK = os.getenv("MARKET_BENCHMARK", "financial_market_env")
MAX_STEPS = int(os.getenv("MARKET_MAX_STEPS", "8"))   # hard cap; env may terminate earlier
TEMPERATURE = 0.3
MAX_TOKENS = 2000
SUCCESS_SCORE_THRESHOLD = 0.3  # score ≥ 0.3 counts as success

# Per-step max reward depends on task: scale so perfect trading ≈ MAX_STEPS * 0.02
_MAX_REWARD_PER_STEP = 0.02
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

# When MARKET_TASK is not explicitly set to a single task, run all three
_ALL_TASKS = ["easy", "medium", "hard"]
TASK_LIST = _ALL_TASKS if TASK_NAME not in ("easy", "medium", "hard", "nifty50") else [TASK_NAME]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert stock trading agent operating in the Indian equity market (NIFTY 50).
    At each step you receive:
      - market_data: OHLCV + RSI + MACD for each available symbol
      - portfolio: shares currently held per symbol
      - cash_balance: available cash (in INR)
      - feedback: result of your last actions

    Your goal is to maximise portfolio value over the episode.

    You MUST respond with a JSON array — one entry per available symbol.
    Use EXACTLY these field names:

    [
      {"symbol": "RELIANCE.NS", "action_type": "buy",  "quantity": 0.40},
      {"symbol": "TCS.NS",      "action_type": "hold", "quantity": 0.00},
      {"symbol": "HDFCBANK.NS", "action_type": "buy",  "quantity": 0.30}
    ]

    Field rules (STRICT):
    - "symbol": exact symbol string from market_data
    - "action_type": must be exactly "buy", "sell", or "hold"
    - "quantity": a fraction between 0.0 and 1.0 (NOT number of shares)
      - buy: spends quantity × available_cash on that symbol
      - sell: sells quantity × shares_held of that symbol
      - hold: do nothing (quantity ignored)

    Strategy rules:
    - Include ALL available symbols in your response
    - Diversify: buy 2-3 stocks using 0.2–0.4 quantity each (not all-in on one)
    - RSI < 30 = oversold → good to buy
    - RSI > 70 = overbought → consider selling
    - MACD positive = uptrend, negative = downtrend
    - Keep at least 10% cash reserve
    - Respond ONLY with the JSON array. No markdown, no explanation.
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers (exact format required by hackathon grader)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(obs) -> str:
    symbols = list(obs.market_data.keys())
    market_lines = []
    for sym in symbols:
        d = obs.market_data[sym]
        market_lines.append(
            f"  {sym}: close=₹{d['close']}, RSI={d['rsi']}, MACD={d['macd']:.4f}"
        )
    portfolio_lines = [
        f"  {sym}: {shares:.4f} shares"
        for sym, shares in obs.portfolio.items()
        if shares > 0
    ] or ["  (no holdings)"]

    return textwrap.dedent(
        f"""
        Step {obs.step_num} | Task: {obs.task} | Cash: ₹{obs.cash_balance:,.2f} | Portfolio value: ₹{obs.portfolio_value:,.2f}

        Market data:
        {chr(10).join(market_lines)}

        Holdings:
        {chr(10).join(portfolio_lines)}

        Last feedback: {obs.feedback}

        Available symbols: {', '.join(symbols)}
        Respond with ONE JSON action.
        """
    ).strip()


def _parse_actions(text: str, valid_symbols: list) -> MarketActions:
    """Extract and validate a MarketActions list from the LLM's text response."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()

    def _fallback() -> MarketActions:
        return MarketActions(actions=[
            MarketAction(symbol=sym, action_type="hold", quantity=0.0)
            for sym in valid_symbols
        ])

    try:
        data = json.loads(text)
        # Handle both array and single object responses
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return _fallback()

        actions = []
        for item in data:
            symbol = item.get("symbol", "")
            action_type = item.get("action_type") or item.get("action", "hold")
            quantity = float(item.get("quantity", 0.0))
            # Clamp: quantity must be a fraction 0-1, not share count
            if quantity > 1.0:
                quantity = 1.0
            if symbol not in valid_symbols:
                continue
            action_type = action_type if action_type in ("buy", "sell", "hold") else "hold"
            quantity = max(0.0, min(1.0, quantity))
            actions.append(MarketAction(symbol=symbol, action_type=action_type, quantity=quantity))

        # Ensure every symbol has an action (default hold for missing ones)
        acted_symbols = {a.symbol for a in actions}
        for sym in valid_symbols:
            if sym not in acted_symbols:
                actions.append(MarketAction(symbol=sym, action_type="hold", quantity=0.0))

        return MarketActions(actions=actions) if actions else _fallback()
    except Exception:
        return _fallback()


async def _get_model_action(client: AsyncOpenAI, obs, history: List[str]) -> MarketActions:
    user_prompt = _build_user_prompt(obs)
    valid_symbols = list(obs.market_data.keys())
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        msg = completion.choices[0].message
        text = (msg.content or getattr(msg, "reasoning_content", None) or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        text = ""

    return _parse_actions(text, valid_symbols)


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

async def run_episode(client: AsyncOpenAI, env: "MarketEnv", task: str) -> None:
    """Run a single episode for the given task and log [START]/[END]."""
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task)
        obs = result.observation
        initial_pv = obs.portfolio_value or obs.cash_balance or 1.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            actions = await _get_model_action(client, obs, history)
            action_str = "+".join(
                f"{a.action_type}({a.symbol},{a.quantity:.2f})"
                for a in actions.actions
                if a.action_type != "hold" or a.quantity > 0
            ) or "hold(all,0.00)"

            result = await env.step(actions)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            pnl_pct = (obs.portfolio_value - initial_pv) / initial_pv * 100
            cumulative_score_pct = sum(rewards) / MAX_TOTAL_REWARD * 100 if MAX_TOTAL_REWARD > 0 else 0.0
            print(
                f"[PNL]  step={step} portfolio=₹{obs.portfolio_value:,.2f} "
                f"pnl={pnl_pct:+.2f}% score={cumulative_score_pct:.1f}%",
                flush=True,
            )

            history.append(
                f"Step {step}: {action_str} → reward {reward:+.4f} | pv=₹{obs.portfolio_value:,.2f} | cash=₹{obs.cash_balance:,.2f}"
            )

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error ({task}): {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = None

    try:
        if IMAGE_NAME:
            env = await MarketEnv.from_docker_image(IMAGE_NAME)
        else:
            base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")
            env = MarketEnv(base_url=base_url)
            await env.connect()

        for task in TASK_LIST:
            await run_episode(client, env, task)

    except Exception as exc:
        print(f"[DEBUG] Setup error: {exc}", flush=True)
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception as e:
                print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

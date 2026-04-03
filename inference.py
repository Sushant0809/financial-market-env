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

from openai import OpenAI

from client import MarketEnv, MarketAction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("MARKET_TASK", "easy")          # easy | medium | hard
BENCHMARK = os.getenv("MARKET_BENCHMARK", "financial_market_env")
MAX_STEPS = int(os.getenv("MARKET_MAX_STEPS", "8"))   # hard cap; env may terminate earlier
TEMPERATURE = 0.3
MAX_TOKENS = 200
SUCCESS_SCORE_THRESHOLD = 0.3  # score ≥ 0.3 counts as success

# Per-step max reward depends on task: scale so perfect trading ≈ MAX_STEPS * 0.02
_MAX_REWARD_PER_STEP = 0.02
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert stock trading agent operating in the Indian equity market (NIFTY 50).
    At each step you receive:
      - market_data: OHLCV + RSI + MACD for each available symbol
      - portfolio: shares currently held per symbol
      - cash_balance: available cash (in INR)
      - feedback: result of your last action

    Your goal is to maximise portfolio value over the episode.

    You MUST respond with a single JSON object on one line — no markdown, no explanation:
    {"symbol": "<SYMBOL>", "action_type": "<buy|sell|hold>", "quantity": <0.0-1.0>}

    Rules:
    - symbol must be one of the symbols shown in market_data
    - action_type: "buy" spends quantity*cash, "sell" sells quantity*holdings, "hold" does nothing
    - quantity: a float in [0.0, 1.0]
    - Respond ONLY with the JSON. Nothing else.
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
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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


def _parse_action(text: str, valid_symbols: list) -> Optional[MarketAction]:
    """Extract and validate a MarketAction from the LLM's text response."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            l for l in lines if not l.strip().startswith("```")
        ).strip()
    try:
        data = json.loads(text)
        symbol = data.get("symbol", "")
        action_type = data.get("action_type", "hold")
        quantity = float(data.get("quantity", 0.5))
        # Validate symbol
        if symbol not in valid_symbols:
            symbol = valid_symbols[0] if valid_symbols else symbol
        action_type = action_type if action_type in ("buy", "sell", "hold") else "hold"
        quantity = max(0.0, min(1.0, quantity))
        return MarketAction(symbol=symbol, action_type=action_type, quantity=quantity)
    except Exception:
        # Fallback: hold the first valid symbol
        sym = valid_symbols[0] if valid_symbols else "RELIANCE.NS"
        return MarketAction(symbol=sym, action_type="hold", quantity=0.0)


def _get_model_action(client: OpenAI, obs, history: List[str]) -> MarketAction:
    user_prompt = _build_user_prompt(obs)
    valid_symbols = list(obs.market_data.keys())
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        text = ""

    action = _parse_action(text, valid_symbols)
    return action


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    if IMAGE_NAME:
        env = await MarketEnv.from_docker_image(IMAGE_NAME)
    else:
        base_url = os.getenv("ENV_BASE_URL", "http://localhost:8000")
        env = MarketEnv(base_url=base_url)
        await env.connect()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=TASK_NAME)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = _get_model_action(client, obs, history)
            action_str = f"{action.action_type}({action.symbol},{action.quantity:.2f})"

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            history.append(
                f"Step {step}: {action_str} → reward {reward:+.4f} | pv=₹{obs.portfolio_value:,.2f}"
            )

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())

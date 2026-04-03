---
title: Financial Market Environment
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Financial Market Environment

A system that teaches an AI (like Claude or ChatGPT) to trade stocks on the Indian stock market (NIFTY 50) and tests how well it performs.

---

## What Is This Project? (Simple Explanation)

Imagine you go to a stock broker and say:

> "Here is ₹10,000. I will show you the price of RELIANCE stock every day for 5 days.
> Based on what you see, decide: should I **buy**, **sell**, or **do nothing**?"

At the end of 5 days, we check — did you make money or lose money?

**That is exactly what this project does — but instead of a human broker, we use an AI.**

The AI reads today's stock prices, decides what to do, and we score it based on how well it traded.

---

## Difficulty Levels — Like a Video Game

| Level | Stocks Available | Trading Days | Starting Money | Scoring Method |
|-------|-----------------|--------------|----------------|----------------|
| `easy` | 1 (RELIANCE only) | 5 days | ₹10,000 | Simple profit (10% = perfect score) |
| `medium` | 3 (RELIANCE, TCS, HDFCBANK) | 10 days | ₹30,000 | Consistent gains (Sharpe Ratio) |
| `hard` | 5 stocks | 20 days | ₹50,000 | Profit minus penalty for big crashes |
| `nifty50` | All 50 NIFTY stocks | 30 days | ₹5,00,000 | Same as hard |

All scores are between **0.0 (worst) and 1.0 (perfect)**. A score above **0.3 counts as a pass**.

---

## How It Works — Step by Step

```
Every trading day, the AI receives this information:
┌─────────────────────────────────────────────────────┐
│ Day 3 | Cash: ₹5,000 | Portfolio Value: ₹11,200     │
│                                                      │
│ RELIANCE: price ₹1,457 | RSI: 29 | MACD: -7.74      │
│ TCS:      price ₹3,821 | RSI: 55 | MACD: +2.31      │
│                                                      │
│ You hold: 3.4 shares of RELIANCE                     │
│ Last action: Bought 3.4 shares at ₹1,457            │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
              AI decides and responds:
        {"symbol": "RELIANCE.NS",
         "action_type": "buy",
         "quantity": 0.5}         ← use 50% of cash to buy

                        │
                        ▼
          System executes the trade:
          - Calculates shares bought
          - Deducts cash
          - Moves to next day
          - Returns new prices + result
```

This repeats every day until the episode ends, then a final score is given.

---

## Action Space — What the AI Can Do

Every day, the AI submits exactly one action:

```json
{"symbol": "RELIANCE.NS", "action_type": "buy", "quantity": 0.5}
```

| Field | Options | Meaning |
|-------|---------|---------|
| `symbol` | Any available stock | Which stock to act on |
| `action_type` | `buy` | Spend `quantity × cash` to buy shares |
| `action_type` | `sell` | Sell `quantity × holdings` (e.g. 0.5 = sell half) |
| `action_type` | `hold` | Do nothing this day |
| `quantity` | 0.0 to 1.0 | Fraction of cash (buy) or holdings (sell) |

---

## What the AI Sees — Observation Space

Every day the AI receives:

```json
{
  "market_data": {
    "RELIANCE.NS": {
      "open": 1464.15,
      "high": 1476.10,
      "low":  1454.74,
      "close": 1457.63,
      "volume": 8574714,
      "rsi": 29.48,
      "macd": -7.7426,
      "macd_signal": -5.5421
    }
  },
  "portfolio": {"RELIANCE.NS": 5.456},
  "cash_balance": 2043.25,
  "portfolio_value": 10000.00,
  "task": "easy",
  "step_num": 1,
  "feedback": "Bought 5.4561 shares of RELIANCE.NS at ₹1457.63. Cash remaining: ₹2043.25."
}
```

### What is RSI and MACD? (Technical Indicators)

These are signals traders use to decide when to buy or sell:

| Indicator | Range | What it means |
|-----------|-------|---------------|
| **RSI** | 0 – 100 | Below 30 → stock is cheap, good to buy. Above 70 → stock is expensive, consider selling |
| **MACD** | Positive/Negative | Positive → price trending up. Negative → price trending down |

The AI uses these signals (along with prices) to decide its next action.

---

## Scoring — How the AI is Graded

### Easy Task
```
10% total return  →  Score 1.0  (full marks)
 5% total return  →  Score 0.5
 0% total return  →  Score 0.0
Lost money        →  Score 0.0
```

### Medium Task (Sharpe Ratio)
Not just *did you make money* but *did you make money consistently?*
```
Steady gains every day           →  High score
Big gain one day, big loss next  →  Low score (even if net profit is same)
```
This rewards disciplined trading over lucky bets.

### Hard and NIFTY 50 Task
```
Score = Total Return  −  (0.5 × Maximum Drawdown)

Drawdown = the biggest crash your portfolio had during the episode.

Example:
  Portfolio grew 10% overall but crashed 20% midway → penalised
  Portfolio grew 10% steadily with no crash         → near-perfect score
```

---

## Project Structure — What Each File Does

```
financial_market_env/
│
├── models.py                   ← Data templates (Action, Observation, State)
├── simulator.py                ← Stock price data (downloads from Yahoo Finance)
├── grader.py                   ← Scoring logic (easy / medium / hard / nifty50)
├── inference.py                ← Main script: AI agent loop
├── client.py                   ← Connects to the server over WebSocket
│
├── server/
│   ├── app.py                  ← Web server (exposes /reset, /step, /state endpoints)
│   └── market_environment.py  ← Core simulation engine (reset, step, portfolio logic)
│
├── data/
│   └── market_data.pkl        ← Cached historical NIFTY 50 data (auto-downloaded)
│
├── .env                        ← Your API keys and settings (never commit this)
├── .gitignore                  ← Ensures .env and cache files are not committed
├── Dockerfile                  ← For running the server in a container
└── pyproject.toml              ← Python project dependencies
```

---

## File-by-File Explanation

### `models.py` — The Forms / Templates
Defines what information flows between the AI and the market.

```python
MarketAction       # What the AI wants to do
  └── symbol       # Which stock
  └── action_type  # buy / sell / hold
  └── quantity     # How much (0.0 to 1.0)

MarketObservation  # Daily report the AI receives
  └── market_data  # Prices + RSI + MACD for each stock
  └── portfolio    # Shares currently held
  └── cash_balance # Cash available
  └── feedback     # Human-readable result of last action

MarketState        # Session metadata
  └── task         # easy / medium / hard / nifty50
  └── max_steps    # Total trading days
  └── symbols      # Available stocks this episode
```

---

### `simulator.py` — The Stock Exchange Data Provider
Downloads and serves historical NIFTY 50 stock prices.

**Data source:** Yahoo Finance (2022–2024 historical data)
**Cached at:** `data/market_data.pkl` (downloaded once, reused every run)
**Fallback:** If no internet, generates realistic synthetic price data

**Stocks available per task:**
```python
easy    → ["RELIANCE.NS"]
medium  → ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
hard    → ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"]
nifty50 → All 50 NIFTY constituent stocks
```

**All 50 NIFTY stocks:**
```
ADANIENT, ADANIPORTS, APOLLOHOSP, ASIANPAINT, AXISBANK,
BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BHARTIARTL, BPCL,
BRITANNIA, CIPLA, COALINDIA, DIVISLAB, DRREDDY,
EICHERMOT, GRASIM, HCLTECH, HDFCBANK, HDFCLIFE,
HEROMOTOCO, HINDALCO, HINDUNILVR, ICICIBANK, INDUSINDBK,
INFY, ITC, JSWSTEEL, KOTAKBANK, LT,
M&M, MARUTI, NESTLEIND, NTPC, ONGC,
POWERGRID, RELIANCE, SBILIFE, SBIN, SHRIRAMFIN,
SUNPHARMA, TATACONSUM, TATAMOTORS, TATASTEEL, TCS,
TECHM, TITAN, ULTRACEMCO, WIPRO, ZOMATO
```

Each episode picks a **random window** of consecutive days from the historical data, so no two episodes are the same.

---

### `grader.py` — The Examiner / Scorer

Calculates the score at the end of each episode.

```
compute_step_reward()   → Small reward given every day (only for positive gains)
compute_final_reward()  → Big reward given at episode end (main score)
```

**Per-step reward formula:**
```
reward = (today's portfolio value - yesterday's value) / starting value
Only positive. Losses give 0, not negative.
```

**Final reward per task:**
- `easy`   → `min(return × 10, 1.0)`  — 10% return = perfect score
- `medium` → Sharpe ratio pushed through sigmoid curve
- `hard`   → `(total return + 0.5) - (0.5 × max drawdown)` — penalises crashes
- `nifty50`→ Same formula as hard

---

### `server/market_environment.py` — The Core Game Engine

This is where the actual trading simulation runs.

**`reset(task="easy")`**
- Sets up portfolio with zero shares
- Loads cash based on task level
- Picks a random historical window from the data
- Returns the first day's market data

**`step(action)`**
- Executes the AI's buy/sell/hold decision
- Updates cash and portfolio
- Advances to next day
- Returns new prices + reward

**Buy logic:**
```
shares_bought = (cash × quantity) / price
cash          = cash - (cash × quantity)
portfolio     = portfolio + shares_bought
```

**Sell logic:**
```
shares_sold = holdings × quantity
cash        = cash + (shares_sold × price)
portfolio   = portfolio - shares_sold
```

Supports up to **16 concurrent sessions** — multiple AIs can trade simultaneously.

---

### `server/app.py` — The Web Server

Exposes the trading environment as a web API.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/reset` | POST | Start a new episode |
| `/step`  | POST | Submit a trading action |
| `/state` | GET  | Check current session state |
| `/schema`| GET  | See action/observation format |
| `/ws`    | WebSocket | Persistent connection (used by client.py) |

---

### `client.py` — The Connection Layer

`MarketEnv` connects to the server and provides a clean Python API:

```python
env = MarketEnv(base_url="http://localhost:8000")
await env.connect()

result = await env.reset(task="medium")
result = await env.step(MarketAction(symbol="RELIANCE.NS", action_type="buy", quantity=0.8))
print(result.reward)
```

Uses **WebSocket** (not regular HTTP) for lower latency across multiple steps.

---

### `inference.py` — The AI Agent (Main Script)

This is the script you run. It orchestrates the entire trading episode:

```
1. Connect to the market server
2. Start episode (reset with chosen task)
3. Every day:
   a. Format today's prices + portfolio into a readable prompt
   b. Send prompt to Claude / Qwen / any LLM
   c. AI responds with JSON action
   d. Send action to server
   e. Receive new prices + reward
   f. Log the step
4. Print final score
```

**The prompt sent to the AI looks like:**
```
Step 3 | Task: medium | Cash: ₹12,450 | Portfolio value: ₹31,200

Market data:
  RELIANCE.NS: close=₹1457, RSI=29.48, MACD=-7.7426
  TCS.NS:      close=₹3821, RSI=55.12, MACD=2.3100
  HDFCBANK.NS: close=₹1612, RSI=42.30, MACD=0.8821

Holdings:
  RELIANCE.NS: 5.4561 shares

Last feedback: Bought 5.4561 shares of RELIANCE.NS at ₹1457.

Available symbols: RELIANCE.NS, TCS.NS, HDFCBANK.NS
Respond with ONE JSON action.
```

---

## Understanding the Output

```
[START] task=medium env=financial_market_env model=claude-haiku-4-5-20251001
```
→ Medium level game starting. Claude Haiku is the trader.

```
[STEP] step=1 action=buy(HDFCBANK.NS,0.80) reward=0.00 done=false error=null
```
→ Day 1: AI bought HDFCBANK using 80% of cash.
  Portfolio did not grow yet (market hadn't moved). Episode still running.

```
[STEP] step=5 action=hold(RELIANCE.NS,0.00) reward=0.03 done=false error=null
```
→ Day 5: AI did nothing. But RELIANCE price went up → portfolio grew 3% of starting value.

```
[STEP] step=10 action=sell(HDFCBANK.NS,0.30) reward=0.88 done=true error=null
```
→ Day 10 (last day): AI sold 30% of HDFCBANK holdings.
  Huge final reward (0.88) because overall Sharpe ratio was excellent.

```
[END] success=true steps=10 score=1.000 rewards=0.00,0.00,0.00,0.01,0.03,...,0.88
```
→ Perfect score! AI passed the medium task.
  `rewards` = per-day reward breakdown.
  `score` = total rewards ÷ maximum possible rewards.

**Success threshold: score ≥ 0.3**

---

## Setup

### 1. Install dependencies
```bash
pip install "openenv-core[core]>=0.2.1" yfinance numpy pandas fastapi uvicorn openai
```

### 2. Configure your API key in `.env`
```bash
# Anthropic (Claude)
API_KEY=sk-ant-your-key-here
API_BASE_URL=https://api.anthropic.com/v1
MODEL_NAME=claude-haiku-4-5-20251001

# Environment server
ENV_BASE_URL=http://localhost:8000
```

### 3. Start the server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Run a task (in a new terminal)
```bash
export $(grep -v '^#' .env | xargs)

MARKET_TASK=easy    MARKET_MAX_STEPS=5  python3 inference.py
MARKET_TASK=medium  MARKET_MAX_STEPS=10 python3 inference.py
MARKET_TASK=hard    MARKET_MAX_STEPS=20 python3 inference.py
MARKET_TASK=nifty50 MARKET_MAX_STEPS=30 python3 inference.py
```

### Docker (alternative)
```bash
docker build -t financial-market-env:latest .
docker run -p 8000:8000 financial-market-env:latest
```

---

## Supported LLM Providers

Any OpenAI-compatible API works. Just change `.env`:

| Provider | API_BASE_URL | MODEL_NAME |
|----------|-------------|------------|
| Anthropic (Claude) | `https://api.anthropic.com/v1` | `claude-haiku-4-5-20251001` |
| HuggingFace | `https://router.huggingface.co/v1` | `Qwen/Qwen2.5-72B-Instruct` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| Ollama (local, free) | `http://localhost:11434/v1` | `qwen2.5:7b` |

---

## Performance Results (Claude Haiku)

| Task | Stocks | Steps | Score | Pass |
|------|--------|-------|-------|------|
| Easy | 1 | 5 | ~0.1–1.0 | Varies by market window |
| Medium | 3 | 10 | 1.000 | ✅ Consistent |
| Hard | 5 | 20 | 1.000 | ✅ Consistent |
| NIFTY 50 | 50 | 30 | ~0.75 | ✅ Good |

**Key observations:**
- Easy is hardest to score well on — only 1 stock, 5 days, heavy market dependency
- Medium rewards consistent trading; Claude actively rotates between stocks
- Hard and NIFTY 50 reward risk management; Claude avoids large drawdowns

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | — | Your LLM API key (checked first) |
| `HF_TOKEN` | — | HuggingFace token (fallback if API_KEY not set) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model to use |
| `ENV_BASE_URL` | `http://localhost:8000` | Environment server URL |
| `MARKET_TASK` | `easy` | Task: `easy`, `medium`, `hard`, `nifty50` |
| `MARKET_MAX_STEPS` | `8` | Hard cap on steps (set to match task: 5/10/20/30) |
| `LOCAL_IMAGE_NAME` | — | Docker image name (if running server via Docker) |

---

## Data

- **Source:** Yahoo Finance historical data (2022–2024)
- **Cache:** `data/market_data.pkl` — downloaded once, reused every run
- **Offline fallback:** Synthetic price data (random walk simulation) if no internet
- **Note:** TATAMOTORS.NS and ZOMATO.NS may fall back to synthetic data due to Yahoo Finance availability issues

---

## Architecture Overview

```
You run inference.py
       │
       │  Formats stock data + portfolio into a text prompt
       ▼
 Claude / Qwen / any LLM
       │
       │  Responds: {"symbol": "RELIANCE.NS", "action_type": "buy", "quantity": 0.8}
       ▼
  client.py  ──WebSocket──►  server/app.py
                                    │
                          market_environment.py
                            ├── simulator.py   (gets historical prices)
                            └── grader.py      (calculates reward/score)
                                    │
                          Returns: new prices, reward, done
       │
       ▼
inference.py logs:
[STEP] step=1 action=buy(RELIANCE.NS,0.80) reward=0.01 done=false
```

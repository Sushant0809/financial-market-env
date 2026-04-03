"""
Market Data Simulator for the Financial Market Environment.

Downloads and caches historical NIFTY 50 stock data from Yahoo Finance.
Falls back to synthetic data if download fails (e.g. in restricted environments).
"""

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# All NIFTY 50 constituent symbols (Yahoo Finance format)
NIFTY_50_SYMBOLS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS", "BPCL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS",
    "INFY.NS", "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ZOMATO.NS",
]

# Task configuration
TASK_SYMBOLS = {
    "easy": ["RELIANCE.NS"],
    "medium": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"],
    "hard": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "SBIN.NS"],
    "nifty50": NIFTY_50_SYMBOLS,
}

TASK_STEPS = {"easy": 10, "medium": 30, "hard": 40, "nifty50": 30}
TASK_INITIAL_CASH = {"easy": 10000.0, "medium": 30000.0, "hard": 50000.0, "nifty50": 500000.0}

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_CACHE_FILE = os.path.join(_DATA_DIR, "market_data.pkl")


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI and MACD indicators and append to dataframe."""
    close = df["Close"].squeeze()

    # RSI (14-period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df = df.copy()
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def _generate_synthetic(symbol: str, n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for a symbol when real data is unavailable."""
    rng = np.random.RandomState(seed + abs(hash(symbol)) % 1000)
    base_price = rng.uniform(500, 3000)
    prices = [base_price]
    for _ in range(n - 1):
        change = rng.normal(0.0005, 0.015)
        prices.append(prices[-1] * (1 + change))

    df = pd.DataFrame(
        {
            "Open": [p * rng.uniform(0.995, 1.005) for p in prices],
            "High": [p * rng.uniform(1.005, 1.020) for p in prices],
            "Low": [p * rng.uniform(0.980, 0.995) for p in prices],
            "Close": prices,
            "Volume": rng.randint(500_000, 5_000_000, size=n).astype(float),
        }
    )
    return _add_indicators(df).dropna().reset_index(drop=True)


def _download_data() -> Dict[str, pd.DataFrame]:
    """Download historical data for all symbols via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return {}

    all_symbols = list({s for syms in TASK_SYMBOLS.values() for s in syms})
    result: Dict[str, pd.DataFrame] = {}
    for symbol in all_symbols:
        try:
            df = yf.download(
                symbol,
                start="2022-01-01",
                end="2024-12-31",
                progress=False,
                auto_adjust=True,
            )
            if df is None or len(df) < 60:
                continue
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = _add_indicators(df)
            df = df.dropna().reset_index(drop=True)
            if len(df) >= 60:
                result[symbol] = df
        except Exception:
            pass
    return result


class MarketDataSimulator:
    """
    Provides episode windows of market data for RL training.

    Loads from a local pickle cache if available, otherwise downloads
    from Yahoo Finance and caches. Falls back to synthetic data if
    download fails entirely.
    """

    def __init__(self) -> None:
        self._data: Dict[str, pd.DataFrame] = {}
        self._load()

    def _load(self) -> None:
        os.makedirs(_DATA_DIR, exist_ok=True)

        # Try loading from cache
        if os.path.exists(_CACHE_FILE):
            try:
                with open(_CACHE_FILE, "rb") as f:
                    self._data = pickle.load(f)
                if self._data:
                    return
            except Exception:
                pass

        # Download from Yahoo Finance
        downloaded = _download_data()
        if downloaded:
            self._data = downloaded
            try:
                with open(_CACHE_FILE, "wb") as f:
                    pickle.dump(self._data, f)
            except Exception:
                pass
            return

        # Fall back to synthetic data
        all_symbols = list({s for syms in TASK_SYMBOLS.values() for s in syms})
        for sym in all_symbols:
            self._data[sym] = _generate_synthetic(sym)

    def get_episode_window(
        self,
        symbols: List[str],
        max_steps: int,
        seed: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Return a random window of `max_steps + 1` rows for each symbol.

        The extra row ensures we can compute the price after the last action.
        """
        rng = random.Random(seed)
        window_size = max_steps + 1

        # Ensure we have data for all symbols (use synthetic if missing)
        for sym in symbols:
            if sym not in self._data:
                self._data[sym] = _generate_synthetic(sym)

        min_len = min(len(self._data[sym]) for sym in symbols)
        if min_len < window_size:
            start_idx = 0
        else:
            start_idx = rng.randint(0, min_len - window_size)

        return {
            sym: self._data[sym].iloc[start_idx : start_idx + window_size].copy()
            for sym in symbols
        }

    @staticmethod
    def get_prices(window: Dict[str, pd.DataFrame], step: int) -> Dict[str, float]:
        """Get closing prices for all symbols at a given step."""
        return {sym: float(df.iloc[step]["Close"]) for sym, df in window.items()}

    @staticmethod
    def get_market_data(window: Dict[str, pd.DataFrame], step: int) -> Dict[str, dict]:
        """Get full OHLCV + indicator data for all symbols at a given step."""
        result: Dict[str, dict] = {}
        for sym, df in window.items():
            row = df.iloc[step]
            result[sym] = {
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(float(row["Volume"])),
                "rsi": round(float(row.get("RSI", 50.0)), 2),
                "macd": round(float(row.get("MACD", 0.0)), 4),
                "macd_signal": round(float(row.get("MACD_Signal", 0.0)), 4),
            }
        return result

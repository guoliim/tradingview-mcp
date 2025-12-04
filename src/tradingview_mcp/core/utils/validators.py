from __future__ import annotations
import os
from typing import Set

ALLOWED_TIMEFRAMES: Set[str] = {"5m", "15m", "1h", "4h", "1D", "1W", "1M"}

# Simplified to support only US and Hong Kong stock markets
EXCHANGE_SCREENER = {
    "nasdaq": "america",    # NASDAQ Stock Exchange
    "nyse": "america",      # New York Stock Exchange
    "hkex": "hongkong",     # Hong Kong Exchange
    "hk": "hongkong",       # Hong Kong (alternate alias)
}


def get_market_for_exchange(exchange: str) -> str:
    """Get the TradingView market type for a given exchange."""
    return EXCHANGE_SCREENER.get(exchange.lower(), "america")

# Get absolute path to coinlist directory relative to this module
# This file is at: src/tradingview_mcp/core/utils/validators.py
# We want: src/tradingview_mcp/coinlist/
_this_file = __file__
_utils_dir = os.path.dirname(_this_file)  # core/utils
_core_dir = os.path.dirname(_utils_dir)   # core  
_package_dir = os.path.dirname(_core_dir) # tradingview_mcp
COINLIST_DIR = os.path.join(_package_dir, 'coinlist')


def sanitize_timeframe(tf: str, default: str = "5m") -> str:
    if not tf:
        return default
    tfs = tf.strip()
    return tfs if tfs in ALLOWED_TIMEFRAMES else default


def sanitize_exchange(ex: str, default: str = "nasdaq") -> str:
    if not ex:
        return default
    exs = ex.strip().lower()
    return exs if exs in EXCHANGE_SCREENER else default

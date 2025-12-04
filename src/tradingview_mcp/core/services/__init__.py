# Services module exports
from .auth import SecureAuthManager, get_auth_manager, get_cookies
from .indicators import compute_metrics
from .coinlist import load_symbols
from .screener_provider import fetch_screener_indicators, fetch_screener_multi_changes
from .watchlist import WatchlistManager, get_watchlist_manager
from .support_resistance import (
    SupportResistanceCalculator,
    calculate_support_resistance_for_symbol,
    batch_support_resistance,
)

__all__ = [
    "SecureAuthManager",
    "get_auth_manager",
    "get_cookies",
    "compute_metrics",
    "load_symbols",
    "fetch_screener_indicators",
    "fetch_screener_multi_changes",
    "WatchlistManager",
    "get_watchlist_manager",
    "SupportResistanceCalculator",
    "calculate_support_resistance_for_symbol",
    "batch_support_resistance",
]

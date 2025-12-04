"""
TradingView Authentication via rookiepy

Automatically extracts TradingView session cookies from the user's browser.
Supports Chrome, Edge, Firefox, Brave, Chromium, and Opera.
"""

from __future__ import annotations

import logging
from typing import Optional, List
from http.cookiejar import CookieJar

logger = logging.getLogger(__name__)

# Supported browsers in order of preference
SUPPORTED_BROWSERS = ["chrome", "edge", "firefox", "brave", "chromium", "opera"]

# Cache for cookies
_cookies_cache: Optional[CookieJar] = None


def get_browser_cookies(browser: str = "chrome") -> Optional[CookieJar]:
    """
    Extract TradingView cookies from specified browser.

    Args:
        browser: Browser name (chrome, edge, firefox, brave, chromium, opera)

    Returns:
        CookieJar with TradingView cookies, or None if not found.
    """
    try:
        import rookiepy
    except ImportError:
        logger.error("rookiepy not installed. Run: pip install rookiepy")
        return None

    browser = browser.lower()

    try:
        # Get browser-specific cookie function
        browser_funcs = {
            "chrome": rookiepy.chrome,
            "edge": rookiepy.edge,
            "firefox": rookiepy.firefox,
            "brave": rookiepy.brave,
            "chromium": rookiepy.chromium,
            "opera": rookiepy.opera,
        }

        if browser not in browser_funcs:
            logger.error(f"Unsupported browser: {browser}")
            return None

        # Extract cookies for tradingview.com
        raw_cookies = browser_funcs[browser]([".tradingview.com"])
        cookies = rookiepy.to_cookiejar(raw_cookies)

        # Check if sessionid exists
        has_session = any(c.name == "sessionid" for c in cookies)
        if not has_session:
            logger.debug(f"No TradingView session found in {browser}")
            return None

        logger.info(f"TradingView cookies loaded from {browser}")
        return cookies

    except Exception as e:
        logger.debug(f"Failed to get cookies from {browser}: {e}")
        return None


def get_cookies() -> Optional[CookieJar]:
    """
    Get TradingView cookies, trying multiple browsers.

    Returns:
        CookieJar with session cookies, or None if not authenticated.
    """
    global _cookies_cache

    if _cookies_cache is not None:
        return _cookies_cache

    # Try each browser in order
    for browser in SUPPORTED_BROWSERS:
        cookies = get_browser_cookies(browser)
        if cookies:
            _cookies_cache = cookies
            return cookies

    logger.info("No TradingView session found in any browser - using delayed data")
    return None


def get_auth_status() -> dict:
    """
    Get current authentication status.

    Returns:
        Dict with authentication info (does not expose actual cookies).
    """
    cookies = get_cookies()

    if cookies:
        # Find which browser had the cookies
        browser_used = "unknown"
        for browser in SUPPORTED_BROWSERS:
            test_cookies = get_browser_cookies(browser)
            if test_cookies:
                browser_used = browser
                break

        return {
            "authenticated": True,
            "browser": browser_used,
            "data_mode": "realtime",
            "message": f"Session loaded from {browser_used} browser",
        }
    else:
        return {
            "authenticated": False,
            "data_mode": "delayed",
            "message": "No TradingView session found",
            "instructions": {
                "step1": "Login to tradingview.com in your browser",
                "step2": "Ensure 'Stay signed in' is checked",
                "step3": "Restart the MCP server to detect cookies",
            },
            "supported_browsers": SUPPORTED_BROWSERS,
        }


def refresh_cookies(browser: str = "chrome") -> dict:
    """
    Refresh cookies from browser.

    Args:
        browser: Specific browser to use, or "auto" to try all.

    Returns:
        Updated auth status.
    """
    global _cookies_cache
    _cookies_cache = None

    if browser.lower() == "auto":
        get_cookies()
    else:
        cookies = get_browser_cookies(browser)
        if cookies:
            _cookies_cache = cookies

    return get_auth_status()

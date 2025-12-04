"""
TradingView Authentication via rookiepy

Automatically extracts TradingView session cookies from the user's browser.
User just needs to be logged in to TradingView in their browser.

Supported browsers: Chrome, Firefox, Edge, Opera, Chromium, Brave
"""

from __future__ import annotations

import logging
from typing import Optional, Any
from http.cookiejar import CookieJar

logger = logging.getLogger(__name__)

# Cache for cookies to avoid repeated browser access
_cookies_cache: Optional[CookieJar] = None
_auth_status: Optional[dict] = None


def get_browser_cookies(browser: str = "chrome") -> Optional[CookieJar]:
    """
    Extract TradingView cookies from user's browser.

    Args:
        browser: Browser to extract from ('chrome', 'firefox', 'edge', 'opera', 'chromium', 'brave')

    Returns:
        CookieJar with TradingView session cookies, or None if not available.
    """
    global _cookies_cache, _auth_status

    # Return cached cookies if available
    if _cookies_cache is not None:
        return _cookies_cache

    try:
        import rookiepy

        # Map browser names to rookiepy functions
        browser_funcs = {
            "chrome": rookiepy.chrome,
            "firefox": rookiepy.firefox,
            "edge": rookiepy.edge,
            "opera": rookiepy.opera,
            "chromium": rookiepy.chromium,
            "brave": rookiepy.brave,
        }

        browser_func = browser_funcs.get(browser.lower())
        if not browser_func:
            logger.warning(f"Unknown browser: {browser}, falling back to chrome")
            browser_func = rookiepy.chrome

        # Extract cookies for tradingview.com domain
        cookies = rookiepy.to_cookiejar(browser_func(['.tradingview.com']))

        # Verify we have the session cookie
        has_session = any(c.name == 'sessionid' for c in cookies)

        if has_session:
            _cookies_cache = cookies
            _auth_status = {
                "authenticated": True,
                "browser": browser,
                "message": f"Session cookies loaded from {browser}",
            }
            logger.info(f"TradingView cookies loaded from {browser}")
            return cookies
        else:
            _auth_status = {
                "authenticated": False,
                "browser": browser,
                "message": f"No TradingView session found in {browser}. Please login to tradingview.com",
            }
            logger.warning(f"No TradingView session found in {browser}")
            return None

    except ImportError:
        _auth_status = {
            "authenticated": False,
            "error": "rookiepy not installed",
            "message": "Run: pip install rookiepy",
        }
        logger.error("rookiepy not installed")
        return None
    except Exception as e:
        _auth_status = {
            "authenticated": False,
            "error": str(type(e).__name__),
            "message": f"Failed to extract cookies: {e}",
        }
        logger.error(f"Failed to extract browser cookies: {e}")
        return None


def get_cookies() -> Optional[CookieJar]:
    """
    Get TradingView cookies, trying multiple browsers.

    Returns:
        CookieJar or None
    """
    global _cookies_cache

    if _cookies_cache is not None:
        return _cookies_cache

    # Try browsers in order of popularity
    browsers = ["chrome", "edge", "firefox", "brave", "chromium", "opera"]

    for browser in browsers:
        try:
            cookies = get_browser_cookies(browser)
            if cookies:
                return cookies
        except Exception:
            continue

    return None


def get_auth_status() -> dict:
    """
    Get current authentication status.

    Returns:
        Dict with authentication info (never exposes actual cookies).
    """
    global _auth_status

    if _auth_status is None:
        # Trigger cookie loading to populate status
        get_cookies()

    return _auth_status or {
        "authenticated": False,
        "message": "Not checked yet",
    }


def clear_cache() -> None:
    """Clear the cookies cache to force refresh on next request."""
    global _cookies_cache, _auth_status
    _cookies_cache = None
    _auth_status = None
    logger.info("Auth cache cleared")


def refresh_cookies(browser: str = "chrome") -> dict:
    """
    Force refresh cookies from browser.

    Args:
        browser: Browser to use

    Returns:
        Updated auth status
    """
    clear_cache()
    get_browser_cookies(browser)
    return get_auth_status()

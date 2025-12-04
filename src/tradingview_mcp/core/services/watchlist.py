"""
TradingView Watchlist Manager

Provides functionality to interact with user's TradingView watchlists.
Requires authentication via TV_SESSION_ID environment variable.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any

import requests

from .auth import get_auth_manager

logger = logging.getLogger(__name__)


class WatchlistManager:
    """
    Manager for TradingView Watchlist operations.

    Requires authenticated session (TV_SESSION_ID) to access user data.
    All operations gracefully handle authentication failures.
    """

    BASE_URL = "https://www.tradingview.com"
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

    def __init__(self):
        """Initialize watchlist manager with auth manager reference."""
        self.auth = get_auth_manager()

    def _get_headers(self) -> Dict[str, str]:
        """Get common headers for API requests."""
        return {
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json",
            "Referer": "https://www.tradingview.com/",
        }

    def _check_auth(self) -> Optional[Dict[str, str]]:
        """Check authentication and return cookies or None."""
        cookies = self.auth.get_cookies()
        if not cookies:
            logger.warning("Watchlist access requires authentication")
            return None
        return cookies

    def get_watchlists(self) -> Dict[str, Any]:
        """
        Get all user watchlists from TradingView account.

        Returns:
            Dict containing list of watchlists or error message.
            Each watchlist has: id, name, symbols count.
        """
        cookies = self._check_auth()
        if not cookies:
            return {
                "error": "Not authenticated",
                "message": "Set TV_SESSION_ID environment variable to access watchlists",
                "authenticated": False,
            }

        try:
            # TradingView watchlist API endpoint
            response = requests.get(
                f"{self.BASE_URL}/api/v1/symbols_list/",
                cookies=cookies,
                headers=self._get_headers(),
                timeout=15,
            )

            if response.status_code == 401:
                return {
                    "error": "Session expired",
                    "message": "Please update TV_SESSION_ID with a fresh session",
                    "authenticated": False,
                }

            if response.status_code == 403:
                return {
                    "error": "Access denied",
                    "message": "Session may be invalid or expired",
                    "authenticated": False,
                }

            if not response.ok:
                return {
                    "error": f"API error: {response.status_code}",
                    "authenticated": True,
                }

            data = response.json()

            # Parse watchlists from response
            watchlists = []
            if isinstance(data, list):
                for item in data:
                    watchlist_info = {
                        "id": item.get("id"),
                        "name": item.get("name", "Unnamed"),
                        "symbols_count": len(item.get("symbols", [])),
                    }
                    watchlists.append(watchlist_info)
            elif isinstance(data, dict):
                # Handle different response format
                for key, value in data.items():
                    if isinstance(value, dict):
                        watchlist_info = {
                            "id": key,
                            "name": value.get("name", key),
                            "symbols_count": len(value.get("symbols", [])),
                        }
                        watchlists.append(watchlist_info)

            return {
                "authenticated": True,
                "count": len(watchlists),
                "watchlists": watchlists,
            }

        except requests.Timeout:
            return {"error": "Request timeout", "authenticated": True}
        except requests.RequestException as e:
            logger.error(f"Watchlist fetch error: {type(e).__name__}")
            return {"error": "Network error", "authenticated": True}
        except Exception as e:
            logger.error(f"Unexpected error: {type(e).__name__}")
            return {"error": "Failed to parse response", "authenticated": True}

    def get_watchlist_symbols(self, watchlist_id: str) -> Dict[str, Any]:
        """
        Get symbols from a specific watchlist.

        Args:
            watchlist_id: The watchlist ID (from get_watchlists or URL)

        Returns:
            Dict with symbols list or error message.
        """
        cookies = self._check_auth()
        if not cookies:
            return {
                "error": "Not authenticated",
                "message": "Set TV_SESSION_ID environment variable",
                "authenticated": False,
            }

        try:
            # Try to fetch specific watchlist
            response = requests.get(
                f"{self.BASE_URL}/api/v1/symbols_list/{watchlist_id}/",
                cookies=cookies,
                headers=self._get_headers(),
                timeout=15,
            )

            if response.status_code == 404:
                return {
                    "error": "Watchlist not found",
                    "watchlist_id": watchlist_id,
                    "authenticated": True,
                }

            if not response.ok:
                return {
                    "error": f"API error: {response.status_code}",
                    "watchlist_id": watchlist_id,
                    "authenticated": True,
                }

            data = response.json()

            # Extract symbols from response
            symbols = []
            if isinstance(data, dict):
                symbols = data.get("symbols", [])
                name = data.get("name", "Unknown")
            elif isinstance(data, list):
                symbols = data
                name = "Unknown"

            # Clean up symbol format (remove exchange prefix if needed)
            clean_symbols = []
            for sym in symbols:
                if isinstance(sym, str):
                    clean_symbols.append(sym)
                elif isinstance(sym, dict):
                    clean_symbols.append(sym.get("s", sym.get("symbol", str(sym))))

            return {
                "authenticated": True,
                "watchlist_id": watchlist_id,
                "name": name,
                "count": len(clean_symbols),
                "symbols": clean_symbols,
            }

        except requests.RequestException as e:
            logger.error(f"Watchlist symbols fetch error: {type(e).__name__}")
            return {"error": "Network error", "authenticated": True}
        except Exception as e:
            logger.error(f"Parse error: {type(e).__name__}")
            return {"error": "Failed to parse response", "authenticated": True}

    def add_to_watchlist(
        self, watchlist_id: str, symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Add symbols to a watchlist.

        Args:
            watchlist_id: Target watchlist ID
            symbols: List of symbols to add (e.g., ["NASDAQ:AAPL", "NYSE:IBM"])

        Returns:
            Dict with operation result.
        """
        cookies = self._check_auth()
        if not cookies:
            return {
                "error": "Not authenticated",
                "authenticated": False,
            }

        if not symbols:
            return {"error": "No symbols provided", "authenticated": True}

        try:
            # Format symbols for API
            formatted_symbols = []
            for sym in symbols:
                if ":" not in sym:
                    # Default to NASDAQ if no exchange specified
                    sym = f"NASDAQ:{sym.upper()}"
                formatted_symbols.append(sym.upper())

            response = requests.post(
                f"{self.BASE_URL}/api/v1/symbols_list/{watchlist_id}/append/",
                cookies=cookies,
                headers={
                    **self._get_headers(),
                    "Content-Type": "application/json",
                },
                json={"symbols": formatted_symbols},
                timeout=15,
            )

            if response.ok:
                return {
                    "success": True,
                    "watchlist_id": watchlist_id,
                    "added_symbols": formatted_symbols,
                    "authenticated": True,
                }
            else:
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}",
                    "watchlist_id": watchlist_id,
                    "authenticated": True,
                }

        except requests.RequestException as e:
            logger.error(f"Add to watchlist error: {type(e).__name__}")
            return {"error": "Network error", "authenticated": True}

    def remove_from_watchlist(
        self, watchlist_id: str, symbols: List[str]
    ) -> Dict[str, Any]:
        """
        Remove symbols from a watchlist.

        Args:
            watchlist_id: Target watchlist ID
            symbols: List of symbols to remove

        Returns:
            Dict with operation result.
        """
        cookies = self._check_auth()
        if not cookies:
            return {
                "error": "Not authenticated",
                "authenticated": False,
            }

        try:
            response = requests.post(
                f"{self.BASE_URL}/api/v1/symbols_list/{watchlist_id}/remove/",
                cookies=cookies,
                headers={
                    **self._get_headers(),
                    "Content-Type": "application/json",
                },
                json={"symbols": [s.upper() for s in symbols]},
                timeout=15,
            )

            return {
                "success": response.ok,
                "watchlist_id": watchlist_id,
                "removed_symbols": symbols,
                "authenticated": True,
            }

        except requests.RequestException as e:
            logger.error(f"Remove from watchlist error: {type(e).__name__}")
            return {"error": "Network error", "authenticated": True}

    def get_user_info(self) -> Dict[str, Any]:
        """
        Get current user information.

        Returns:
            Dict with user info (username, subscription plan, etc.)
        """
        cookies = self._check_auth()
        if not cookies:
            return {
                "error": "Not authenticated",
                "authenticated": False,
            }

        try:
            response = requests.get(
                f"{self.BASE_URL}/api/v1/account/",
                cookies=cookies,
                headers=self._get_headers(),
                timeout=15,
            )

            if not response.ok:
                # Try alternative endpoint
                response = requests.get(
                    f"{self.BASE_URL}/u/",
                    cookies=cookies,
                    headers=self._get_headers(),
                    timeout=15,
                    allow_redirects=False,
                )

                if response.status_code == 302:
                    return {
                        "error": "Session expired",
                        "authenticated": False,
                    }

            # Extract basic info from auth manager
            auth_status = self.auth.get_status()

            return {
                "authenticated": True,
                "user": auth_status.get("account", {}),
            }

        except requests.RequestException as e:
            logger.error(f"User info fetch error: {type(e).__name__}")
            return {"error": "Network error", "authenticated": True}


# Module-level convenience functions
_watchlist_manager: Optional[WatchlistManager] = None


def get_watchlist_manager() -> WatchlistManager:
    """Get singleton watchlist manager instance."""
    global _watchlist_manager
    if _watchlist_manager is None:
        _watchlist_manager = WatchlistManager()
    return _watchlist_manager

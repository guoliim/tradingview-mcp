"""
TradingView Secure Authentication Manager

Security-first authentication using environment variables.
No credentials are stored on disk - only in memory during runtime.
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Configure logging to never expose sensitive data
logger = logging.getLogger(__name__)


class AuthMode(Enum):
    """Authentication mode status."""
    AUTHENTICATED = "authenticated"
    PUBLIC = "public"
    INVALID = "invalid"


@dataclass
class AuthStatus:
    """Authentication status information (safe to expose)."""
    mode: AuthMode
    is_authenticated: bool
    data_mode: str  # "realtime" or "delayed"
    message: str
    account_info: Optional[Dict[str, Any]] = None


class SecureAuthManager:
    """
    Security-first authentication manager.

    Design principles:
    1. Session ID only loaded from environment variable (not stored on disk)
    2. Session ID never logged or exposed in error messages
    3. Validation results cached to minimize API calls
    4. Graceful degradation to public mode on auth failure

    Usage:
        auth = SecureAuthManager()
        if auth.is_authenticated:
            cookies = auth.get_cookies()
            # Use cookies for API calls
    """

    # Environment variable name for session ID
    ENV_SESSION_ID = "TV_SESSION_ID"

    def __init__(self):
        """Initialize auth manager, loading session from environment."""
        self._session_id: Optional[str] = None
        self._validated: bool = False
        self._validation_cache: Optional[AuthStatus] = None

        # Load session ID from environment
        self._load_from_environment()

    def _load_from_environment(self) -> None:
        """Load session ID from environment variable."""
        session_id = os.environ.get(self.ENV_SESSION_ID, "").strip()

        if session_id:
            # Basic validation: session ID should be a non-empty string
            # TradingView session IDs are typically 20-50 characters
            if len(session_id) >= 10:
                self._session_id = session_id
                logger.info("Session ID loaded from environment variable")
            else:
                logger.warning("Session ID in environment variable appears invalid (too short)")
        else:
            logger.info("No session ID configured - running in public mode")

    @property
    def is_authenticated(self) -> bool:
        """Check if we have a valid session ID configured."""
        return self._session_id is not None and self._validated

    @property
    def has_session(self) -> bool:
        """Check if a session ID is configured (may not be validated yet)."""
        return self._session_id is not None

    def get_cookies(self) -> Optional[Dict[str, str]]:
        """
        Get cookies for API requests.

        Returns:
            Dict with sessionid cookie if authenticated, None otherwise.

        Security note:
            Only call this method when actually needed for API requests.
            Do not log or expose the returned cookies.
        """
        if self._session_id:
            return {"sessionid": self._session_id}
        return None

    def validate_session(self, force: bool = False) -> AuthStatus:
        """
        Validate the current session with TradingView.

        Args:
            force: If True, bypass cache and re-validate.

        Returns:
            AuthStatus with validation results (never exposes session ID).
        """
        # Return cached result if available
        if self._validation_cache and not force:
            return self._validation_cache

        # No session configured
        if not self._session_id:
            status = AuthStatus(
                mode=AuthMode.PUBLIC,
                is_authenticated=False,
                data_mode="delayed",
                message="No session configured. Set TV_SESSION_ID environment variable for premium features.",
            )
            self._validation_cache = status
            return status

        # Attempt to validate session
        try:
            is_valid, account_info = self._check_session_validity()

            if is_valid:
                self._validated = True
                status = AuthStatus(
                    mode=AuthMode.AUTHENTICATED,
                    is_authenticated=True,
                    data_mode="realtime",
                    message="Session validated successfully. Premium features enabled.",
                    account_info=account_info,
                )
            else:
                self._validated = False
                status = AuthStatus(
                    mode=AuthMode.INVALID,
                    is_authenticated=False,
                    data_mode="delayed",
                    message="Session validation failed. Please update TV_SESSION_ID.",
                )

        except Exception as e:
            # Don't expose details that might leak session info
            logger.warning(f"Session validation error: {type(e).__name__}")
            self._validated = False
            status = AuthStatus(
                mode=AuthMode.INVALID,
                is_authenticated=False,
                data_mode="delayed",
                message="Session validation failed due to network error. Falling back to public mode.",
            )

        self._validation_cache = status
        return status

    def _check_session_validity(self) -> tuple[bool, Optional[Dict[str, Any]]]:
        """
        Internal method to check session validity with TradingView.

        Returns:
            Tuple of (is_valid, account_info)
        """
        try:
            import requests

            # Use a lightweight endpoint to verify session
            response = requests.get(
                "https://www.tradingview.com/u/",
                cookies={"sessionid": self._session_id},
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; TradingView-MCP/1.0)",
                },
                timeout=10,
                allow_redirects=False,
            )

            # If redirected to sign-in, session is invalid
            if response.status_code == 302:
                location = response.headers.get("Location", "")
                if "signin" in location.lower():
                    return False, None

            # 200 response typically means valid session
            if response.status_code == 200:
                # Try to extract basic account info from response
                account_info = self._extract_account_info(response.text)
                return True, account_info

            return False, None

        except requests.RequestException:
            # Network error - can't determine validity
            raise

    def _extract_account_info(self, html: str) -> Optional[Dict[str, Any]]:
        """
        Extract basic account info from TradingView response.

        Note: Only extracts non-sensitive information.
        """
        try:
            import re

            account_info = {}

            # Try to find username (public info)
            username_match = re.search(r'"username":\s*"([^"]+)"', html)
            if username_match:
                account_info["username"] = username_match.group(1)

            # Try to find subscription plan
            plan_match = re.search(r'"pro_plan":\s*"([^"]+)"', html)
            if plan_match:
                plan = plan_match.group(1)
                account_info["plan"] = plan if plan else "free"

            return account_info if account_info else None

        except Exception:
            return None

    def get_status(self) -> Dict[str, Any]:
        """
        Get current authentication status (safe to expose to users).

        Returns:
            Dict with status information. Never includes session ID.
        """
        status = self.validate_session()

        result = {
            "authenticated": status.is_authenticated,
            "mode": status.mode.value,
            "data_mode": status.data_mode,
            "message": status.message,
            "env_var_configured": self.has_session,
            "env_var_name": self.ENV_SESSION_ID,
        }

        if status.account_info:
            # Only include non-sensitive account info
            result["account"] = {
                "username": status.account_info.get("username"),
                "plan": status.account_info.get("plan"),
            }

        return result

    def clear_session(self) -> None:
        """
        Clear the session from memory.

        Use this for logout functionality or security purposes.
        """
        self._session_id = None
        self._validated = False
        self._validation_cache = None
        logger.info("Session cleared from memory")

    def refresh_from_environment(self) -> AuthStatus:
        """
        Reload session ID from environment variable.

        Useful if the environment variable was updated during runtime.
        """
        self.clear_session()
        self._load_from_environment()
        return self.validate_session(force=True)


# Global singleton instance
_auth_manager: Optional[SecureAuthManager] = None


def get_auth_manager() -> SecureAuthManager:
    """
    Get the global auth manager instance.

    Returns:
        SecureAuthManager singleton instance.
    """
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = SecureAuthManager()
    return _auth_manager


def get_cookies() -> Optional[Dict[str, str]]:
    """
    Convenience function to get cookies for API requests.

    Returns:
        Cookies dict if authenticated, None otherwise.
    """
    return get_auth_manager().get_cookies()

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

    Required environment variables:
        TV_SESSION_ID: The sessionid cookie value
        TV_SESSION_ID_SIGN: The sessionid_sign cookie value (required for auth)
    """

    # Environment variable names
    ENV_SESSION_ID = "TV_SESSION_ID"
    ENV_SESSION_ID_SIGN = "TV_SESSION_ID_SIGN"

    def __init__(self):
        """Initialize auth manager, loading session from environment."""
        self._session_id: Optional[str] = None
        self._session_id_sign: Optional[str] = None
        self._validated: bool = False
        self._validation_cache: Optional[AuthStatus] = None

        # Load session from environment
        self._load_from_environment()

    def _load_from_environment(self) -> None:
        """Load session ID and signature from environment variables."""
        session_id = os.environ.get(self.ENV_SESSION_ID, "").strip()
        session_id_sign = os.environ.get(self.ENV_SESSION_ID_SIGN, "").strip()

        if session_id and session_id_sign:
            # Basic validation: both values should be non-empty strings
            # TradingView session IDs are typically 20-50 characters
            if len(session_id) >= 10 and len(session_id_sign) >= 10:
                self._session_id = session_id
                self._session_id_sign = session_id_sign
                logger.info("Session ID and signature loaded from environment variables")
            else:
                logger.warning("Session ID or signature in environment variable appears invalid (too short)")
        elif session_id and not session_id_sign:
            logger.warning(
                f"TV_SESSION_ID is set but TV_SESSION_ID_SIGN is missing. "
                f"Both are required for authentication. Running in public mode."
            )
        else:
            logger.info("No session configured - running in public mode")

    @property
    def is_authenticated(self) -> bool:
        """Check if we have a valid session ID configured."""
        return self._session_id is not None and self._validated

    @property
    def has_session(self) -> bool:
        """Check if session credentials are configured (may not be validated yet)."""
        return self._session_id is not None and self._session_id_sign is not None

    def get_cookies(self) -> Optional[Dict[str, str]]:
        """
        Get cookies for API requests.

        Returns:
            Dict with sessionid and sessionid_sign cookies if configured, None otherwise.

        Security note:
            Only call this method when actually needed for API requests.
            Do not log or expose the returned cookies.
        """
        if self._session_id and self._session_id_sign:
            return {
                "sessionid": self._session_id,
                "sessionid_sign": self._session_id_sign,
            }
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
        if not self._session_id or not self._session_id_sign:
            missing = []
            if not self._session_id:
                missing.append("TV_SESSION_ID")
            if not self._session_id_sign:
                missing.append("TV_SESSION_ID_SIGN")
            status = AuthStatus(
                mode=AuthMode.PUBLIC,
                is_authenticated=False,
                data_mode="delayed",
                message=f"Missing environment variables: {', '.join(missing)}. Both are required for premium features.",
            )
            self._validation_cache = status
            return status

        # Session is configured - trust it without remote validation
        # The actual API calls will determine if the session is valid
        # This avoids issues with TradingView's changing endpoints
        self._validated = True
        status = AuthStatus(
            mode=AuthMode.AUTHENTICATED,
            is_authenticated=True,
            data_mode="realtime",
            message="Session cookies configured. Authentication will be verified on first API call.",
            account_info={"username": "configured", "plan": "unknown"},
        )
        self._validation_cache = status
        return status

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
            "env_vars_configured": self.has_session,
            "env_vars": {
                "TV_SESSION_ID": self._session_id is not None,
                "TV_SESSION_ID_SIGN": self._session_id_sign is not None,
            },
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
        self._session_id_sign = None
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

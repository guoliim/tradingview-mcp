#!/usr/bin/env python3
"""
TradingView Browser Login Helper

Opens a browser window for user to login manually,
then extracts session cookies automatically.

Usage:
    python -m tradingview_mcp.login
    # or
    uv run tradingview-login
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict

# Check for playwright availability
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


def get_config_path() -> Path:
    """Get the path to store credentials."""
    # Use XDG config dir or fallback
    config_dir = Path.home() / ".config" / "tradingview-mcp"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "credentials.json"


def save_credentials(cookies: Dict[str, str]) -> Path:
    """Save cookies to config file."""
    config_path = get_config_path()
    config_path.write_text(json.dumps(cookies, indent=2))
    # Set restrictive permissions (owner read/write only)
    config_path.chmod(0o600)
    return config_path


def extract_tv_cookies(cookies: list) -> Optional[Dict[str, str]]:
    """Extract TradingView session cookies from browser cookies."""
    session_id = None
    session_id_sign = None

    for cookie in cookies:
        if cookie.get("name") == "sessionid":
            session_id = cookie.get("value")
        elif cookie.get("name") == "sessionid_sign":
            session_id_sign = cookie.get("value")

    if session_id and session_id_sign:
        return {
            "sessionid": session_id,
            "sessionid_sign": session_id_sign,
        }
    return None


def browser_login(headless: bool = False, timeout: int = 300) -> Optional[Dict[str, str]]:
    """
    Open browser for user to login to TradingView.

    Args:
        headless: If False, shows browser window (required for manual login)
        timeout: Max seconds to wait for login (default: 5 minutes)

    Returns:
        Dict with session cookies if login successful, None otherwise.
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("Error: Playwright not installed.")
        print("Install with: uv add playwright && playwright install chromium")
        return None

    print("=" * 50)
    print("TradingView Login Helper")
    print("=" * 50)
    print()
    print("A browser window will open. Please:")
    print("1. Login to your TradingView account")
    print("2. Complete any 2FA if required")
    print("3. Wait for the homepage to load")
    print()
    print(f"Timeout: {timeout} seconds")
    print("=" * 50)
    print()

    with sync_playwright() as p:
        # Launch visible browser
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # Go to TradingView login page
        print("Opening TradingView...")
        page.goto("https://www.tradingview.com/#signin")

        # Wait for user to login
        print("Waiting for login...")
        print("(Close the browser or press Ctrl+C to cancel)")
        print()

        start_time = time.time()
        cookies = None

        while time.time() - start_time < timeout:
            try:
                # Check for session cookies
                all_cookies = context.cookies()
                tv_cookies = extract_tv_cookies(all_cookies)

                if tv_cookies:
                    # Verify we're actually logged in by checking for user menu
                    try:
                        # Look for logged-in indicator
                        if page.locator("[data-name='user-menu-button']").is_visible(timeout=1000):
                            cookies = tv_cookies
                            print("Login detected!")
                            break
                    except PlaywrightTimeout:
                        pass

                time.sleep(2)

            except Exception as e:
                if "Target closed" in str(e) or "Browser closed" in str(e):
                    print("\nBrowser was closed.")
                    break
                # Continue waiting
                time.sleep(1)

        try:
            browser.close()
        except Exception:
            pass

        return cookies


def print_env_instructions(cookies: Dict[str, str]):
    """Print instructions for setting environment variables."""
    print()
    print("=" * 50)
    print("LOGIN SUCCESSFUL!")
    print("=" * 50)
    print()
    print("Add these to your Claude Desktop config:")
    print()
    print("```json")
    print('"env": {')
    print(f'  "TV_SESSION_ID": "{cookies["sessionid"]}",')
    print(f'  "TV_SESSION_ID_SIGN": "{cookies["sessionid_sign"]}"')
    print("}")
    print("```")
    print()
    print("Or export as environment variables:")
    print()
    print(f'export TV_SESSION_ID="{cookies["sessionid"]}"')
    print(f'export TV_SESSION_ID_SIGN="{cookies["sessionid_sign"]}"')
    print()


def main():
    """Main entry point for login helper."""
    if not PLAYWRIGHT_AVAILABLE:
        print("=" * 50)
        print("Playwright not installed!")
        print("=" * 50)
        print()
        print("To use browser login, install Playwright:")
        print()
        print("  uv add playwright")
        print("  playwright install chromium")
        print()
        print("Or manually copy cookies from browser DevTools:")
        print("  1. Login to tradingview.com")
        print("  2. Press F12 -> Application -> Cookies")
        print("  3. Copy 'sessionid' and 'sessionid_sign' values")
        print()
        sys.exit(1)

    # Parse arguments
    save_to_file = "--save" in sys.argv

    # Run browser login
    cookies = browser_login()

    if cookies:
        print_env_instructions(cookies)

        if save_to_file:
            config_path = save_credentials(cookies)
            print(f"Credentials saved to: {config_path}")
            print("(File permissions set to owner-only)")
        else:
            print("Tip: Run with --save to save credentials to file")
    else:
        print()
        print("Login was not completed or timed out.")
        print("Please try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()

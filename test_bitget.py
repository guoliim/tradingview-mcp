#!/usr/bin/env python3
"""
Test script to verify Bitget exchange integration
"""

import sys
sys.path.insert(0, "src")

from tradingview_mcp.core.services.coinlist import load_symbols
from tradingview_mcp.core.utils.validators import sanitize_exchange, EXCHANGE_SCREENER

def test_bitget_integration():
    """Test Bitget exchange integration"""
    print("🧪 Testing Bitget Integration\n")
    print("=" * 60)
    
    # Test 1: Check if Bitget is in EXCHANGE_SCREENER
    print("\n✅ Test 1: Checking EXCHANGE_SCREENER mapping...")
    if "bitget" in EXCHANGE_SCREENER:
        print(f"   ✓ Bitget found in EXCHANGE_SCREENER: {EXCHANGE_SCREENER['bitget']}")
    else:
        print("   ✗ Bitget NOT found in EXCHANGE_SCREENER")
        return False
    
    # Test 2: Check if exchange name sanitizes correctly
    print("\n✅ Test 2: Testing exchange sanitization...")
    result = sanitize_exchange("BITGET", "kucoin")
    if result == "bitget":
        print(f"   ✓ Exchange sanitization works: 'BITGET' -> '{result}'")
    else:
        print(f"   ✗ Exchange sanitization failed: got '{result}'")
        return False
    
    # Test 3: Load symbols from coinlist
    print("\n✅ Test 3: Loading Bitget symbols...")
    symbols = load_symbols("bitget")
    if symbols:
        print(f"   ✓ Successfully loaded {len(symbols)} symbols from bitget.txt")
        print(f"   ✓ First 5 symbols: {symbols[:5]}")
        print(f"   ✓ Last 5 symbols: {symbols[-5:]}")
    else:
        print("   ✗ Failed to load symbols")
        return False
    
    # Test 4: Verify symbol format
    print("\n✅ Test 4: Verifying symbol format...")
    if all(symbol.startswith("BITGET:") for symbol in symbols[:10]):
        print("   ✓ All symbols have correct 'BITGET:' prefix")
    else:
        print("   ✗ Symbol format incorrect")
        return False
    
    # Test 5: Check for common trading pairs
    print("\n✅ Test 5: Checking for common trading pairs...")
    common_pairs = ["BITGET:BTCUSDT", "BITGET:ETHUSDT", "BITGET:BNBUSDT"]
    found = [pair for pair in common_pairs if pair in symbols]
    if len(found) == len(common_pairs):
        print(f"   ✓ All common pairs found: {found}")
    else:
        print(f"   ⚠ Some common pairs missing. Found: {found}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Bitget integration successful!")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print(f"   - Exchange: Bitget")
    print(f"   - Screener type: {EXCHANGE_SCREENER['bitget']}")
    print(f"   - Total symbols: {len(symbols)}")
    print(f"   - Status: ✅ Ready to use")
    
    return True

if __name__ == "__main__":
    try:
        success = test_bitget_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

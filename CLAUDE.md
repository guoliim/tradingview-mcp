# CLAUDE.md

## Project Overview

TradingView MCP Server - A Model Context Protocol (MCP) server providing advanced cryptocurrency and stock market analysis using TradingView data. Built for traders, analysts, and AI assistants who need real-time market intelligence.

## Tech Stack

- **Language**: Python 3.10+
- **MCP Framework**: FastMCP (`mcp[cli]>=1.12.0`)
- **Data Sources**:
  - `tradingview-screener>=0.6.4` - Market screening
  - `tradingview-ta>=3.3.0` - Technical analysis
- **Package Manager**: UV (recommended)
- **Build System**: setuptools

## Project Structure

```
tradingview-mcp/
├── src/tradingview_mcp/
│   ├── __init__.py
│   ├── server.py              # Main MCP server with all tools
│   ├── coinlist/              # Exchange symbol lists (.txt files)
│   └── core/
│       ├── services/
│       │   ├── coinlist.py    # Symbol loading from coinlist files
│       │   ├── indicators.py  # Technical indicator calculations
│       │   └── screener_provider.py
│       └── utils/
│           └── validators.py  # Input sanitization, exchange/timeframe validation
├── pyproject.toml             # Python project configuration
├── package.json               # MCP metadata
└── test_api.py               # API testing
```

## Key Components

### server.py (Main Entry Point)
- Defines all MCP tools using `@mcp.tool()` decorator
- Implements market screening logic with TradingView APIs
- Handles multi-timeframe analysis

### validators.py
- `sanitize_exchange()` - Validates exchange names
- `sanitize_timeframe()` - Validates timeframe strings
- `EXCHANGE_SCREENER` - Maps exchanges to screener types (crypto/america/turkey)
- `ALLOWED_TIMEFRAMES` - Valid timeframes: 5m, 15m, 1h, 4h, 1D, 1W, 1M

### indicators.py
- `compute_metrics()` - Calculates Bollinger Band metrics, ratings, and signals

## Available MCP Tools

| Tool | Purpose |
|------|---------|
| `top_gainers` | Find highest performing assets |
| `top_losers` | Find biggest declining assets |
| `bollinger_scan` | Find assets with tight Bollinger Bands (squeeze detection) |
| `rating_filter` | Filter by Bollinger Band rating (-3 to +3) |
| `coin_analysis` | Complete technical analysis for specific symbol |
| `consecutive_candles_scan` | Detect candlestick patterns |
| `advanced_candle_pattern` | Multi-timeframe pattern analysis |
| `volume_breakout_scanner` | Detect volume + price breakouts |
| `volume_confirmation_analysis` | Detailed volume analysis for symbol |
| `smart_volume_scanner` | Combined volume + technical scanner |

## Supported Exchanges

**Crypto**: KUCOIN, BINANCE, BYBIT, BITGET, OKX, COINBASE, GATEIO, HUOBI, BITFINEX
**Stocks**: NASDAQ, NYSE, BIST (Turkish), KLSE, HKEX

## Development Commands

```bash
# Install dependencies
uv sync

# Run server (stdio mode)
uv run tradingview-mcp

# Run with MCP Inspector for debugging
uv run mcp dev src/tradingview_mcp/server.py

# Test API functions
uv run python test_api.py

# Build package
python -m build
```

## Adding New Exchanges

1. Create a symbol list file in `src/tradingview_mcp/coinlist/{EXCHANGE}.txt`
2. Add exchange to `EXCHANGE_SCREENER` mapping in `validators.py`
3. One symbol per line in format: `EXCHANGE:SYMBOLUSDT`

## Bollinger Band Rating System

| Rating | Signal | Meaning |
|--------|--------|---------|
| +3 | Strong Buy | Price above upper BB |
| +2 | Buy | Price in upper 50% |
| +1 | Weak Buy | Price above middle |
| 0 | Neutral | Price at middle |
| -1 | Weak Sell | Price below middle |
| -2 | Sell | Price in lower 50% |
| -3 | Strong Sell | Price below lower BB |

## Common Issues

- **Rate limiting**: TradingView may throttle requests - wait 5-10 min between sessions
- **Empty results**: Try different exchange (KuCoin works best) or check symbol format
- **Missing data**: Ensure symbol format is `EXCHANGE:SYMBOLUSDT`

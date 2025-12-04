"""
Support and Resistance Level Calculator

Calculates support and resistance levels using multiple technical analysis methods:
- Pivot Points (Classic, Fibonacci, Woodie, Camarilla)
- Historical High/Low (Swing Points)
- Fibonacci Retracement
- Key Moving Averages (dynamic S/R)

Supports multiple lookback periods: 1Y, 6M, 3M, 1M, 1W, 1D
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PivotType(Enum):
    """Pivot point calculation methods."""
    CLASSIC = "classic"
    FIBONACCI = "fibonacci"
    WOODIE = "woodie"
    CAMARILLA = "camarilla"
    DEMARK = "demark"


@dataclass
class PriceData:
    """OHLC price data for a period."""
    high: float
    low: float
    close: float
    open: float = 0.0


@dataclass
class SupportResistanceLevels:
    """Container for support and resistance levels."""
    pivot: float
    supports: List[float]  # S1, S2, S3...
    resistances: List[float]  # R1, R2, R3...
    method: str


class SupportResistanceCalculator:
    """
    Calculator for support and resistance levels.

    Uses multiple technical analysis methods to identify key price levels.
    """

    # Lookback period mappings (in trading days approximately)
    PERIOD_DAYS = {
        "1D": 1,
        "1W": 5,
        "1M": 22,
        "3M": 66,
        "6M": 132,
        "1Y": 252,
    }

    # Fibonacci ratios for retracement
    FIB_RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]

    def __init__(self):
        """Initialize calculator."""
        pass

    def calculate_pivot_points(
        self,
        price_data: PriceData,
        pivot_type: PivotType = PivotType.CLASSIC
    ) -> SupportResistanceLevels:
        """
        Calculate pivot points using specified method.

        Args:
            price_data: OHLC data for the period
            pivot_type: Type of pivot calculation

        Returns:
            SupportResistanceLevels with pivot, supports, and resistances
        """
        high, low, close = price_data.high, price_data.low, price_data.close
        open_price = price_data.open

        if pivot_type == PivotType.CLASSIC:
            return self._classic_pivots(high, low, close)
        elif pivot_type == PivotType.FIBONACCI:
            return self._fibonacci_pivots(high, low, close)
        elif pivot_type == PivotType.WOODIE:
            return self._woodie_pivots(high, low, close, open_price)
        elif pivot_type == PivotType.CAMARILLA:
            return self._camarilla_pivots(high, low, close)
        elif pivot_type == PivotType.DEMARK:
            return self._demark_pivots(high, low, close, open_price)
        else:
            return self._classic_pivots(high, low, close)

    def _classic_pivots(self, high: float, low: float, close: float) -> SupportResistanceLevels:
        """Classic (Floor) Pivot Points."""
        pivot = (high + low + close) / 3

        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        return SupportResistanceLevels(
            pivot=round(pivot, 4),
            supports=[round(s1, 4), round(s2, 4), round(s3, 4)],
            resistances=[round(r1, 4), round(r2, 4), round(r3, 4)],
            method="classic"
        )

    def _fibonacci_pivots(self, high: float, low: float, close: float) -> SupportResistanceLevels:
        """Fibonacci Pivot Points."""
        pivot = (high + low + close) / 3
        range_hl = high - low

        s1 = pivot - (0.382 * range_hl)
        s2 = pivot - (0.618 * range_hl)
        s3 = pivot - (1.0 * range_hl)

        r1 = pivot + (0.382 * range_hl)
        r2 = pivot + (0.618 * range_hl)
        r3 = pivot + (1.0 * range_hl)

        return SupportResistanceLevels(
            pivot=round(pivot, 4),
            supports=[round(s1, 4), round(s2, 4), round(s3, 4)],
            resistances=[round(r1, 4), round(r2, 4), round(r3, 4)],
            method="fibonacci"
        )

    def _woodie_pivots(self, high: float, low: float, close: float, open_price: float) -> SupportResistanceLevels:
        """Woodie Pivot Points (weights current open)."""
        pivot = (high + low + 2 * open_price) / 4

        s1 = (2 * pivot) - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)

        r1 = (2 * pivot) - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)

        return SupportResistanceLevels(
            pivot=round(pivot, 4),
            supports=[round(s1, 4), round(s2, 4), round(s3, 4)],
            resistances=[round(r1, 4), round(r2, 4), round(r3, 4)],
            method="woodie"
        )

    def _camarilla_pivots(self, high: float, low: float, close: float) -> SupportResistanceLevels:
        """Camarilla Pivot Points (tighter levels, good for intraday)."""
        pivot = (high + low + close) / 3
        range_hl = high - low

        s1 = close - (range_hl * 1.1 / 12)
        s2 = close - (range_hl * 1.1 / 6)
        s3 = close - (range_hl * 1.1 / 4)
        s4 = close - (range_hl * 1.1 / 2)

        r1 = close + (range_hl * 1.1 / 12)
        r2 = close + (range_hl * 1.1 / 6)
        r3 = close + (range_hl * 1.1 / 4)
        r4 = close + (range_hl * 1.1 / 2)

        return SupportResistanceLevels(
            pivot=round(pivot, 4),
            supports=[round(s1, 4), round(s2, 4), round(s3, 4), round(s4, 4)],
            resistances=[round(r1, 4), round(r2, 4), round(r3, 4), round(r4, 4)],
            method="camarilla"
        )

    def _demark_pivots(self, high: float, low: float, close: float, open_price: float) -> SupportResistanceLevels:
        """DeMark Pivot Points."""
        if close < open_price:
            x = high + 2 * low + close
        elif close > open_price:
            x = 2 * high + low + close
        else:
            x = high + low + 2 * close

        pivot = x / 4
        s1 = x / 2 - high
        r1 = x / 2 - low

        return SupportResistanceLevels(
            pivot=round(pivot, 4),
            supports=[round(s1, 4)],
            resistances=[round(r1, 4)],
            method="demark"
        )

    def calculate_fibonacci_retracement(
        self,
        high: float,
        low: float,
        trend: str = "up"
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.

        Args:
            high: Period high
            low: Period low
            trend: 'up' for uptrend (retracements from high),
                   'down' for downtrend (retracements from low)

        Returns:
            Dict with Fibonacci levels
        """
        range_hl = high - low

        if trend == "up":
            # Uptrend: measure retracements from high down
            levels = {
                "0%": round(high, 4),
                "23.6%": round(high - range_hl * 0.236, 4),
                "38.2%": round(high - range_hl * 0.382, 4),
                "50%": round(high - range_hl * 0.5, 4),
                "61.8%": round(high - range_hl * 0.618, 4),
                "78.6%": round(high - range_hl * 0.786, 4),
                "100%": round(low, 4),
            }
        else:
            # Downtrend: measure retracements from low up
            levels = {
                "0%": round(low, 4),
                "23.6%": round(low + range_hl * 0.236, 4),
                "38.2%": round(low + range_hl * 0.382, 4),
                "50%": round(low + range_hl * 0.5, 4),
                "61.8%": round(low + range_hl * 0.618, 4),
                "78.6%": round(low + range_hl * 0.786, 4),
                "100%": round(high, 4),
            }

        return levels

    def identify_key_levels(
        self,
        current_price: float,
        pivot_levels: SupportResistanceLevels,
        fib_levels: Dict[str, float],
        moving_averages: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Identify and consolidate key support/resistance levels.

        Combines pivot points, Fibonacci, and MAs to find confluence zones.

        Returns:
            Dict with nearest support/resistance and all levels sorted
        """
        all_supports = []
        all_resistances = []

        # Add pivot levels
        for i, s in enumerate(pivot_levels.supports):
            all_supports.append({"level": s, "source": f"Pivot S{i+1}", "strength": 3 - i})
        for i, r in enumerate(pivot_levels.resistances):
            all_resistances.append({"level": r, "source": f"Pivot R{i+1}", "strength": 3 - i})

        # Add Fibonacci levels relative to current price
        for name, level in fib_levels.items():
            if level < current_price:
                all_supports.append({"level": level, "source": f"Fib {name}", "strength": 2 if "61.8" in name or "38.2" in name else 1})
            elif level > current_price:
                all_resistances.append({"level": level, "source": f"Fib {name}", "strength": 2 if "61.8" in name or "38.2" in name else 1})

        # Add moving averages
        for ma_name, ma_value in moving_averages.items():
            if ma_value is not None:
                entry = {"level": ma_value, "source": ma_name, "strength": 2 if "200" in ma_name else 1}
                if ma_value < current_price:
                    all_supports.append(entry)
                else:
                    all_resistances.append(entry)

        # Sort by distance from current price
        all_supports.sort(key=lambda x: current_price - x["level"])
        all_resistances.sort(key=lambda x: x["level"] - current_price)

        # Find nearest levels
        nearest_support = all_supports[0] if all_supports else None
        nearest_resistance = all_resistances[0] if all_resistances else None

        return {
            "current_price": current_price,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "all_supports": all_supports[:5],  # Top 5 nearest
            "all_resistances": all_resistances[:5],
        }


def calculate_support_resistance_for_symbol(
    symbol: str,
    exchange: str = "NASDAQ",
    periods: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Calculate support and resistance levels for a symbol across multiple periods.

    Args:
        symbol: Stock/crypto symbol (e.g., "AAPL", "BTCUSDT")
        exchange: Exchange name (NASDAQ, NYSE, HKEX, etc.)
        periods: List of periods to analyze (default: ["1D", "1W", "1M", "3M", "6M", "1Y"])

    Returns:
        Dict with support/resistance levels for each period
    """
    if periods is None:
        periods = ["1D", "1W", "1M", "3M"]

    try:
        from tradingview_ta import TA_Handler
    except ImportError:
        return {"error": "tradingview_ta not installed"}

    # Map exchange to screener
    screener_map = {
        "NASDAQ": "america",
        "NYSE": "america",
        "HKEX": "hongkong",
        "BIST": "turkey",
        "BINANCE": "crypto",
        "KUCOIN": "crypto",
        "BYBIT": "crypto",
    }

    screener = screener_map.get(exchange.upper(), "america")
    calculator = SupportResistanceCalculator()

    # Timeframe mapping for tradingview_ta
    tf_map = {
        "1D": "1d",
        "1W": "1W",
        "1M": "1M",
    }

    results = {
        "symbol": symbol,
        "exchange": exchange,
        "periods": {}
    }

    for period in periods:
        try:
            # Use daily timeframe for all calculations
            # The period determines the lookback for high/low
            handler = TA_Handler(
                symbol=symbol,
                screener=screener,
                exchange=exchange,
                interval="1d"  # Daily candles
            )

            analysis = handler.get_analysis()
            indicators = analysis.indicators

            # Extract price data
            current_close = indicators.get("close", 0)

            # For different periods, we use different high/low calculations
            # TradingView provides some historical data
            high_key = f"High.{period}" if period != "1D" else "high"
            low_key = f"Low.{period}" if period != "1D" else "low"

            # Get high/low (fallback to daily if period-specific not available)
            period_high = indicators.get(high_key) or indicators.get("high") or indicators.get("High.All") or current_close * 1.1
            period_low = indicators.get(low_key) or indicators.get("low") or indicators.get("Low.All") or current_close * 0.9

            # For multi-period, try to get extended data
            if period in ["1M", "3M", "6M", "1Y"]:
                # Use 52-week high/low as approximation for longer periods
                high_52w = indicators.get("price_52_week_high") or indicators.get("High.All")
                low_52w = indicators.get("price_52_week_low") or indicators.get("Low.All")

                if period == "1Y" and high_52w and low_52w:
                    period_high = high_52w
                    period_low = low_52w
                elif period in ["3M", "6M"]:
                    # Approximate by interpolating between current and 52w
                    factor = 0.5 if period == "6M" else 0.25
                    if high_52w and low_52w:
                        period_high = current_close + (high_52w - current_close) * factor
                        period_low = current_close - (current_close - low_52w) * factor

            # Create price data
            price_data = PriceData(
                high=period_high,
                low=period_low,
                close=current_close,
                open=indicators.get("open", current_close)
            )

            # Calculate pivot points (multiple methods)
            classic_pivots = calculator.calculate_pivot_points(price_data, PivotType.CLASSIC)
            fib_pivots = calculator.calculate_pivot_points(price_data, PivotType.FIBONACCI)

            # Calculate Fibonacci retracement
            # Determine trend based on current price position
            mid_point = (period_high + period_low) / 2
            trend = "up" if current_close > mid_point else "down"
            fib_levels = calculator.calculate_fibonacci_retracement(period_high, period_low, trend)

            # Get moving averages as dynamic S/R
            moving_averages = {
                "SMA20": indicators.get("SMA20"),
                "EMA50": indicators.get("EMA50"),
                "SMA100": indicators.get("SMA100"),
                "SMA200": indicators.get("SMA200"),
            }

            # Identify key levels
            key_levels = calculator.identify_key_levels(
                current_close,
                classic_pivots,
                fib_levels,
                moving_averages
            )

            results["periods"][period] = {
                "price_range": {
                    "high": round(period_high, 4),
                    "low": round(period_low, 4),
                    "current": round(current_close, 4),
                },
                "pivot_points": {
                    "classic": {
                        "pivot": classic_pivots.pivot,
                        "supports": classic_pivots.supports,
                        "resistances": classic_pivots.resistances,
                    },
                    "fibonacci": {
                        "pivot": fib_pivots.pivot,
                        "supports": fib_pivots.supports,
                        "resistances": fib_pivots.resistances,
                    },
                },
                "fibonacci_retracement": fib_levels,
                "moving_averages": {k: round(v, 4) if v else None for k, v in moving_averages.items()},
                "key_levels": key_levels,
                "trend": trend,
            }

        except Exception as e:
            logger.warning(f"Error calculating S/R for {symbol} period {period}: {e}")
            results["periods"][period] = {"error": str(e)}

    return results


def batch_support_resistance(
    symbols: List[str],
    exchange: str = "NASDAQ",
    periods: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """
    Calculate support/resistance for multiple symbols.

    Args:
        symbols: List of symbols
        exchange: Exchange name
        periods: Periods to analyze

    Returns:
        List of results for each symbol
    """
    results = []
    for symbol in symbols[:10]:  # Limit to 10 symbols
        result = calculate_support_resistance_for_symbol(symbol, exchange, periods)
        results.append(result)
    return results

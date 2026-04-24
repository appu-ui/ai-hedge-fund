import datetime
import pandas as pd
import yfinance as yf

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
)
from src.tools import yfinance_provider


def get_prices(
    ticker: str, start_date: str, end_date: str, api_key: str | None = None
) -> list[Price]:
    """Fetch price data using yfinance."""
    return yfinance_provider.get_prices(ticker, start_date, end_date)


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str | None = None,
) -> list[FinancialMetrics]:
    """Fetch financial metrics using yfinance."""
    return yfinance_provider.get_financial_metrics(ticker, end_date, period=period, limit=limit)


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str | None = None,
) -> list[LineItem]:
    """Fetch line items using yfinance."""
    return yfinance_provider.search_line_items(
        ticker, line_items, end_date, period=period, limit=limit
    )


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
) -> list[InsiderTrade]:
    """Fetch insider trades using yfinance."""
    return yfinance_provider.get_insider_trades(
        ticker, end_date, start_date=start_date, limit=limit
    )


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str | None = None,
) -> list[CompanyNews]:
    """Fetch company news using yfinance."""
    return yfinance_provider.get_company_news(
        ticker, end_date, start_date=start_date, limit=limit
    )


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str | None = None,
) -> float | None:
    """Fetch market cap using yfinance."""
    return yfinance_provider.get_market_cap(ticker, end_date)


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Convert prices to a DataFrame using yfinance data."""
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Yahoo OHLCV names and return a DataFrame."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    normalized = df.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in normalized.columns:
            normalized[col] = pd.NA
    return normalized[required]


def get_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch single-ticker historical OHLCV using yfinance."""
    try:
        df = yf.Ticker(ticker).history(period=period)
    except Exception as exc:
        raise ValueError(f"Failed to fetch stock data for '{ticker}': {exc}") from exc
    return _normalize_ohlcv_columns(df)


def fetch_market_data(tickers: list[str], period: str = "1y") -> pd.DataFrame:
    """Fetch multi-ticker market data using yfinance download."""
    if not tickers:
        return pd.DataFrame()
    try:
        return yf.download(tickers, period=period, progress=False, threads=False)
    except Exception as exc:
        raise ValueError(f"Failed to fetch market data for tickers {tickers}: {exc}") from exc


def load_prices(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Compatibility helper for loading normalized OHLCV prices."""
    return get_stock_data(ticker, period=period)


def get_latest_price(ticker: str) -> float:
    return yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]

import datetime
import pandas as pd

from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
)
from src.tools import yfinance_provider


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data using yfinance."""
    return yfinance_provider.get_prices(ticker, start_date, end_date)


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics using yfinance."""
    return yfinance_provider.get_financial_metrics(ticker, end_date, period=period, limit=limit)


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
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
) -> list[CompanyNews]:
    """Fetch company news using yfinance."""
    return yfinance_provider.get_company_news(
        ticker, end_date, start_date=start_date, limit=limit
    )


def get_market_cap(
    ticker: str,
    end_date: str,
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

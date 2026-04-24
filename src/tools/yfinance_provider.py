"""yfinance-backed data provider.

Provides drop-in replacements for the financialdatasets.ai fetchers in
``src/tools/api.py`` so users without an API key can still run the agents
on US-listed equities.

Caveats:
- Data comes from Yahoo Finance public endpoints and is typically delayed
  ~15 minutes for US equities. Suitable for end-of-day analysis only.
- yfinance is an unofficial scraper and can break when Yahoo changes its
  endpoints. Pin a known-good version in pyproject.toml.
- Insider trades and "line items" coverage is best-effort; not every field
  the agents request exists in Yahoo's data, missing values are returned
  as ``None``.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import pandas as pd
import yfinance as yf

from src.data.cache import get_cache
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    InsiderTrade,
    LineItem,
    Price,
)

_cache = get_cache()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _safe_int(value: Any) -> int | None:
    f = _safe_float(value)
    if f is None:
        return None
    return int(f)


def _row(df: pd.DataFrame | None, *names: str) -> pd.Series | None:
    """Return the first matching row from a yfinance statement DataFrame."""
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            return df.loc[name]
    # case-insensitive fallback
    lower = {str(idx).lower(): idx for idx in df.index}
    for name in names:
        key = name.lower()
        if key in lower:
            return df.loc[lower[key]]
    return None


def _value_at(series: pd.Series | None, column) -> float | None:
    if series is None or column not in series.index:
        return None
    return _safe_float(series[column])


def _date_str(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, _dt.datetime, _dt.date)):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


# ---------------------------------------------------------------------------
# prices
# ---------------------------------------------------------------------------

def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    cache_key = f"{ticker}_{start_date}_{end_date}"
    if cached := _cache.get_prices(cache_key):
        return [Price(**p) for p in cached]

    # yfinance end is exclusive; bump by 1 day so end_date is included
    try:
        end_dt = _dt.datetime.strptime(end_date, "%Y-%m-%d") + _dt.timedelta(days=1)
    except ValueError:
        end_dt = _dt.datetime.strptime(end_date.split("T")[0], "%Y-%m-%d") + _dt.timedelta(days=1)

    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
            threads=False,
        )
    except Exception:
        return []

    if df is None or df.empty:
        return []

    # Flatten potential MultiIndex columns (single-ticker downloads)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    prices: list[Price] = []
    for ts, row in df.iterrows():
        try:
            prices.append(
                Price(
                    open=float(row["Open"]),
                    close=float(row["Close"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    volume=int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
                    time=_date_str(ts),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue

    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


# ---------------------------------------------------------------------------
# financial metrics
# ---------------------------------------------------------------------------

def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    cache_key = f"{ticker}_{period}_{end_date}_{limit}_yf"
    if cached := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**m) for m in cached]

    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception:
        info = {}

    use_quarterly = period in ("quarterly", "quarter", "q", "ttm")

    income = t.quarterly_income_stmt if use_quarterly else t.income_stmt
    balance = t.quarterly_balance_sheet if use_quarterly else t.balance_sheet
    cashflow = t.quarterly_cashflow if use_quarterly else t.cashflow

    # rows we'll need
    revenue_row = _row(income, "Total Revenue", "Revenue")
    cogs_row = _row(income, "Cost Of Revenue", "Cost Of Goods Sold")
    gross_profit_row = _row(income, "Gross Profit")
    op_income_row = _row(income, "Operating Income", "Total Operating Income As Reported")
    net_income_row = _row(income, "Net Income", "Net Income Common Stockholders")
    interest_exp_row = _row(income, "Interest Expense", "Interest Expense Non Operating")
    ebit_row = _row(income, "EBIT", "Normalized EBIT")
    ebitda_row = _row(income, "EBITDA", "Normalized EBITDA")
    eps_row = _row(income, "Diluted EPS", "Basic EPS")

    total_assets_row = _row(balance, "Total Assets")
    total_liab_row = _row(
        balance, "Total Liabilities Net Minority Interest", "Total Liabilities"
    )
    equity_row = _row(
        balance, "Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"
    )
    current_assets_row = _row(balance, "Current Assets", "Total Current Assets")
    current_liab_row = _row(balance, "Current Liabilities", "Total Current Liabilities")
    cash_row = _row(
        balance, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"
    )
    inventory_row = _row(balance, "Inventory")
    receivables_row = _row(balance, "Accounts Receivable", "Receivables")
    total_debt_row = _row(balance, "Total Debt")
    shares_row = _row(
        balance, "Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"
    )

    fcf_row = _row(cashflow, "Free Cash Flow")
    op_cf_row = _row(cashflow, "Operating Cash Flow", "Cash Flow From Continuing Operating Activities")

    # gather sorted period columns (most recent first)
    periods: list[pd.Timestamp] = []
    for src in (income, balance, cashflow):
        if src is not None and not src.empty:
            periods.extend(list(src.columns))
    periods = sorted({p for p in periods}, reverse=True)
    # Filter to <= end_date
    try:
        end_ts = pd.Timestamp(end_date)
        periods = [p for p in periods if pd.Timestamp(p) <= end_ts]
    except Exception:
        pass
    periods = periods[:limit]

    metrics: list[FinancialMetrics] = []
    market_cap = _safe_float(info.get("marketCap"))
    enterprise_value = _safe_float(info.get("enterpriseValue"))
    currency = info.get("financialCurrency") or info.get("currency") or "USD"

    for col in periods:
        revenue = _value_at(revenue_row, col)
        gross_profit = _value_at(gross_profit_row, col)
        if gross_profit is None and revenue is not None:
            cogs = _value_at(cogs_row, col)
            if cogs is not None:
                gross_profit = revenue - cogs

        op_income = _value_at(op_income_row, col)
        net_income = _value_at(net_income_row, col)
        ebitda = _value_at(ebitda_row, col)
        ebit = _value_at(ebit_row, col) or op_income

        total_assets = _value_at(total_assets_row, col)
        total_liab = _value_at(total_liab_row, col)
        equity = _value_at(equity_row, col)
        current_assets = _value_at(current_assets_row, col)
        current_liab = _value_at(current_liab_row, col)
        cash = _value_at(cash_row, col)
        inventory = _value_at(inventory_row, col)
        receivables = _value_at(receivables_row, col)
        total_debt = _value_at(total_debt_row, col)
        fcf = _value_at(fcf_row, col)
        op_cf = _value_at(op_cf_row, col)
        eps = _value_at(eps_row, col)
        shares = _value_at(shares_row, col)

        def _ratio(n, d):
            if n is None or d in (None, 0) or d == 0.0:
                return None
            try:
                return n / d
            except ZeroDivisionError:
                return None

        gross_margin = _ratio(gross_profit, revenue)
        operating_margin = _ratio(op_income, revenue)
        net_margin = _ratio(net_income, revenue)
        roe = _ratio(net_income, equity)
        roa = _ratio(net_income, total_assets)
        invested_capital = None
        if equity is not None and total_debt is not None:
            invested_capital = equity + total_debt
        roic = _ratio(net_income, invested_capital) if invested_capital else None
        asset_turnover = _ratio(revenue, total_assets)
        inv_turnover = _ratio(revenue, inventory)
        rec_turnover = _ratio(revenue, receivables)
        dso = _ratio(receivables, revenue)
        if dso is not None:
            dso = dso * 365
        current_ratio = _ratio(current_assets, current_liab)
        quick_ratio = None
        if current_assets is not None and inventory is not None and current_liab:
            quick_ratio = _ratio(current_assets - inventory, current_liab)
        cash_ratio = _ratio(cash, current_liab)
        op_cf_ratio = _ratio(op_cf, current_liab)
        debt_to_equity = _ratio(total_debt, equity)
        debt_to_assets = _ratio(total_debt, total_assets)
        interest_coverage = _ratio(ebit, _value_at(interest_exp_row, col))
        book_value_per_share = _ratio(equity, shares)
        fcf_per_share = _ratio(fcf, shares)
        fcf_yield = _ratio(fcf, market_cap) if market_cap else None
        ev_ebitda = _ratio(enterprise_value, ebitda) if enterprise_value else None
        ev_revenue = _ratio(enterprise_value, revenue) if enterprise_value else None

        # P/E, P/B, P/S — only meaningful for the most recent period
        p_e = p_b = p_s = peg = payout = None
        if col == periods[0]:
            p_e = _safe_float(info.get("trailingPE"))
            p_b = _safe_float(info.get("priceToBook"))
            p_s = _safe_float(info.get("priceToSalesTrailing12Months"))
            peg = _safe_float(info.get("pegRatio"))
            payout = _safe_float(info.get("payoutRatio"))

        metrics.append(
            FinancialMetrics(
                ticker=ticker,
                report_period=_date_str(col),
                period=period,
                currency=currency,
                market_cap=market_cap if col == periods[0] else None,
                enterprise_value=enterprise_value if col == periods[0] else None,
                price_to_earnings_ratio=p_e,
                price_to_book_ratio=p_b,
                price_to_sales_ratio=p_s,
                enterprise_value_to_ebitda_ratio=ev_ebitda,
                enterprise_value_to_revenue_ratio=ev_revenue,
                free_cash_flow_yield=fcf_yield,
                peg_ratio=peg,
                gross_margin=gross_margin,
                operating_margin=operating_margin,
                net_margin=net_margin,
                return_on_equity=roe,
                return_on_assets=roa,
                return_on_invested_capital=roic,
                asset_turnover=asset_turnover,
                inventory_turnover=inv_turnover,
                receivables_turnover=rec_turnover,
                days_sales_outstanding=dso,
                operating_cycle=None,
                working_capital_turnover=None,
                current_ratio=current_ratio,
                quick_ratio=quick_ratio,
                cash_ratio=cash_ratio,
                operating_cash_flow_ratio=op_cf_ratio,
                debt_to_equity=debt_to_equity,
                debt_to_assets=debt_to_assets,
                interest_coverage=interest_coverage,
                revenue_growth=None,
                earnings_growth=None,
                book_value_growth=None,
                earnings_per_share_growth=None,
                free_cash_flow_growth=None,
                operating_income_growth=None,
                ebitda_growth=None,
                payout_ratio=payout,
                earnings_per_share=eps,
                book_value_per_share=book_value_per_share,
                free_cash_flow_per_share=fcf_per_share,
            )
        )

    # Compute YoY growth rates where possible (oldest -> newest scan)
    for i in range(len(metrics) - 1):
        cur = metrics[i]
        prev = metrics[i + 1]

        def _g(curv, prevv):
            if curv is None or prevv in (None, 0) or prevv == 0.0:
                return None
            try:
                return (curv - prevv) / abs(prevv)
            except ZeroDivisionError:
                return None

        # We have to mutate via dict because pydantic models are immutable-ish
        updates = {
            "revenue_growth": _g(
                _ratio(cur.net_margin, 1) if False else None, None
            ),  # placeholder, handled below
        }
        # actually compute from raw via re-extraction would be ideal; use what we have
        cur_dict = cur.model_dump()
        prev_dict = prev.model_dump()
        cur_dict["earnings_per_share_growth"] = _g(cur.earnings_per_share, prev.earnings_per_share)
        cur_dict["book_value_growth"] = _g(cur.book_value_per_share, prev.book_value_per_share)
        cur_dict["free_cash_flow_growth"] = _g(
            cur.free_cash_flow_per_share, prev.free_cash_flow_per_share
        )
        metrics[i] = FinancialMetrics(**cur_dict)

    if metrics:
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
    return metrics


# ---------------------------------------------------------------------------
# line items
# ---------------------------------------------------------------------------

# Map agent line-item names → (statement, list of yfinance row labels, sign)
# statement: "income" | "balance" | "cashflow" | "info"
_LINE_ITEM_MAP: dict[str, tuple[str, tuple[str, ...], int]] = {
    "revenue": ("income", ("Total Revenue", "Revenue"), 1),
    "gross_profit": ("income", ("Gross Profit",), 1),
    "operating_income": ("income", ("Operating Income", "Total Operating Income As Reported"), 1),
    "operating_expense": ("income", ("Operating Expense", "Total Operating Expenses"), 1),
    "net_income": ("income", ("Net Income", "Net Income Common Stockholders"), 1),
    "interest_expense": ("income", ("Interest Expense", "Interest Expense Non Operating"), 1),
    "ebit": ("income", ("EBIT", "Normalized EBIT"), 1),
    "ebitda": ("income", ("EBITDA", "Normalized EBITDA"), 1),
    "earnings_per_share": ("income", ("Diluted EPS", "Basic EPS"), 1),
    "research_and_development": ("income", ("Research And Development",), 1),
    "depreciation_and_amortization": (
        "cashflow",
        ("Depreciation Amortization Depletion", "Depreciation And Amortization"),
        1,
    ),
    "free_cash_flow": ("cashflow", ("Free Cash Flow",), 1),
    "capital_expenditure": ("cashflow", ("Capital Expenditure",), 1),
    "dividends_and_other_cash_distributions": (
        "cashflow",
        ("Cash Dividends Paid", "Common Stock Dividend Paid"),
        1,
    ),
    "issuance_or_purchase_of_equity_shares": (
        "cashflow",
        ("Net Common Stock Issuance", "Issuance Of Capital Stock", "Repurchase Of Capital Stock"),
        1,
    ),
    "total_assets": ("balance", ("Total Assets",), 1),
    "total_liabilities": (
        "balance",
        ("Total Liabilities Net Minority Interest", "Total Liabilities"),
        1,
    ),
    "current_assets": ("balance", ("Current Assets", "Total Current Assets"), 1),
    "current_liabilities": ("balance", ("Current Liabilities", "Total Current Liabilities"), 1),
    "cash_and_equivalents": (
        "balance",
        ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
        1,
    ),
    "shareholders_equity": (
        "balance",
        ("Stockholders Equity", "Common Stock Equity"),
        1,
    ),
    "total_debt": ("balance", ("Total Debt",), 1),
    "goodwill_and_intangible_assets": (
        "balance",
        ("Goodwill And Other Intangible Assets",),
        1,
    ),
    "intangible_assets": ("balance", ("Other Intangible Assets", "Goodwill And Other Intangible Assets"), 1),
    "outstanding_shares": (
        "balance",
        ("Ordinary Shares Number", "Share Issued"),
        1,
    ),
    "book_value_per_share": ("derived", ("book_value_per_share",), 1),
    "working_capital": ("balance", ("Working Capital",), 1),
}


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    try:
        t = yf.Ticker(ticker)
    except Exception:
        return []

    use_quarterly = period in ("quarterly", "quarter", "q", "ttm")
    income = t.quarterly_income_stmt if use_quarterly else t.income_stmt
    balance = t.quarterly_balance_sheet if use_quarterly else t.balance_sheet
    cashflow = t.quarterly_cashflow if use_quarterly else t.cashflow

    statements = {"income": income, "balance": balance, "cashflow": cashflow}

    # gather and sort period columns
    periods: list[pd.Timestamp] = []
    for src in statements.values():
        if src is not None and not src.empty:
            periods.extend(list(src.columns))
    periods = sorted({p for p in periods}, reverse=True)
    try:
        end_ts = pd.Timestamp(end_date)
        periods = [p for p in periods if pd.Timestamp(p) <= end_ts]
    except Exception:
        pass
    periods = periods[:limit]

    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass
    currency = info.get("financialCurrency") or info.get("currency") or "USD"

    # Pre-fetch shares for book_value_per_share derivation
    shares_row = _row(balance, "Ordinary Shares Number", "Share Issued")
    equity_row = _row(balance, "Stockholders Equity", "Common Stock Equity")

    results: list[LineItem] = []
    for col in periods:
        item = LineItem(
            ticker=ticker,
            report_period=_date_str(col),
            period=period,
            currency=currency,
        )
        for name in line_items:
            mapping = _LINE_ITEM_MAP.get(name)
            if mapping is None:
                setattr(item, name, None)
                continue
            statement, row_names, sign = mapping
            if statement == "derived" and name == "book_value_per_share":
                eq = _value_at(equity_row, col)
                sh = _value_at(shares_row, col)
                if eq is not None and sh:
                    setattr(item, name, eq / sh)
                else:
                    setattr(item, name, None)
                continue
            df = statements.get(statement)
            row = _row(df, *row_names)
            value = _value_at(row, col)
            if value is not None:
                value = value * sign
            setattr(item, name, value)
        results.append(item)

    return results


# ---------------------------------------------------------------------------
# insider trades
# ---------------------------------------------------------------------------

def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[InsiderTrade]:
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}_yf"
    if cached := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**t) for t in cached]

    try:
        t = yf.Ticker(ticker)
        df = t.insider_transactions
    except Exception:
        df = None

    if df is None or df.empty:
        return []

    trades: list[InsiderTrade] = []
    for _, row in df.iterrows():
        try:
            start_ts = pd.Timestamp(row.get("Start Date")) if "Start Date" in row else None
        except Exception:
            start_ts = None
        filing_date = _date_str(start_ts) if start_ts is not None else end_date

        # date filtering
        if start_date and filing_date < start_date:
            continue
        if filing_date > end_date:
            continue

        shares = _safe_float(row.get("Shares"))
        value = _safe_float(row.get("Value"))
        text = str(row.get("Text") or "").lower()
        # treat sales/dispositions as negative shares
        if shares is not None and ("sale" in text or "sold" in text or "disposition" in text):
            shares = -abs(shares)
            if value is not None:
                value = -abs(value)

        price = None
        if shares and value:
            try:
                price = abs(value) / abs(shares)
            except ZeroDivisionError:
                price = None

        trades.append(
            InsiderTrade(
                ticker=ticker,
                issuer=None,
                name=row.get("Insider"),
                title=row.get("Position"),
                is_board_director=None,
                transaction_date=filing_date,
                transaction_shares=shares,
                transaction_price_per_share=price,
                transaction_value=value,
                shares_owned_before_transaction=None,
                shares_owned_after_transaction=_safe_float(row.get("Ownership")),
                security_title=row.get("Transaction"),
                filing_date=filing_date,
            )
        )
        if len(trades) >= limit:
            break

    if trades:
        _cache.set_insider_trades(cache_key, [t.model_dump() for t in trades])
    return trades


# ---------------------------------------------------------------------------
# company news
# ---------------------------------------------------------------------------

def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}_yf"
    if cached := _cache.get_company_news(cache_key):
        return [CompanyNews(**n) for n in cached]

    try:
        t = yf.Ticker(ticker)
        raw = t.news or []
    except Exception:
        raw = []

    items: list[CompanyNews] = []
    for entry in raw:
        # yfinance >=0.2.40 wraps payload under "content"
        content = entry.get("content") if isinstance(entry, dict) else None
        if content:
            title = content.get("title") or ""
            url = (content.get("canonicalUrl") or {}).get("url") or (content.get("clickThroughUrl") or {}).get("url") or ""
            publisher = (content.get("provider") or {}).get("displayName") or ""
            pub_date = content.get("pubDate") or content.get("displayTime") or ""
            date_str = _date_str(pub_date) if pub_date else end_date
        else:
            title = entry.get("title", "")
            url = entry.get("link", "")
            publisher = entry.get("publisher", "")
            ts = entry.get("providerPublishTime")
            if ts:
                date_str = _dt.datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
            else:
                date_str = end_date

        if start_date and date_str < start_date:
            continue
        if date_str > end_date:
            continue

        items.append(
            CompanyNews(
                ticker=ticker,
                title=title or "(no title)",
                author=publisher or "",
                source=publisher or "Yahoo Finance",
                date=date_str,
                url=url or "",
                sentiment=None,
            )
        )
        if len(items) >= limit:
            break

    if items:
        _cache.set_company_news(cache_key, [n.model_dump() for n in items])
    return items


# ---------------------------------------------------------------------------
# market cap
# ---------------------------------------------------------------------------

def get_market_cap(ticker: str, end_date: str) -> float | None:
    today = _dt.datetime.now().strftime("%Y-%m-%d")
    if end_date >= today:
        try:
            info = yf.Ticker(ticker).info or {}
        except Exception:
            info = {}
        mc = _safe_float(info.get("marketCap"))
        if mc is not None:
            return mc

    metrics = get_financial_metrics(ticker, end_date)
    if metrics and metrics[0].market_cap:
        return metrics[0].market_cap
    return None


def get_latest_price(ticker: str) -> float:
    """Get the latest closing price for a ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL")
    
    Returns:
        Latest closing price as a float
    
    Raises:
        ValueError: If no data is available or price cannot be retrieved
    """
    try:
        df = yf.Ticker(ticker).history(period="1d")
        if df is None or df.empty:
            raise ValueError(f"No data available for ticker: {ticker}")
        latest_price = df["Close"].iloc[-1]
        if pd.isna(latest_price):
            raise ValueError(f"Price data is unavailable for ticker: {ticker}")
        return float(latest_price)
    except Exception as e:
        raise ValueError(f"Failed to fetch latest price for {ticker}: {str(e)}")

# pip install requests beautifulsoup4 pandas yfinance

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


URL_TRENDING = "https://finance.yahoo.com/markets/stocks/trending/"
URL_MOST_ACTIVE = "https://finance.yahoo.com/markets/stocks/most-active/"

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

RUN_DATE = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "name",
    "price",
    "change",
    "percent_change",
    "sector",
    "industry",
]


def clean_number(value):
    """
    Convert strings/numbers into clean floats where possible.
    """
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    value = (
        value.replace(",", "")
             .replace("%", "")
             .replace("+", "")
             .replace("$", "")
    )

    try:
        return float(value)
    except ValueError:
        return value


def _fast_info_get(fast_info, keys):
    """
    yfinance fast_info can behave like a dict-like object.
    This helper tries several safe access patterns.
    """
    if fast_info is None:
        return None

    for key in keys:
        try:
            value = fast_info[key]
            if value is not None and str(value).strip() != "":
                return value
        except Exception:
            pass

        try:
            value = fast_info.get(key)
            if value is not None and str(value).strip() != "":
                return value
        except Exception:
            pass

        try:
            value = getattr(fast_info, key)
            if value is not None and str(value).strip() != "":
                return value
        except Exception:
            pass

    return None


def _sector_industry_from_info(info: dict):
    """
    Normalize sector/industry from possible yfinance keys.
    """
    if not info:
        return None, None

    sector = info.get("sector") or info.get("Sector")
    industry = info.get("industry") or info.get("Industry")

    return sector, industry


def fetch_ticker_details(symbol: str) -> dict:
    """
    Fetch price, change, percent_change, sector, and industry from yfinance.

    This avoids relying on Yahoo's changing HTML table structure.
    """
    result = {
        "symbol": symbol,
        "price": None,
        "change": None,
        "percent_change": None,
        "sector": None,
        "industry": None,
    }

    try:
        tkr = yf.Ticker(symbol)

        price = None
        previous_close = None

        # 1. Try fast_info first
        try:
            fast_info = tkr.fast_info

            price = _fast_info_get(
                fast_info,
                [
                    "lastPrice",
                    "last_price",
                    "regularMarketPrice",
                    "regular_market_price",
                ],
            )

            previous_close = _fast_info_get(
                fast_info,
                [
                    "previousClose",
                    "previous_close",
                    "regularMarketPreviousClose",
                    "regular_market_previous_close",
                ],
            )
        except Exception:
            pass

        # 2. Fallback: latest intraday close
        if price is None:
            try:
                hist_intraday = tkr.history(period="1d", interval="1m", prepost=False)
                close_series = hist_intraday["Close"].dropna()

                if not close_series.empty:
                    price = close_series.iloc[-1]
            except Exception:
                pass

        # 3. Fallback: daily history for previous close
        if previous_close is None:
            try:
                hist_daily = tkr.history(period="5d", interval="1d", prepost=False)
                close_series = hist_daily["Close"].dropna()

                if len(close_series) >= 2:
                    previous_close = close_series.iloc[-2]
                elif len(close_series) == 1 and price is None:
                    price = close_series.iloc[-1]
            except Exception:
                pass

        price = clean_number(price)
        previous_close = clean_number(previous_close)

        result["price"] = price

        if price is not None and previous_close is not None and previous_close != 0:
            change = price - previous_close
            percent_change = (change / previous_close) * 100

            result["change"] = round(change, 6)
            result["percent_change"] = round(percent_change, 6)

        # 4. Sector / industry
        try:
            info = tkr.get_info() or {}
        except Exception:
            info = {}

        sector, industry = _sector_industry_from_info(info)

        if not sector or not industry:
            try:
                profile = tkr.get_summary_profile() or {}
            except Exception:
                profile = {}

            sector = sector or profile.get("sector")
            industry = industry or profile.get("industry")

        result["sector"] = sector
        result["industry"] = industry

    except Exception:
        pass

    return result


def _parse_yahoo_table(url: str, max_workers: int = 10) -> pd.DataFrame:
    """
    Parse ticker and name from Yahoo's table, then fetch quote details via yfinance.
    """
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    rows = []
    seen_tickers = set()

    for row in soup.select("table tbody tr"):
        cols = row.find_all("td")

        if len(cols) < 2:
            continue

        ticker = cols[0].get_text(strip=True)
        name = cols[1].get_text(strip=True)

        if not ticker or ticker in seen_tickers:
            continue

        seen_tickers.add(ticker)

        rows.append({
            "date": RUN_DATE,
            "ticker": ticker,
            "name": name,
            "price": None,
            "change": None,
            "percent_change": None,
            "sector": None,
            "industry": None,
        })

    if not rows:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Enrich all tickers concurrently
    details_by_symbol = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_ticker_details, row["ticker"]): row["ticker"]
            for row in rows
        }

        for future in as_completed(futures):
            details = future.result()
            details_by_symbol[details["symbol"]] = details

    for row in rows:
        details = details_by_symbol.get(row["ticker"], {})

        row["price"] = details.get("price")
        row["change"] = details.get("change")
        row["percent_change"] = details.get("percent_change")
        row["sector"] = details.get("sector")
        row["industry"] = details.get("industry")

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def get_trending_data(max_workers: int = 10) -> pd.DataFrame:
    """
    Yahoo Finance Trending stocks.
    """
    return _parse_yahoo_table(URL_TRENDING, max_workers=max_workers)


def get_most_active_data(max_workers: int = 10) -> pd.DataFrame:
    """
    Yahoo Finance Most Active stocks.
    """
    return _parse_yahoo_table(URL_MOST_ACTIVE, max_workers=max_workers)


def save_replace_run_date(df: pd.DataFrame, csv_path: str) -> None:
    """
    Replace existing rows for RUN_DATE, then save the full CSV.
    This prevents duplicate rows when the script is run more than once per day.
    """
    directory = os.path.dirname(csv_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    if os.path.isfile(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            existing_df = pd.DataFrame(columns=OUTPUT_COLUMNS)

        if "date" in existing_df.columns:
            existing_df["date"] = existing_df["date"].astype(str)
            existing_df = existing_df[existing_df["date"] != RUN_DATE]

        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df.copy()

    combined_df = combined_df[OUTPUT_COLUMNS]
    combined_df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # --------- Trending ---------
    df_trending = get_trending_data()
    print("Trending:")
    print(df_trending.to_string(index=False))

    save_replace_run_date(df_trending, "data/trending.csv")

    # --------- Most Active ---------
    df_most_active = get_most_active_data()
    print("\nMost Active:")
    print(df_most_active.to_string(index=False))

    save_replace_run_date(df_most_active, "data/most_active.csv")

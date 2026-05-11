# pip install requests beautifulsoup4 pandas yfinance

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

URL_TRENDING = "https://finance.yahoo.com/markets/stocks/trending/"
URL_MOST_ACTIVE = "https://finance.yahoo.com/markets/stocks/most-active/"
HEADERS = {
    "User-Agent": "Mozilla/5.0"
}
RUN_DATE = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
# ---------- helpers for sector / industry via yfinance ----------

def _sector_industry_from_info(info: dict):
    """Try to normalize sector/industry from different possible keys."""
    if not info:
        return None, None
    sector = info.get("sector") or info.get("Sector")
    industry = info.get("industry") or info.get("Industry")
    return sector, industry

def fetch_sector_industry(symbol: str) -> dict:
    """
    Fetch sector/industry for a symbol (using yfinance),
    returning {"symbol", "sector", "industry"}.
    """
    result = {"symbol": symbol, "sector": None, "industry": None}
    try:
        tkr = yf.Ticker(symbol)

        # Primary: get_info (aggregate metadata)
        try:
            info = tkr.get_info()
        except Exception:
            info = {}

        sector, industry = _sector_industry_from_info(info)

        # Fallback: get_summary_profile (profile-only dict)
        if not sector or not industry:
            try:
                prof = tkr.get_summary_profile() or {}
            except Exception:
                prof = {}
            sector = sector or prof.get("sector")
            industry = industry or prof.get("industry")

        result["sector"] = sector
        result["industry"] = industry
    except Exception:
        # Keep defaults (None) if anything fails
        pass

    return result


# ---------- generic parser for a Yahoo table page ----------
def clean_number(value):
    """
    Convert Yahoo-style strings like '+1.23', '13.90%', '1,234.56'
    into floats where possible.
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
def fetch_price_fallback(symbol: str):
    """
    Fallback price from yfinance when Yahoo page parsing gives empty price.
    """
    try:
        tkr = yf.Ticker(symbol)

        # Try fast_info first
        try:
            fast_info = tkr.fast_info

            for key in [
                "last_price",
                "lastPrice",
                "regular_market_price",
                "regularMarketPrice",
            ]:
                try:
                    value = fast_info.get(key)
                except Exception:
                    value = getattr(fast_info, key, None)

                if value is not None and str(value).strip() != "":
                    return value
        except Exception:
            pass

        # Try info next
        try:
            info = tkr.get_info() or {}

            for key in [
                "currentPrice",
                "regularMarketPrice",
                "previousClose",
            ]:
                value = info.get(key)

                if value is not None and str(value).strip() != "":
                    return value
        except Exception:
            pass

        # Final fallback: latest close from history
        try:
            hist = tkr.history(period="5d", interval="1d")

            if not hist.empty:
                return hist["Close"].dropna().iloc[-1]
        except Exception:
            pass

    except Exception:
        pass

    return None
def clean_number(value):
    """
    Convert Yahoo/yfinance values into clean numeric values where possible.
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
def _parse_yahoo_table(url: str, max_workers: int = 10) -> pd.DataFrame:
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    rows = []
    tickers = []

    # Parse each row of the table
    for row in soup.select("table tbody tr"):
        cols = row.find_all("td")
        if len(cols) < 5:
            continue

        # ticker & name
        ticker = cols[0].get_text(strip=True)
        name = cols[1].get_text(strip=True)

        # --- robust price parsing ---
        price = None
        
        price_tag = row.select_one('fin-streamer[data-field="regularMarketPrice"]')
        
        if price_tag is not None:
            price = price_tag.get("data-value") or price_tag.get_text(strip=True)
        
        # Yahoo sometimes returns the tag but leaves it empty
        if not price and len(cols) > 2:
            price = cols[2].get_text(strip=True)
        
        # yfinance fallback
        if not price:
            price = fetch_price_fallback(ticker)
        change_tag = row.select_one('fin-streamer[data-field="regularMarketChange"]')
        pct_tag = row.select_one('fin-streamer[data-field="regularMarketChangePercent"]')

        change = (change_tag.get("data-value")
                  if change_tag is not None else cols[3].get_text(strip=True))
        percent_change = (pct_tag.get("data-value")
                          if pct_tag is not None else cols[4].get_text(strip=True))

        rows.append({
            "date": RUN_DATE,
            "ticker": ticker,
            "name": name,
            "price": clean_number(price),
            "change": clean_number(change),
            "percent_change": clean_number(percent_change),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "ticker", "name", "price",
            "change", "percent_change", "sector", "industry"
        ])

    # ---------- enrich with sector & industry (concurrently) ----------
    meta_by_symbol = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_sector_industry, sym): sym for sym in tickers}
        for fut in as_completed(futures):
            meta = fut.result()
            meta_by_symbol[meta["symbol"]] = meta

    for row in rows:
        meta = meta_by_symbol.get(row["ticker"], {})
        row["sector"] = meta.get("sector")
        row["industry"] = meta.get("industry")

    return pd.DataFrame(rows)


# ---------- public functions ----------

def get_trending_data(max_workers: int = 10) -> pd.DataFrame:
    """Yahoo Finance 'Trending' stocks."""
    return _parse_yahoo_table(URL_TRENDING, max_workers=max_workers)

def get_most_active_data(max_workers: int = 10) -> pd.DataFrame:
    """Yahoo Finance 'Most Active' stocks."""
    return _parse_yahoo_table(URL_MOST_ACTIVE, max_workers=max_workers)
def save_replace_run_date(df: pd.DataFrame, csv_path: str) -> None:
    """
    Save today's scraped data by replacing existing rows for RUN_DATE.
    If the file does not exist, create it.
    If the file exists, remove rows with the same date and append the new rows.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.isfile(csv_path):
        existing_df = pd.read_csv(csv_path)

        if "date" in existing_df.columns:
            existing_df["date"] = existing_df["date"].astype(str)
            existing_df = existing_df[existing_df["date"] != RUN_DATE]

        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df.copy()

    combined_df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # --------- Trending ---------
    df_trending = get_trending_data()
    print("Trending:")
    print(df_trending.to_string(index=False))

    trending_csv = "data/trending.csv"
    save_replace_run_date(df_trending, trending_csv)

    # --------- Most Active ---------
    df_most_active = get_most_active_data()
    print("\nMost Active:")
    print(df_most_active.to_string(index=False))

    most_active_csv = "data/most_active.csv"
    save_replace_run_date(df_most_active, most_active_csv)

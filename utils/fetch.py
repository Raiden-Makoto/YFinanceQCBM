import yfinance as yf # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os
import time
from datetime import datetime

# --- CONFIGURATION ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
TICKER = "VFV.TO"
CACHE_FILE = os.path.join(data_dir, "vfv_market_data.csv")
CACHE_DURATION_SECONDS = 900  # 15 minutes

def get_vfv_data():
    """
    Fetches VFV.TO market data. Use a local CSV cache to avoid 
    Yahoo Finance rate limits if called frequently.
    """
    # 1. Check if cache exists and is fresh
    if os.path.exists(CACHE_FILE):
        last_modified = os.path.getmtime(CACHE_FILE)
        age_seconds = time.time() - last_modified
        
        if age_seconds < CACHE_DURATION_SECONDS:
            print(f"--- [CACHE HIT] Loading data from {CACHE_FILE} ---")
            print(f"--- Data age: {int(age_seconds // 60)}m {int(age_seconds % 60)}s ---")
            return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)

    # 2. Fetch fresh data if cache is stale or missing
    print(f"--- [CACHE MISS/STALE] Fetching fresh data for {TICKER}... ---")
    try:
        # Fetching 5 days of 1-minute interval data
        # We use 5d to ensure we capture the most recent Friday if it's the weekend
        data = yf.download(TICKER, period="5d", interval="1m", progress=False)
        
        if data.empty:
            print("Warning: No data returned from API.")
            return None
        
        # Save to cache
        data.to_csv(CACHE_FILE)
        print("--- [SUCCESS] Cache updated with fresh data ---")
        return data

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        # Fallback: if API fails, try to return stale cache if it exists
        if os.path.exists(CACHE_FILE):
            print("--- [FALLBACK] Returning stale cache data ---")
            return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return None

if __name__ == "__main__":
    vfv_df = get_vfv_data()
    if vfv_df is not None:
        print("\nLatest Market Snapshot:")
        print(vfv_df.tail(5))
        
        # FIX: Use .values to get the raw number and flatten it 
        # to avoid the MultiIndex Series error
        last_close = vfv_df['Close'].values[-1]
        
        # Handle case where it might still be an array
        if isinstance(last_close, (list, np.ndarray)):
            last_close = last_close[0]

        print(f"\nMost recent close: ${float(last_close):.2f}")
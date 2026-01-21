import yfinance as yf
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
TICKER = "VFV.TO"
CACHE_FILE = os.path.join(data_dir, "vfv_market_data.csv")
CACHE_DURATION_SECONDS = 900  # 15 minutes default

def sync_market_clock():
    """
    Blocks execution until the start of the next minute + buffer.
    Ensures we pull data right after the candle closes.
    """
    now = datetime.now()
    # Calculate seconds until the next minute mark (xx:xx:00)
    # We add 2 seconds buffer to allow Yahoo's API to propagate the close
    sleep_seconds = 60 - now.second + 2
    
    if sleep_seconds < 5:
        # If we are too close to the boundary (e.g., xx:xx:59), wait an extra minute
        sleep_seconds += 60
        
    next_pull = now + timedelta(seconds=sleep_seconds)
    print(f"   [SYNC] Waiting {sleep_seconds}s for candle close ({next_pull.strftime('%H:%M:%S')})...")
    time.sleep(sleep_seconds)

def get_vfv_data(force_refresh=False):
    """
    Fetches VFV.TO market data. 
    Args:
        force_refresh (bool): If True, ignores cache timer and forces API pull.
    """
    # 1. Check if cache exists and is fresh (unless forced)
    if os.path.exists(CACHE_FILE) and not force_refresh:
        last_modified = os.path.getmtime(CACHE_FILE)
        age_seconds = time.time() - last_modified
        
        if age_seconds < CACHE_DURATION_SECONDS:
            print(f"--- [CACHE HIT] Loading data from {CACHE_FILE} ---")
            return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)

    # 2. Fetch fresh data
    print(f"--- [{'FORCE' if force_refresh else 'STALE'}] Fetching fresh data for {TICKER}... ---")
    try:
        # Fetching 5 days to ensure continuity
        data = yf.download(TICKER, period="5d", interval="1m", progress=False)
        
        if data.empty:
            print("Warning: No data returned from API.")
            return None
        
        # Save to cache (Your existing logic)
        data.to_csv(CACHE_FILE)
        print("--- [SUCCESS] Cache updated with fresh data ---")
        return data

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        # Fallback to cache if API fails
        if os.path.exists(CACHE_FILE):
            print("--- [FALLBACK] Returning stale cache data ---")
            return pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
        return None

if __name__ == "__main__":
    # Test the sync and force logic
    # sync_market_clock() 
    df = get_vfv_data(force_refresh=True)
    if df is not None:
        print(f"Latest Close: {df['Close'].iloc[-1]}")
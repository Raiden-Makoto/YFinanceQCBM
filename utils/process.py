import pandas as pd # type: ignore
import numpy as np # type: ignore
import torch # type: ignore

import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
CACHE_FILE = os.path.join(data_dir, "vfv_market_data.csv")

WINDOW_SIZE = 15

def get_processed_tensors():
    """
    Reads the yfinance CSV, cleans MultiIndex headers, 
    and converts prices into normalized 15-element windows.
    """
    if not os.path.exists(CACHE_FILE):
        print(f"Error: {CACHE_FILE} not found. Run your fetcher script first.")
        return None

    # Load CSV with MultiIndex (Price/Ticker)
    # yfinance saves two header rows. header=[0,1] ensures we capture both.
    df = pd.read_csv(CACHE_FILE, header=[0, 1], index_col=0, parse_dates=True)

    # Flatten MultiIndex
    # Converts (Price, VFV.TO) -> Price. This allows df['Close'] to work.
    df.columns = df.columns.get_level_values(0)

    # Extract Close prices and convert to float
    # errors='coerce' turns any non-numeric strings (like Ticker names) into NaN
    prices = pd.to_numeric(df['Close'], errors='coerce').dropna().values
    
    if len(prices) < WINDOW_SIZE + 1:
        print(f"Error: Not enough data. Need at least {WINDOW_SIZE + 1} points.")
        return None

    # Calculate Log Returns
    # r_t = ln(P_t / P_{t-1})
    # This results in a vector of length len(prices) - 1
    log_returns = np.log(prices[1:] / prices[:-1])

    # Create Sliding Windows
    windows = []
    for i in range(len(log_returns) - WINDOW_SIZE + 1):
        window = log_returns[i : i + WINDOW_SIZE]
        
        # Z-Score Normalization
        # (x - mean) / std_dev
        # Essential for Quantum Angle Embedding to avoid saturation
        mu = np.mean(window)
        std = np.std(window)
        
        if std > 1e-9:
            norm_window = (window - mu) / std
        else:
            norm_window = window - mu  # Handle zero-variance cases
            
        windows.append(norm_window)

    # Convert to PyTorch Tensor
    return torch.tensor(np.array(windows), dtype=torch.float32)

if __name__ == "__main__":
    tensors = get_processed_tensors()
    if tensors is not None:
        print("--- Processing Complete ---")
        print(f"Tensor Shape: {tensors.shape}")  # Should be [N, 15]
        print("\nFirst Window Example:")
        print(tensors[0])
        print("\nLatest Window Example (Last 15 minutes):")
        print(tensors[-1])
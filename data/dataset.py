import pandas as pd # type: ignore
import torch # type: ignore
from torch.utils.data import Dataset # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
 
class VFVDataset(Dataset):
    def __init__(self, csv_path: str, window_size: int = 15):
        # 1. Load CSV, skipping the 'Ticker' and empty 'Datetime' rows
        # Based on your file, we skip rows 1 and 2 (0-indexed)
        df = pd.read_csv(csv_path, skiprows=[1, 2])
        
        # 2. Force 'Close' to numeric and drop any failed conversions
        # The 'Price' column actually contains the Datetime in your CSV
        prices = pd.to_numeric(df['Close'], errors='coerce').dropna().values
        
        # 3. Calculate log returns: log(P_t / P_{t-1})
        # This makes the data 'stationary' (meaning it has a constant mean/variance)
        returns = pd.Series(prices).pct_change().dropna().values
        
        # 4. Normalize to Z-scores (mean=0, std=1)
        # This is vital for Quantum Circuits which are sensitive to input scales
        self.scaler = StandardScaler()
        returns_scaled = self.scaler.fit_transform(returns.reshape(-1, 1)).flatten()
        
        # 5. Create Sliding Windows of 15 minutes
        self.windows = []
        for i in range(len(returns_scaled) - window_size):
            self.windows.append(returns_scaled[i : i + window_size])
            
        self.windows = torch.tensor(self.windows, dtype=torch.float32)
        print(f"Dataset Loaded: {len(self.windows)} windows of {window_size} minutes.")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
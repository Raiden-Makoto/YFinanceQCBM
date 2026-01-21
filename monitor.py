import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from datetime import datetime

# --- IMPORTS ---
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from model.generator import QuantumGenerator
from model.discriminator import Discriminator
from utils.fetch import get_vfv_data, sync_market_clock
from utils.process import get_processed_tensors

# --- CONFIG ---
WEIGHTS_PATH = "vfv_wgan_final.pt"

# --- TRADING SETTINGS ---
# We need the trend to be strong (> 5%) to care.
RISE_THRESHOLD = 0.05   
DROP_THRESHOLD = -0.05
CONFIRMATION_MINUTES = 3 # Wait 3 minutes to confirm a move is real

# --- 1. SETUP SYSTEM ---
def load_brain():
    print(">> Initializing Quantum Core...")
    gen = QuantumGenerator()
    disc = Discriminator()

    if not os.path.exists(WEIGHTS_PATH):
        print(f"CRITICAL: Weights '{WEIGHTS_PATH}' not found.")
        sys.exit(1)
        
    try:
        gen.load_state_dict(torch.load(WEIGHTS_PATH))
        gen.eval()
        disc.eval()
        return gen, disc
    except Exception as e:
        print(f"Weight Load Error: {e}")
        sys.exit(1)

# --- 2. MAIN MONITOR ---
def main():
    gen, disc = load_brain()
    ANOMALY_THRESHOLD = -0.5 

    # --- MEMORY ---
    # These count how many minutes a signal has lasted
    rising_streak = 0
    dropping_streak = 0
    
    # This is the "Final Decision" displayed to you
    current_status = "WAITING FOR DATA..."

    print("\nQUANTUM SENTINEL ACTIVE [Plain English Mode]")
    print(f"Sensitivity: {RISE_THRESHOLD} | Confirmation Time: {CONFIRMATION_MINUTES} Minutes")

    while True:
        try:
            # 1. Sync Time & Get Data
            sync_market_clock()
            print(">> Checking Market...")
            get_vfv_data(force_refresh=True)

            # 2. Process Data
            all_tensors = get_processed_tensors()
            
            if all_tensors is None or all_tensors.shape[0] == 0:
                print("Not enough data yet...")
                continue
            
            last_window = all_tensors[-1].unsqueeze(0)

            # 3. AI Prediction
            with torch.no_grad():
                critic_score = disc(last_window).item()
                futures = gen(batch_size=200).detach().numpy()
                trend = np.mean(futures) # Average predicted move

            # 4. Logic Engine
            # Determine what the AI sees RIGHT NOW
            if trend > RISE_THRESHOLD:
                instant_signal = "RISING"
            elif trend < DROP_THRESHOLD:
                instant_signal = "DROPPING"
            else:
                instant_signal = "FLAT"

            # Update Streaks (Confirmation Logic)
            if instant_signal == "RISING":
                rising_streak += 1
                dropping_streak = 0
            elif instant_signal == "DROPPING":
                dropping_streak += 1
                rising_streak = 0
            else:
                # If it goes flat, we reset the counters to avoid false alarms
                rising_streak = 0
                dropping_streak = 0

            # Make Final Decision
            # Only change the status if we have seen the same signal for 3 minutes straight
            if rising_streak >= CONFIRMATION_MINUTES:
                current_status = "DETECTED UPTREND -> BUY/HOLD"
                # Cap the counter so it doesn't count to infinity
                rising_streak = CONFIRMATION_MINUTES 
            elif dropping_streak >= CONFIRMATION_MINUTES:
                current_status = "DETECTED DOWNTREND -> SELL/EXIT"
                dropping_streak = CONFIRMATION_MINUTES
            elif instant_signal == "FLAT":
                 current_status = "MARKET STABLE (NO ACTION)"

            # 5. Display Output
            os.system('cls' if os.name == 'nt' else 'clear')
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Anomaly Warning (Overrides everything)
            if critic_score < ANOMALY_THRESHOLD:
                health_status = "CRITICAL WARNING (Crash Risk)"
                current_status = "CASH OUT IMMEDIATELY"
            else:
                health_status = "Normal"

            print(f"--- VFV.TO MONITOR [{timestamp}] ---")
            print(f"Market Health: {health_status}")
            print(f"Safety Score:  {critic_score:.4f}")
            print("------------------------------------------")
            print(f"AI Prediction: {instant_signal}")
            print(f"Strength:      {trend:+.4f}")
            print(f"Confirmation: Rising [{rising_streak}/{CONFIRMATION_MINUTES}]  Dropping [{dropping_streak}/{CONFIRMATION_MINUTES}]")
            print("------------------------------------------")
            print(f"FINAL ADVICE:  >>> {current_status} <<<")
            print("Disclaimer: AI model may not always be accurate. Use at own risk.")
            print("------------------------------------------")
            
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
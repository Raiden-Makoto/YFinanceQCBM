import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from datetime import datetime
from termcolor import colored

# --- IMPORTS ---
# Add project root to path so we can import modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from model.generator import QuantumGenerator
from model.discriminator import Discriminator

# Import your helper functions
from utils.fetch import get_vfv_data, sync_market_clock
from utils.process import get_processed_tensors

# --- CONFIG ---
WEIGHTS_PATH = "vfv_wgan_final.pt"

# --- 1. SETUP SYSTEM ---
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def load_brain():
    print(colored(">> Initializing Quantum Core...", "cyan"))
    gen = QuantumGenerator()
    disc = Discriminator()
    disc.model[-1] = nn.Identity() # WGAN Mode (No Sigmoid)

    if not os.path.exists(WEIGHTS_PATH):
        print(colored(f"❌ CRITICAL: Weights '{WEIGHTS_PATH}' not found.", "red"))
        sys.exit(1)
        
    try:
        gen.load_state_dict(torch.load(WEIGHTS_PATH))
        gen.eval()
        disc.eval()
        print(colored("✓ Neural Weights Loaded.", "green"))
        return gen, disc
    except Exception as e:
        print(colored(f"❌ Weight Load Error: {e}", "red"))
        sys.exit(1)

# --- 2. MAIN WATCHTOWER ---
def main():
    gen, disc = load_brain()
    
    # Calibration thresholds (You can auto-calc this, but hardcoding prevents startup lag)
    ANOMALY_THRESHOLD = -0.5 

    print(colored(f"\n   QUANTUM SENTINEL ACTIVE", "cyan", attrs=['bold']))
    print("   (Ctrl+C to stop)\n")

    while True:
        try:
            # A. SYNC CLOCK (Wait for :00 seconds)
            sync_market_clock()

            # B. FETCH (Force update to CSV)
            print(">> Updating Market Data Cache...")
            # We use your utils/fetch.py logic
            raw_data = get_vfv_data(force_refresh=True)
            
            if raw_data is None:
                print(colored("⚠ API Error. Retrying next cycle.", "yellow"))
                continue

            # C. PROCESS (Read CSV -> Tensor)
            # We use your utils/process.py logic
            all_tensors = get_processed_tensors()
            
            if all_tensors is None or len(all_tensors) == 0:
                print(colored("⚠ Processing Error: No valid windows found.", "yellow"))
                continue
            
            # [CRITICAL FIX] Ensure we slice exactly ONE window
            # all_tensors is [N, 15]. We want the last one: [15]
            # unsqueeze(0) makes it [1, 15], which fits the model.
            last_window = all_tensors[-1].unsqueeze(0)

            # D. INFERENCE
            with torch.no_grad():
                # 1. Critic (Anomaly Check)
                # This .item() call is where your code was crashing.
                # Now that shape is guaranteed [1, 15] -> Output [1, 1] -> Scalar OK.
                critic_score = disc(last_window).item()
                
                # 2. Generator (Trend Prediction)
                futures = gen(batch_size=200).detach().numpy()
                trend = np.mean(futures)
                vol = np.std(futures)

            # E. RENDER DASHBOARD
            clear_screen()
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # State Logic
            if critic_score < ANOMALY_THRESHOLD:
                status = "🚨 ANOMALY"
                col = "red"
            else:
                status = "✅ NORMAL"
                col = "green"

            # Trend Logic
            if trend > 0.02:
                signal = "LONG 🚀"
                sig_col = "green"
            elif trend < -0.02:
                signal = "SHORT 📉"
                sig_col = "red"
            else:
                signal = "HOLD ⏸"
                sig_col = "yellow"

            print(colored("╔════════════════════════════════════════════╗", "cyan"))
            print(colored(f"║   QUANTUM VFV.TO ENGINE | {timestamp}         ║", "cyan"))
            print(colored("╠════════════════════════════════════════════╣", "cyan"))
            print(f"║  State:   {colored(status, col):<29}║")
            print(f"║  Critic:  {critic_score:.4f}                       ║")
            print(colored("╠════════════════════════════════════════════╣", "cyan"))
            print(f"║  Signal:  {colored(signal, sig_col):<29}║")
            print(f"║  Trend:   {trend:+.4f}                           ║")
            print(f"║  Vol:     {vol:.4f}                           ║")
            print(colored("╚════════════════════════════════════════════╝", "cyan"))
            
        except KeyboardInterrupt:
            print("\nSentinel Shutdown.")
            sys.exit(0)
        except Exception as e:
            # Catch unexpected crashes and keep running
            print(colored(f"\nCRITICAL FAILURE: {e}", "red"))
            time.sleep(5)

if __name__ == "__main__":
    main()
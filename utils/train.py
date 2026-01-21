import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
import timeit

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.dataset import VFVDataset
from model.generator import QuantumGenerator
from model.discriminator import Discriminator

# --- FINAL CONFIG ---
EPOCHS = 60         
BATCH_SIZE = 64
LR_G = 0.0002
LR_D = 0.00005
CLIP_VALUE = 0.01
N_CRITIC = 1 

CSV_PATH = os.path.join(project_root, "data", "vfv_market_data.csv")

# --- SETUP ---
dataset = VFVDataset(CSV_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

gen = QuantumGenerator()
disc = Discriminator()
disc.model[-1] = nn.Identity() # WGAN Mode

opt_G = optim.RMSprop(gen.parameters(), lr=LR_G)
opt_D = optim.RMSprop(disc.parameters(), lr=LR_D)

# This acts like a "parachute" for the sine wave, forcing it to land smoothly.
scheduler_G = optim.lr_scheduler.StepLR(opt_G, step_size=15, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(opt_D, step_size=15, gamma=0.5)

w_distances = [] 

print(f"Starting Long-Run Training ({EPOCHS} Epochs)...")
starttime = timeit.default_timer()

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for i, real_windows in enumerate(pbar):
        real_windows = real_windows.float()
        batch_sz = real_windows.size(0)
        
        # --- TRAIN CRITIC ---
        opt_D.zero_grad()
        loss_real = -torch.mean(disc(real_windows))
        loss_fake = torch.mean(disc(gen(batch_sz).detach()))
        d_loss = loss_real + loss_fake
        d_loss.backward()
        opt_D.step()

        for p in disc.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        # --- TRAIN GENERATOR ---
        opt_G.zero_grad()
        g_loss = -torch.mean(disc(gen(batch_sz)))
        g_loss.backward()
        opt_G.step()
        
        # Metric: Wasserstein Distance
        dist = -d_loss.item() # approximate distance
        w_distances.append(dist)
        pbar.set_postfix(Gap=f"{dist:.4f}")

    # Step the schedulers at the end of the epoch
    scheduler_G.step()
    scheduler_D.step()

# --- PLOT THE CONVERGENCE ---
print(f"Training complete in {timeit.default_timer() - starttime}s. Generating plot...")
plt.figure(figsize=(12, 6))

# 1. Determine Cutoff (Skip first 10% of training to hide the startup spike)
cutoff = int(len(w_distances) * 0.1) 
zoomed_data = w_distances[cutoff:]
zoomed_steps = range(cutoff, len(w_distances))

# 2. Plot Raw Gap (Zoomed)
plt.plot(zoomed_steps, zoomed_data, color='purple', alpha=0.4, label="Raw Gap")

# 3. Plot Trend Line (Moving Average)
window = 50
if len(zoomed_data) > window:
    # Calculate moving avg on the ZOOMED data only
    moving_avg = [sum(zoomed_data[i:i+window])/window for i in range(len(zoomed_data)-window)]
    # Align x-axis
    plt.plot(range(cutoff+window, len(w_distances)), moving_avg, color='black', linewidth=2, label="Trend")

# 4. Force Y-Axis Focus
# We set the limits based on the ZOOMED data, ignoring the initial spike
y_min = min(zoomed_data)
y_max = max(zoomed_data)
margin = (y_max - y_min) * 0.2
plt.ylim(y_min - margin, y_max + margin)

plt.title(f"WGAN Convergence (Zoomed In - Last 90%)")
plt.xlabel("Training Steps")
plt.ylabel("Wasserstein Distance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("wgan_final_convergence.png")
print("\nTraining Complete. Check wgan_final_convergence.png")
torch.save(gen.state_dict(), "vfv_wgan_final.pt")
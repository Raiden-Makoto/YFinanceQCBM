import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.dataset import VFVDataset
from model.generator import QuantumGenerator
from model.discriminator import Discriminator

# --- FINAL CONFIG ---
EPOCHS = 150           # Increased from 50 to 300 for full convergence
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

# [NEW] Learning Rate Schedulers
# Every 50 epochs, cut the learning rate by 50%.
# This acts like a "parachute" for the sine wave, forcing it to land smoothly.
scheduler_G = optim.lr_scheduler.StepLR(opt_G, step_size=25, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(opt_D, step_size=25, gamma=0.5)

w_distances = [] 

print(f"Starting Long-Run Training ({EPOCHS} Epochs)...")

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
plt.figure(figsize=(10, 5))
plt.plot(w_distances, color='purple', alpha=0.3, label="Raw Gap")
# Plot a moving average to see the trend clearly
moving_avg = [sum(w_distances[i:i+100])/100 for i in range(len(w_distances)-100)]
plt.plot(range(100, len(w_distances)), moving_avg, color='black', linewidth=2, label="Trend (Moving Avg)")

plt.title("Final Convergence: The 'Flatline'")
plt.xlabel("Steps")
plt.ylabel("Wasserstein Distance")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("wgan_final_convergence.png")
print("\nTraining Complete. Check wgan_final_convergence.png")
torch.save(gen.state_dict(), "vfv_wgan_final.pt")
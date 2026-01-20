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

# --- CONFIG (WGAN Standard) ---
EPOCHS = 50
BATCH_SIZE = 64  # Larger batch size helps WGAN stability
LR = 0.00005     # WGAN needs very small learning rates
CLIP_VALUE = 0.01 # The "Speed Limit" for Discriminator weights
N_CRITIC = 5     # Train Discriminator 5 times for every 1 Generator step

CSV_PATH = os.path.join(project_root, "data", "vfv_market_data.csv")

# --- SETUP ---
dataset = VFVDataset(CSV_PATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

gen = QuantumGenerator()
disc = Discriminator()

# [CRITICAL FIX] Remove Sigmoid from Discriminator for WGAN
# WGAN needs unbounded output (Score), not probability (0-1)
if isinstance(disc.model[-1], nn.Sigmoid):
    print("Adjusting Discriminator for WGAN (Removing Sigmoid)...")
    disc.model[-1] = nn.Identity()

# Use RMSprop for WGAN (Standard practice over Adam)
opt_G = optim.RMSprop(gen.parameters(), lr=LR)
opt_D = optim.RMSprop(disc.parameters(), lr=LR)

d_losses = []
g_losses = []

print("Starting Wasserstein GAN Training...")

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for i, real_windows in enumerate(pbar):
        real_windows = real_windows.float()
        batch_sz = real_windows.size(0)
        
        # ==================================
        #  1. TRAIN DISCRIMINATOR (The Critic)
        # ==================================
        opt_D.zero_grad()
        
        # Real Data
        real_loss = -torch.mean(disc(real_windows)) # Maximize score for real
        
        # Fake Data
        fake_windows = gen(batch_sz).detach()
        fake_loss = torch.mean(disc(fake_windows)) # Minimize score for fake
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        opt_D.step()

        # [CRITICAL FIX] Clip weights to enforce 1-Lipschitz continuity
        for p in disc.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        # ==================================
        #  2. TRAIN GENERATOR
        # ==================================
        # Only train G every N_CRITIC steps (gives D time to estimate distance)
        if i % N_CRITIC == 0:
            opt_G.zero_grad()
            
            # Generate fresh fakes
            gen_windows = gen(batch_sz)
            
            # Generator wants to maximize the Critic's score for its fakes
            # (In WGAN, this means minimizing -D(G(z)))
            g_loss = -torch.mean(disc(gen_windows))
            
            g_loss.backward()
            opt_G.step()
            
            # Track losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            pbar.set_postfix(D_WDist=f"{-d_loss.item():.4f}", G_Score=f"{-g_loss.item():.4f}")

# --- PLOTTING ---
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label="Critic Loss (Wasserstein Dist)", color='red', alpha=0.5)
plt.plot(g_losses, label="Generator Score", color='blue', alpha=0.5)
plt.title("WGAN Training: No Spikes, Just Convergence")
plt.xlabel("Steps")
plt.ylabel("Wasserstein Estimate")
plt.legend()
plt.grid(True)
plt.savefig("wgan_training.png")
print("WGAN Training Complete. Plot saved to wgan_training.png")
torch.save(gen.state_dict(), "vfv_wgan_weights.pt")
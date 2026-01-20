import torch # type: ignore
import torch.nn as nn # type: ignore    
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from tqdm import tqdm # type: ignore
import os
import sys

# Setup project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.dataset import VFVDataset
from model.generator import QuantumGenerator
from model.discriminator import Discriminator

# --- 1. DATA LOADING ---
CACHE_FILE = os.path.join(project_root, "data", "vfv_market_data.csv")
dataset = VFVDataset(CACHE_FILE)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- 2. INITIALIZATION ---
gen = QuantumGenerator()
disc = Discriminator()

# Adversarial learning rates
optimizer_G = optim.Adam(gen.parameters(), lr=0.01)
optimizer_D = optim.Adam(disc.parameters(), lr=0.005)
criterion = nn.BCELoss()

# --- 3. TRAINING LOOP ---
EPOCHS = 50

for epoch in range(EPOCHS):
    # Setup progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
    
    for real_windows in pbar:
        # A. Prepare Data (Force Float32 for PyTorch-Quantum compatibility)
        real_windows = real_windows.float()
        batch_size = real_windows.size(0)
        
        real_labels = torch.ones(batch_size, 1).float()
        fake_labels = torch.zeros(batch_size, 1).float()

        # ---------------------
        #  B. TRAIN DISCRIMINATOR
        # ---------------------
        optimizer_D.zero_grad()
        
        # Real pass
        outputs_real = disc(real_windows)
        d_loss_real = criterion(outputs_real, real_labels)
        
        # Fake pass: PennyLane returns float64, so we cast to float() immediately
        fake_windows = torch.cat([gen().float() for _ in range(batch_size)])
        
        # Detach fake windows so we only update the Discriminator here
        outputs_fake = disc(fake_windows.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  C. TRAIN GENERATOR
        # -----------------
        optimizer_G.zero_grad()
        
        # Generator wants Discriminator to think fake windows are REAL (1)
        outputs_g = disc(fake_windows)
        g_loss = criterion(outputs_g, real_labels) 
        
        g_loss.backward()
        optimizer_G.step()

        # -----------------
        #  D. UPDATE PROGRESS
        # -----------------
        # Use set_postfix to display losses without printing new lines
        pbar.set_postfix({
            'D_loss': f"{d_loss.item():.4f}", 
            'G_loss': f"{g_loss.item():.4f}"
        })

# Save the trained generator weights for the decision engine
torch.save(gen.state_dict(), "vfv_qgan_weights.pt")
print("\nTraining Complete.")
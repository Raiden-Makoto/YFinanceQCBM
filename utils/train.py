import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from tqdm import tqdm # type: ignore

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, "data")
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
CACHE_FILE = os.path.join(data_dir, "vfv_market_data.csv")

sys.path.append(project_root)
from data.dataset import VFVDataset

# --- 1. LOAD DATASET ---
print(f"Loading dataset from {CACHE_FILE}")
dataset = VFVDataset(CACHE_FILE)
print(f"Dataset loaded with {len(dataset)} samples")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- 2. INITIALIZE MODELS ---
from model.generator import QuantumGenerator # type: ignore
from model.discriminator import Discriminator # type: ignore

print("Initializing models and optimizers...")
gen = QuantumGenerator()
disc = Discriminator()

# --- 3. OPTIMIZERS & LOSS ---
# Learning rates: We often keep the Generator's LR slightly higher or lower 
# to maintain the "adversarial balance."
lr_gen = 0.01
lr_disc = 0.005

optimizer_G = optim.Adam(gen.parameters(), lr=lr_gen)
optimizer_D = optim.Adam(disc.parameters(), lr=lr_disc)

criterion = nn.BCELoss() # Binary Cross Entropy

# --- 4. THE TRAINING LOOP ---
epochs = 10

for epoch in tqdm(range(epochs), desc="Training", total=epochs):
    for i, (real_windows,) in tqdm(enumerate(dataloader), desc="Processing Batch", total=len(dataloader), leave=False):
        batch_size = real_windows.size(0)
        
        # Labels for Real (1) and Fake (0) data
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # ---------------------
        #  TRAIN DISCRIMINATOR
        # ---------------------
        optimizer_D.zero_grad()
        
        # Score real windows
        outputs_real = disc(real_windows)
        d_loss_real = criterion(outputs_real, real_labels)
        
        # Generate and score fake windows
        # Note: we use .detach() so we don't calculate Gen gradients yet
        fake_windows = torch.cat([gen() for _ in range(batch_size)])
        outputs_fake = disc(fake_windows.detach())
        d_loss_fake = criterion(outputs_fake, fake_labels)
        
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  TRAIN GENERATOR
        # -----------------
        optimizer_G.zero_grad()
        
        # The generator wants the discriminator to think the fake data is REAL (1)
        outputs_g = disc(fake_windows)
        g_loss = criterion(outputs_g, real_labels) 
        
        g_loss.backward()
        optimizer_G.step()

        tqdm.write(f"Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

print("Training Complete. Decision Engine is live.")
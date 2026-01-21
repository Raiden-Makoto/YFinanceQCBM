from .ansatz import quantum_circuit # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from .ansatz import N_LAYERS, N_QUBITS

N_TIMESTEPS = 15

class QuantumGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(N_LAYERS, N_QUBITS, 3))
        self.shaping = nn.Linear(N_QUBITS, N_TIMESTEPS)

    def forward(self, batch_size):
        # 1. GENERATE NOISE (The "Seed")
        # Create a batch of random numbers (z)
        z = torch.randn(batch_size, N_QUBITS)
        
        # 2. RUN CIRCUIT (Broadcasted)
        # We pass 'z' as inputs. PennyLane runs 'batch_size' simulations in parallel.
        # Output shape: [15, batch_size] -> Transpose to [batch_size, 15]
        q_out = torch.stack(quantum_circuit(z, self.weights), dim=1).float()
        
        # 3. SHAPE OUTPUT
        return self.shaping(q_out)
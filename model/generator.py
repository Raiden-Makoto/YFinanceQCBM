from .ansatz import quantum_circuit # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from .ansatz import N_LAYERS, N_QUBITS

class QuantumGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Quantum Weights
        self.weights = nn.Parameter(0.01 * torch.randn(N_LAYERS, N_QUBITS, 3))
        
        # 2. Classical Shaping Layer (The "Expressivity" Boost)
        # Instead of a simple scalar multiplier, we use a Linear layer.
        # This helps map the [-1, 1] quantum output to the exact stock returns.
        self.shaping = nn.Linear(N_QUBITS, N_QUBITS)

    def forward(self, batch_size):
        # 1. Run Quantum Circuit (Once)
        # Output: [15]
        q_out = torch.stack(quantum_circuit(self.weights)).float()
        
        # 2. Duplicate for Batch
        # Output: [batch_size, 15]
        batched_out = q_out.unsqueeze(0).repeat(batch_size, 1)
        
        # 3. Apply Classical Shaping
        # This gives the generator the fine-tuning control it lacked before.
        final_out = self.shaping(batched_out)
        
        return final_out
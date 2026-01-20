from .ansatz import quantum_circuit # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from .ansatz import N_LAYERS, N_QUBITS

class QuantumGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(N_LAYERS, N_QUBITS, 3))
        self.scale = nn.Parameter(torch.tensor([2.5]))

    def forward(self):
        out = torch.stack(quantum_circuit(self.weights))
        return (out * self.scale).unsqueeze(0)

if __name__ == "__main__":
    generator = QuantumGenerator()
    fake_window = generator()
    print(f"Output Shape: {fake_window.shape}")
    print(f"Generated Fake Returns:\n{fake_window.detach().numpy()}")
    loss = fake_window.sum()
    loss.backward()
    print(f"Gradient Check (Weight Grad Sum): {generator.weights.grad.abs().sum().item():.6f}")
from ansatz import quantum_circuit
import torch # type: ignore
import torch.nn as nn # type: ignore

class QuantumGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Initializing weights with a smaller range (0.01) helps stability
        self.weights = nn.Parameter(0.01 * torch.randn(N_LAYERS, N_QUBITS, 3))
        # Scaling factor: maps quantum [-1, 1] to a more realistic return range
        self.scale = nn.Parameter(torch.tensor([2.0]))

    def forward(self):
        # Execute circuit and stack the 15 results into a single tensor
        out = torch.stack(quantum_circuit(self.weights))
        # Reshape to (1, 15) for compatibility with Discriminator batches
        return (out * self.scale).unsqueeze(0)

if __name__ == "__main__":
    # Test the generator
    generator = QuantumGenerator()
    fake_window = generator()
    
    print(f"--- 15-Qubit Hybrid Generator ---")
    print(f"Output Shape: {fake_window.shape}") # Should be [1, 15]
    print(f"Generated Fake Returns:\n{fake_window.detach().numpy()}")
    
    # Check if gradients are flowing
    loss = fake_window.sum()
    loss.backward()
    print(f"\nGradient Check (Weight Grad Sum): {generator.weights.grad.abs().sum().item():.6f}")
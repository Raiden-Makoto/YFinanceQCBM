from .ansatz import quantum_circuit # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
from .ansatz import N_LAYERS, N_QUBITS
import pennylane as qml # type: ignore

N_QUBITS = 15
N_LAYERS = 2
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(weights):
    # Standard circuit, NO BATCH DIMENSION in weights
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        for i in range(N_QUBITS - 1):
            qml.CZ(wires=[i, i+1])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class QuantumGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(N_LAYERS, N_QUBITS, 3))
        self.scale = nn.Parameter(torch.tensor([2.0]))

    def forward(self, batch_size):
        # 1. RUN CIRCUIT EXACTLY ONCE (Fast)
        # Output shape is [15]
        single_output = torch.stack(quantum_circuit(self.weights)).float()
        
        # 2. SCALE IT
        scaled_output = single_output * self.scale
        
        # 3. DUPLICATE IN PYTORCH (Instant)
        # Shape becomes [batch_size, 15] to match the Discriminator
        batched_output = scaled_output.unsqueeze(0).repeat(batch_size, 1)
        
        return batched_output
        
if __name__ == "__main__":
    generator = QuantumGenerator()
    fake_window = generator()
    print(f"Output Shape: {fake_window.shape}")
    print(f"Generated Fake Returns:\n{fake_window.detach().numpy()}")
    loss = fake_window.sum()
    loss.backward()
    print(f"Gradient Check (Weight Grad Sum): {generator.weights.grad.abs().sum().item():.6f}")
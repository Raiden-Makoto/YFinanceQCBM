import pennylane as qml # type: ignore
import torch # type: ignore

N_QUBITS = 10
N_LAYERS = 3
dev = qml.device("lightning.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="adjoint")
def quantum_circuit(inputs, weights):
    # inputs shape: [batch_size, N_QUBITS] (Random Noise)
    # weights shape: [N_LAYERS, N_QUBITS, 3] (Parameters)
    
    # 1. ENCODING LAYER (Inject the Randomness)
    # We use Angle Embedding to load the noise 'z' into the quantum state.
    # PennyLane automatically broadcasts this batch across the wires.
    for i in range(N_QUBITS):
        qml.RX(inputs[:, i], wires=i)

    # 2. VARIATIONAL LAYERS (The "Brain")
    # These weights are fixed for the batch, but applied to diverse states.
    for l in range(N_LAYERS):
        for i in range(N_QUBITS):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        
        # Circular Entanglement
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

if __name__ == "__main__":
    weights = torch.randn(N_LAYERS, N_QUBITS, 3)
    inputs = torch.randn(N_LAYERS, N_QUBITS)
    results = quantum_circuit(inputs, weights)
    print(f"Calculated {len(results)} results")
import pennylane as qml # type: ignore
import torch # type: ignore

N_QUBITS = 15
N_LAYERS = 2 
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(weights):
    # weights shape: [N_LAYERS, N_QUBITS, 3]
    for l in range(N_LAYERS):
        # 1. ROTATION BLOCK (The "Thinking" Part)
        for i in range(N_QUBITS):
            qml.RX(weights[l, i, 0], wires=i)
            qml.RY(weights[l, i, 1], wires=i)
            qml.RZ(weights[l, i, 2], wires=i)
        
        # 2. ENTANGLEMENT BLOCK (The "Correlation" Part)
        # We switch to Circular Entanglement for better connectivity
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i + 1) % N_QUBITS]) # ladder style
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

if __name__ == "__main__":
    weights = torch.randn(N_LAYERS, N_QUBITS, 3)
    results = quantum_circuit(weights)
    print(results)
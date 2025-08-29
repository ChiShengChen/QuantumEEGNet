import torch
import torch.nn as nn
import pennylane as qml


class QuantumLayer(nn.Module):
    """
    Quantum layer implementation using PennyLane.
    
    This layer implements a variational quantum circuit (VQC) that can be
    integrated into classical neural networks for hybrid quantum-classical computing.
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit
        n_layers (int): Number of layers in the quantum circuit
        device_name (str): Quantum device to use (default: "default.qubit")
    """
    
    def __init__(self, n_qubits, n_layers, device_name="default.qubit"):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name

        # Create quantum device
        dev = qml.device(device_name, wires=n_qubits)
        
        # Define quantum circuit
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Encode classical inputs into quantum states
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            
            # Apply variational layers
            for j in range(n_layers):
                # Entangle qubits using CNOT gates in ring pattern
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])
                
                # Apply rotation gates
                for i in range(n_qubits):
                    qml.RY(weights[j, i], wires=i)
            
            # Measure all qubits in Z basis
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Define weight shapes for the quantum circuit
        weight_shapes = {"weights": (n_layers, n_qubits)}
        
        # Create TorchLayer for integration with PyTorch
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
    def forward(self, x):
        """
        Forward pass through the quantum layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_qubits)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_qubits)
        """
        # Ensure x is 2D (batch_size, n_qubits)
        batch_size = x.shape[0]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Process each sample in the batch
        output = []
        for i in range(batch_size):
            result = self.q_layer(x[i])
            output.append(result)
        
        return torch.stack(output)
    
    def get_circuit_info(self):
        """
        Get information about the quantum circuit.
        
        Returns:
            dict: Circuit information including number of qubits and layers
        """
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "device": self.device_name,
            "total_parameters": self.n_layers * self.n_qubits
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# define quantum layer
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for j in range(n_layers):
                qml.broadcast(qml.CNOT, wires=range(n_qubits), pattern="ring")
                for i in range(n_qubits):
                    qml.RY(weights[j, i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)
        
    def forward(self, x):
        return self.q_layer(x)

# define quantum EEGNet
class QuantumEEGNet(nn.Module):
    def __init__(self, F1=8, D=2, F2=16, dropout_rate=0.25, num_classes=2, n_qubits=4, n_layers=2):
        super(QuantumEEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # first convolution layer
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # depthwise convolution layer
        self.conv2 = nn.Conv2d(F1, F1 * D, (2, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)

        # separable convolution layer
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F1 * D)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(F2)

        # quantum layer
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)

        # fully connected layer
        self.fc1 = nn.Linear(F2 * n_qubits, num_classes)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))
        x = self.dropout(x)
        
        # quantum layer
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.cat([self.quantum_layer(x[:, i, :]) for i in range(x.size(1))], dim=1)
        
        x = self.fc1(x)
        
        return x

# example usage:
if __name__ == "__main__":
    model = QuantumEEGNet(num_classes=2, n_qubits=4, n_layers=2)
    print(model)
    x = torch.randn(1, 1, 2, 128)  # Example input: batch size 1, 1 channel, 2 electrodes, 128 time points
    output = model(x)
    print(output)

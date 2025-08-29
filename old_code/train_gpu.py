import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Define the QuantumLayer
class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super(QuantumLayer, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device("lightning.gpu", wires=n_qubits)

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
        # Ensure x is 2D (batch_size, n_qubits)
        batch_size = x.shape[0]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        output = []
        for i in range(batch_size):
            result = self.q_layer(x[i])
            output.append(result)

        return torch.stack(output)

# Define the QuantumEEGNet
class QuantumEEGNet(nn.Module):
    def __init__(self, F1=8, D=2, F2=16, dropout_rate=0.25, num_classes=2, n_qubits=4, n_layers=2):
        super(QuantumEEGNet, self).__init__()

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F1 * D, (2, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F1 * D)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(F2)

        self.quantum_layer = QuantumLayer(n_qubits, n_layers)
        self.fc1 = nn.Linear(F2 * n_qubits, num_classes)
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

        # Reshape for the quantum layer
        x = x.view(x.size(0), x.size(1), -1)

        # Pass each channel through the quantum layer separately and concatenate the results
        quantum_outs = []
        for i in range(x.size(1)):
            quantum_out = self.quantum_layer(x[:, i, :])
            quantum_outs.append(quantum_out)

        x = torch.cat(quantum_outs, dim=1)

        x = self.fc1(x)

        return x

# Training script
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def main():
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10

    # Generate synthetic data (replace with actual data)
    X_train = torch.randn(1000, 1, 2, 128)
    y_train = torch.randint(0, 2, (1000,))
    X_test = torch.randn(200, 1, 2, 128)
    y_test = torch.randint(0, 2, (200,))

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = QuantumEEGNet(num_classes=2, n_qubits=4, n_layers=2).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data as Data
from scipy import io
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

def split_train_valid_set(x_train, y_train, ratio):
    s = y_train.argsort()
    x_train = x_train[s]
    y_train = y_train[s]

    cL = int(len(x_train) / 4)

    class1_x = x_train[0 * cL: 1 * cL]
    class2_x = x_train[1 * cL: 2 * cL]
    class3_x = x_train[2 * cL: 3 * cL]
    class4_x = x_train[3 * cL: 4 * cL]

    class1_y = y_train[0 * cL: 1 * cL]
    class2_y = y_train[1 * cL: 2 * cL]
    class3_y = y_train[2 * cL: 3 * cL]
    class4_y = y_train[3 * cL: 4 * cL]

    vL = int(len(class1_x) / ratio)

    x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
    y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))

    x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
    y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))

    return x_train, y_train, x_valid, y_valid

def getAllDataloader(subject, ratio, data_path, bs):
    train = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_T.mat'))
    test = io.loadmat(os.path.join(data_path, 'BCIC_S' + f'{subject:02d}' + '_E.mat'))

    x_train = torch.Tensor(train['x_train']).unsqueeze(1)
    y_train = torch.Tensor(train['y_train']).view(-1)
    x_test = torch.Tensor(test['x_test']).unsqueeze(1)
    y_test = torch.Tensor(test['y_test']).view(-1)

    x_train, y_train, x_valid, y_valid = split_train_valid_set(x_train, y_train, ratio=ratio)
    dev = torch.device('cpu')

    x_train = x_train[:, :, :, 124:562].to(dev)
    y_train = y_train.long().to(dev)
    x_valid = x_valid[:, :, :, 124:562].to(dev)
    y_valid = y_valid.long().to(dev)
    x_test = x_test[:, :, :, 124:562].to(dev)
    y_test = y_test.long().to(dev)
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    print('x_valid.shape: ', x_valid.shape)
    print('y_valid.shape: ', y_valid.shape)
    print('x_test.shape: ', x_test.shape)
    print('y_test.shape: ', y_test.shape)
    train_dataset = Data.TensorDataset(x_train, y_train)
    valid_dataset = Data.TensorDataset(x_valid, y_valid)
    test_dataset = Data.TensorDataset(x_test, y_test)

    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = bs,
        shuffle = True,
        num_workers = os.cpu_count(),  # Use all available CPU cores
        pin_memory=True
    )
    validloader = Data.DataLoader(
        dataset = valid_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = os.cpu_count(),  # Use all available CPU cores
        pin_memory=True
    )
    testloader = Data.DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = os.cpu_count(),  # Use all available CPU cores
        pin_memory=True
    )

    return trainloader, validloader, testloader

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
        # Ensure x is 2D (batch_size, n_qubits)
        batch_size = x.shape[0]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        output = []
        for i in range(batch_size):
            result = self.q_layer(x[i])
            output.append(result)

        return torch.stack(output)


class QuantumEEGNet(nn.Module):
    def __init__(self, F1=16, D=2, F2=16, dropout_rate=0.25, num_classes=4, n_qubits=9, n_layers=4):
        super(QuantumEEGNet, self).__init__()

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Layer 1
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, affine=False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(F1, F1 * D, (2, 32), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, affine=False)
        self.pooling2 = nn.MaxPool2d((2, 4))
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (8, 4), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F1 * D, affine=False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # Quantum layer
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)
        
        # Fully connected layer
        self.fc1 = nn.Linear(F1 * D * n_qubits, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        
        x = self.padding1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout(x)

        x = self.padding2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling3(x)
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

def save_metrics(metrics, output_dir, subject):
    os.makedirs(output_dir, exist_ok=True)
    for key, values in metrics.items():
        np.savetxt(os.path.join(output_dir, str(subject)+f"{key}.txt"), values, fmt="%.4f")

    epochs = range(1, len(metrics['train_loss']) + 1)
    plt.figure()
    plt.plot(epochs, metrics['train_loss'], 'r', label='Training loss')
    plt.plot(epochs, metrics['valid_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, str(subject)+'_loss.png'))

    plt.figure()
    plt.plot(epochs, metrics['train_accuracy'], 'r', label='Training accuracy')
    plt.plot(epochs, metrics['valid_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, str(subject)+'_accuracy.png'))

    plt.figure()
    test_epochs = range(5, len(metrics['test_accuracy']) * 5 + 1, 5)
    plt.plot(test_epochs, metrics['test_accuracy'], 'g', label='Test accuracy')
    plt.title('Test accuracy over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, str(subject)+'_test_accuracy.png'))


def train(model, device, train_loader, optimizer, criterion, epoch, metrics):
    model.train()
    train_loss = 0
    correct = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        progress_bar.set_postfix(loss=loss.item())

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100. * correct / len(train_loader.dataset)
    metrics['train_loss'].append(train_loss)
    metrics['train_accuracy'].append(train_accuracy)

    print(f'Epoch {epoch} Training: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({train_accuracy:.0f}%)')

def validate(model, device, valid_loader, criterion, metrics):
    model.eval()
    valid_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            valid_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)
    accuracy = 100. * correct / len(valid_loader.dataset)
    metrics['valid_loss'].append(valid_loss)
    metrics['valid_accuracy'].append(accuracy)

    print(f'Validation: Average loss: {valid_loss:.4f}, Accuracy: {correct}/{len(valid_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)\n')
    return test_accuracy

def main():
    parser = argparse.ArgumentParser(description='Quantum EEGNet Training')
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--subject', type=int, default=5, help='subject number')
    parser.add_argument('--data-path', type=str, default='/home/aidan/data/matt_data/data/BCICIV_2a_mat', help='path to data')
    parser.add_argument('--ratio', type=int, default=5, help='ratio for validation set split (default: 5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--output-dir', type=str, default='/home/aidan/qeegnet/output_qeegnet_9q4l', help='directory to save metrics and model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner to find the best algorithm for your hardware

    # load data
    train_loader, valid_loader, test_loader = getAllDataloader(args.subject, args.ratio, args.data_path, args.batch_size)

    model = QuantumEEGNet(num_classes=4).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        'train_loss': [],
        'valid_loss': [],
        'train_accuracy': [],
        'valid_accuracy': [],
        'test_accuracy': []
    }

    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch, metrics)
        validate(model, device, valid_loader, criterion, metrics)
        
        if epoch % 5 == 0:
            test_accuracy = test(model, device, test_loader, criterion)
            metrics['test_accuracy'].append(test_accuracy)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(model.state_dict(), os.path.join(args.output_dir, "sub"+str(args.subject)+"_best_model.pth"))

    save_metrics(metrics, args.output_dir, args.subject)
    # load best model
    model.load_state_dict(torch.load(os.path.join(args.output_dir, "sub"+str(args.subject)+"_best_model.pth")))
    final_test_accuracy = test(model, device, test_loader, criterion)
    metrics['test_accuracy'].append(final_test_accuracy)

    print(f'Best test accuracy: {best_accuracy:.4f}')
    save_metrics(metrics, args.output_dir, args.subject)


if __name__ == '__main__':
    main()

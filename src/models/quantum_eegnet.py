import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantum_layer import QuantumLayer


class QuantumEEGNet(nn.Module):
    """
    QuantumEEGNet: Hybrid quantum-classical neural network for EEG signal classification.
    
    This model combines the classical EEGNet architecture with quantum layers to enhance
    feature extraction and classification performance for EEG signals.
    
    Args:
        F1 (int): Number of temporal filters (default: 8)
        D (int): Depth multiplier (default: 2)
        F2 (int): Number of pointwise filters (default: 16)
        dropout_rate (float): Dropout probability (default: 0.25)
        num_classes (int): Number of output classes (default: 2)
        n_qubits (int): Number of qubits in quantum layer (default: 4)
        n_layers (int): Number of layers in quantum circuit (default: 2)
        input_channels (int): Number of input EEG channels (default: 2)
        input_length (int): Length of input time series (default: 128)
    """
    
    def __init__(self, F1=8, D=2, F2=16, dropout_rate=0.25, num_classes=2, 
                 n_qubits=4, n_layers=2, input_channels=2, input_length=128):
        super(QuantumEEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_channels = input_channels
        self.input_length = input_length

        # First temporal convolution layer
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution layer
        self.conv2 = nn.Conv2d(F1, F1 * D, (input_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)

        # Separable temporal convolution layers
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F1 * D)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(F2)

        # Quantum layer
        self.quantum_layer = QuantumLayer(n_qubits, n_layers)

        # Fully connected classification layer
        self.fc1 = nn.Linear(F2 * n_qubits, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the QuantumEEGNet.
        
        Args:
            x (torch.Tensor): Input EEG data of shape (batch_size, 1, channels, time_points)
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        # First temporal convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        
        # Depthwise spatial convolution
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout(x)
        
        # Separable temporal convolution
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))
        x = self.dropout(x)
        
        # Reshape for quantum layer processing
        x = x.view(x.size(0), x.size(1), -1)
        
        # Process each channel through quantum layer separately
        quantum_outs = []
        for i in range(x.size(1)):
            quantum_out = self.quantum_layer(x[:, i, :])
            quantum_outs.append(quantum_out)
        
        # Concatenate quantum outputs from all channels
        x = torch.cat(quantum_outs, dim=1)
        
        # Final classification
        x = self.fc1(x)
        
        return x
    
    def get_model_info(self):
        """
        Get information about the model architecture.
        
        Returns:
            dict: Model information including layer configurations
        """
        return {
            "model_type": "QuantumEEGNet",
            "F1": self.F1,
            "D": self.D,
            "F2": self.F2,
            "dropout_rate": self.dropout_rate,
            "num_classes": self.num_classes,
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "input_channels": self.input_channels,
            "input_length": self.input_length,
            "quantum_circuit_info": self.quantum_layer.get_circuit_info()
        }
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            int: Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClassicalEEGNet(nn.Module):
    """
    Classical EEGNet for comparison with QuantumEEGNet.
    
    This is the original EEGNet architecture without quantum layers for baseline comparison.
    """
    
    def __init__(self, F1=8, D=2, F2=16, dropout_rate=0.25, num_classes=2, 
                 input_channels=2, input_length=128):
        super(ClassicalEEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.input_length = input_length

        # First temporal convolution layer
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Depthwise spatial convolution layer
        self.conv2 = nn.Conv2d(F1, F1 * D, (input_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)

        # Separable temporal convolution layers
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F1 * D)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(F2)

        # Fully connected classification layer
        self.fc1 = nn.Linear(F2, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the Classical EEGNet.
        
        Args:
            x (torch.Tensor): Input EEG data of shape (batch_size, 1, channels, time_points)
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        # First temporal convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        
        # Depthwise spatial convolution
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout(x)
        
        # Separable temporal convolution
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))
        x = self.dropout(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Final classification
        x = self.fc1(x)
        
        return x

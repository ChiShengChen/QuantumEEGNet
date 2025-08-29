# QuantumEEGNet

## QEEGNet: Quantum Machine Learning for Enhanced Electroencephalography Encoding (IEEE SiPS 2024)
[![arXiv](https://img.shields.io/badge/arXiv-2407.19214-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2407.19214)  
[IEEE SiPS 2024](https://ieeexplore.ieee.org/abstract/document/10768221/)

![image](https://github.com/user-attachments/assets/76786943-880b-4134-a34e-d828f89a00f1)

## Overview

**QEEGNet** is a hybrid neural network integrating **quantum computing** and the classical **EEGNet** architecture to enhance the encoding and analysis of EEG signals. By incorporating **variational quantum circuits (VQC)**, QEEGNet captures more intricate patterns within EEG data, offering improved performance and robustness compared to traditional models.

This repository contains the implementation and experimental results for **QEEGNet**, evaluated on the **BCI Competition IV 2a** dataset.

## Key Features

- **Hybrid Architecture**: Combines the EEGNet convolutional framework with quantum encoding layers for advanced feature extraction.
- **Quantum Layer Integration**: Leverages the unique properties of quantum mechanics, such as superposition and entanglement, for richer data representation.
- **Improved Robustness**: Demonstrates enhanced accuracy and resilience to noise in EEG signal classification tasks.
- **Generalizability**: Consistently outperforms EEGNet across most subjects in benchmark datasets.
- **Modular Design**: Clean, maintainable codebase with separated concerns for models, data, training, and configuration.

## Project Structure

```
QuantumEEGNet/
├── src/                          # Source code
│   ├── models/                   # Model implementations
│   │   ├── quantum_layer.py     # Quantum layer implementation
│   │   ├── quantum_eegnet.py    # QuantumEEGNet and ClassicalEEGNet
│   │   └── __init__.py
│   ├── data/                     # Data handling
│   │   ├── dataset.py           # BCI and synthetic datasets
│   │   └── __init__.py
│   ├── training/                 # Training utilities
│   │   ├── trainer.py           # Training manager
│   │   └── __init__.py
│   ├── configs/                  # Configuration management
│   │   ├── config.py            # Configuration classes
│   │   └── __init__.py
│   ├── utils/                    # Utility functions
│   │   └── __init__.py
│   └── __init__.py
├── examples/                     # Example scripts
│   ├── train_synthetic.py       # Synthetic data training example
│   └── train_bci.py             # BCI data training example
├── experiments/                  # Experiment results
├── data/                         # Data storage
├── logs/                         # Training logs
├── train_quantum_eegnet.py      # Main training script
├── requirements.txt              # Dependencies
├── setup.py                      # Installation script
└── README.md
```

## Architecture

QEEGNet consists of:
1. **Classical EEGNet Layers**: Initial convolutional layers process EEG signals to extract temporal and spatial features.
2. **Quantum Encoding Layer**: Encodes classical features into quantum states using a parameterized quantum circuit.
3. **Fully Connected Layers**: Converts quantum outputs into final classifications.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/QuantumEEGNet.git
   cd QuantumEEGNet
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package** (optional):
   ```bash
   pip install -e .
   ```

## Quick Start

### 1. Test the Project Structure

First, verify that everything is set up correctly:

```bash
python test_structure.py
```

This will test all components and confirm the installation is working.

### 2. Training on Synthetic Data

For quick testing and development, you can train on synthetic data:

```bash
# Basic training with default parameters
python train_quantum_eegnet.py --dataset synthetic --epochs 20

# Customize quantum parameters
python train_quantum_eegnet.py --dataset synthetic --epochs 30 --n_qubits 6 --n_layers 3

# Compare with classical EEGNet
python train_quantum_eegnet.py --model classical --dataset synthetic --epochs 20
```

Or use the example script for a complete comparison:

```bash
python examples/train_synthetic.py
```

### 3. Training on BCI Dataset

To train on the BCI Competition IV 2a dataset:

1. **Download the dataset** and place it in the `data/` directory:
   ```
   data/
   ├── BCIC_S01_T.mat
   ├── BCIC_S01_E.mat
   ├── BCIC_S02_T.mat
   ├── BCIC_S02_E.mat
   ├── BCIC_S03_T.mat
   ├── BCIC_S03_E.mat
   └── ...
   ```

2. **Train the model**:
   ```bash
   # Train on subject 1
   python train_quantum_eegnet.py --dataset bci --subject 1 --epochs 100
   
   # Train on different subjects
   python train_quantum_eegnet.py --dataset bci --subject 2 --epochs 100
   python train_quantum_eegnet.py --dataset bci --subject 3 --epochs 100
   
   # Customize training parameters
   python train_quantum_eegnet.py --dataset bci --subject 1 --epochs 150 --batch_size 32 --learning_rate 5e-4
   ```

Or use the example script:

```bash
python examples/train_bci.py
```

## Usage

### Command Line Interface

The main training script supports comprehensive options:

```bash
python train_quantum_eegnet.py [OPTIONS]

Model Options:
  --model {quantum,classical}    Model type (default: quantum)
  --n_qubits INT                 Number of qubits in quantum layer (default: 4)
  --n_layers INT                 Number of quantum circuit layers (default: 2)

Dataset Options:
  --dataset {bci,synthetic}      Dataset type (default: synthetic)
  --subject INT                   BCI subject number (1-9, default: 1)
  --data_path STR                Path to BCI dataset (default: data)
  --num_samples INT              Number of synthetic samples (default: 1000)
  --num_classes INT              Number of classes for synthetic data (default: 2)

Training Options:
  --epochs INT                    Number of training epochs (default: 50)
  --batch_size INT               Batch size (default: 64)
  --learning_rate FLOAT          Learning rate (default: 1e-3)
  --patience INT                 Early stopping patience (default: 10)
  --device {auto,cpu,cuda}       Device to use (default: auto)
```

### Common Usage Examples

```bash
# Quick test with synthetic data
python train_quantum_eegnet.py --dataset synthetic --epochs 10

# Full training on BCI dataset
python train_quantum_eegnet.py --dataset bci --subject 1 --epochs 100 --batch_size 32

# Experiment with quantum parameters
python train_quantum_eegnet.py --dataset synthetic --n_qubits 8 --n_layers 4 --epochs 50

# Compare quantum vs classical
python train_quantum_eegnet.py --model quantum --dataset synthetic --epochs 30
python train_quantum_eegnet.py --model classical --dataset synthetic --epochs 30

# GPU training (if available)
python train_quantum_eegnet.py --dataset bci --subject 1 --device cuda --epochs 100
```

### Programmatic Usage

#### Basic Training

```python
from src import QuantumEEGNet, BCIDataset, Trainer, get_quantum_eegnet_config
import torch

# Get configuration
config = get_quantum_eegnet_config()

# Load dataset
dataset = BCIDataset(data_path="data", subject=1)
train_loader, valid_loader, test_loader = dataset.get_dataloaders()

# Create model
model = QuantumEEGNet(**config.model_config)

# Create trainer
trainer = Trainer(
    model=model,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    config=config.model_config
)

# Train model
results = trainer.train()
```

#### Custom Configuration

```python
from src import QuantumEEGNet, SyntheticDataset, Trainer, Config

# Create custom configuration
config = Config({
    'model_config': {
        'model_name': 'quantum_eegnet',
        'n_qubits': 6,
        'n_layers': 3,
        'num_classes': 2,
        'F1': 16,
        'F2': 32
    },
    'training_config': {
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'optimizer': 'adamw'
    }
})

# Load synthetic dataset
dataset = SyntheticDataset(
    num_samples=1000,
    num_channels=2,
    time_points=128,
    num_classes=2
)

train_loader, valid_loader, test_loader = dataset.get_dataloaders(
    batch_size=config.training_config['batch_size']
)

# Create and train model
model = QuantumEEGNet(**config.model_config)
trainer = Trainer(
    model=model,
    device=torch.device('cpu'),
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    config=config.model_config
)

results = trainer.train()
```

#### Model Comparison

```python
from src import QuantumEEGNet, ClassicalEEGNet, SyntheticDataset, Trainer

# Create datasets
dataset = SyntheticDataset(num_samples=500, num_classes=2)
train_loader, valid_loader, test_loader = dataset.get_dataloaders(batch_size=32)

# Train QuantumEEGNet
quantum_model = QuantumEEGNet(num_classes=2, n_qubits=4, n_layers=2)
quantum_trainer = Trainer(
    model=quantum_model,
    device=torch.device('cpu'),
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    config={'model_name': 'quantum_eegnet'}
)
quantum_results = quantum_trainer.train()

# Train ClassicalEEGNet
classical_model = ClassicalEEGNet(num_classes=2)
classical_trainer = Trainer(
    model=classical_model,
    device=torch.device('cpu'),
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    config={'model_name': 'classical_eegnet'}
)
classical_results = classical_trainer.train()

# Compare results
print(f"QuantumEEGNet Test Accuracy: {quantum_results['test_accuracy']:.2f}%")
print(f"ClassicalEEGNet Test Accuracy: {classical_results['test_accuracy']:.2f}%")
print(f"Improvement: {quantum_results['test_accuracy'] - classical_results['test_accuracy']:.2f}%")
```

### Working with Results

Training results are automatically saved to the `experiments/` directory:

```python
# Load trained model
from src import load_trained_model, QuantumEEGNet

model, config = load_trained_model(
    'experiments/quantum_eegnet_20241201_143022/model.pth',
    QuantumEEGNet,
    torch.device('cpu')
)

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

## Dataset

### BCI Competition IV 2a Dataset

The **BCI Competition IV 2a** dataset was used for evaluation, featuring EEG signals from motor-imagery tasks.  
- **Subjects**: 9  
- **Classes**: Right hand, left hand, feet, tongue  
- **Preprocessing**: Downsampled to 128 Hz, band-pass filtered (4-38 Hz).  

For more details, refer to the [dataset documentation](https://www.bbci.de/competition/iv/).  
Or you can use the organized data format in https://github.com/CECNL/MAtt repo.

### Synthetic Dataset

For rapid testing and development, we provide a **synthetic EEG dataset** that simulates real EEG signals without requiring the actual BCI dataset.

#### Features

- **Artificial EEG-like signals** with class-dependent frequency patterns
- **Configurable parameters**: number of samples, channels, time points, and classes
- **Realistic structure**: follows the same format as real EEG data
- **Fast generation**: no need to download large datasets

#### Generation Process

The synthetic data is generated using the following approach:

1. **Base Data**: Random noise serves as the foundation
   ```python
   x = torch.randn(samples, 1, channels, time_points)
   ```

2. **Class Labels**: Random assignment of class labels
   ```python
   y = torch.randint(0, num_classes, (samples,))
   ```

3. **Class-dependent Patterns**: Each class gets a unique frequency signature
   ```python
   for class_i in range(num_classes):
       freq = 10 + class_i * 5  # Class 0: 10Hz, Class 1: 15Hz, etc.
       pattern = torch.sin(freq * time_axis)
       x[class_mask] += 0.1 * pattern
   ```

#### Data Characteristics

- **Shape**: `(batch_size, 1, channels, time_points)`
  - `batch_size`: Number of samples in batch
  - `1`: Single channel dimension (EEG format)
  - `channels`: Number of electrodes (default: 2)
  - `time_points`: Number of time points (default: 128)

- **Class Patterns**:
  - Class 0: 10Hz sinusoidal pattern + noise
  - Class 1: 15Hz sinusoidal pattern + noise
  - Class 2: 20Hz sinusoidal pattern + noise
  - And so on...

#### Usage

```python
from src import SyntheticDataset

# Create synthetic dataset
dataset = SyntheticDataset(
    num_samples=1000,      # Total samples
    num_channels=2,        # Number of EEG channels
    time_points=128,       # Time series length
    num_classes=2,         # Number of classes
    validation_ratio=0.2   # Validation split ratio
)

# Get data loaders
train_loader, valid_loader, test_loader = dataset.get_dataloaders(batch_size=64)

# Get dataset information
info = dataset.get_data_info()
print(f"Train samples: {info['train_samples']}")
print(f"Valid samples: {info['valid_samples']}")
print(f"Test samples: {info['test_samples']}")
print(f"Class distribution: {info['class_distribution']}")
```

#### When to Use Synthetic Data

**✅ Recommended for:**
- Code development and debugging
- Model architecture testing
- Hyperparameter optimization
- Learning and tutorials
- Quick prototyping

**❌ Not suitable for:**
- Final performance evaluation
- Research publications
- Production deployment

#### Advantages

1. **Fast Testing**: No need to download large datasets
2. **Controllable**: Can control data characteristics and distribution
3. **Reproducible**: Consistent results across runs
4. **Development-friendly**: Ideal for code debugging and parameter testing

#### Limitations

1. **Simplified**: Much simpler than real EEG signals
2. **Artificial**: Cannot fully simulate real brain activity patterns
3. **Performance**: Training results don't represent real-world performance

#### Switching to Real Data

When ready for serious evaluation, switch to the BCI dataset:

```bash
# Use BCI dataset instead of synthetic
python train_quantum_eegnet.py --dataset bci --subject 1 --epochs 100
```

## Configuration

The project uses a centralized configuration system. You can:

1. **Use predefined configurations**:
   ```python
   from src.configs import get_quantum_eegnet_config, get_synthetic_data_config
   
   config = get_quantum_eegnet_config()
   ```

2. **Create custom configurations**:
   ```python
   from src.configs import Config
   
   config = Config({
       'model_config': {
           'n_qubits': 6,
           'n_layers': 3,
           'num_classes': 4
       },
       'training_config': {
           'epochs': 150,
           'learning_rate': 5e-4
       }
   })
   ```

3. **Save and load configurations**:
   ```python
   config.save_config('my_config.json')
   loaded_config = Config.load_config('my_config.json')
   ```

## Results

Training results are automatically saved to the `experiments/` directory, including:
- Model checkpoints
- Training curves
- Configuration files
- Performance metrics

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{chen2024qeegnet,
  title={Qeegnet: Quantum machine learning for enhanced electroencephalography encoding},
  author={Chen, Chi-Sheng and Chen, Samuel Yen-Chi and Tsai, Aidan Hung-Wen and Wei, Chun-Shu},
  journal={arXiv preprint arXiv:2407.19214},
  year={2024}
}
```
```bibtex
@INPROCEEDINGS{10768221,
  author={Chen, Chi-Sheng and Chen, Samuel Yen-Chi and Tsai, Aidan Hung-Wen and Wei, Chun-Shu},
  booktitle={2024 IEEE Workshop on Signal Processing Systems (SiPS)}, 
  title={QEEGNet: Quantum Machine Learning for Enhanced Electroencephalography Encoding}, 
  year={2024},
  volume={},
  number={},
  pages={153-158},
  keywords={Quantum computing;Neuroscience;Computational modeling;Computer architecture;Brain modeling;Electroencephalography;Robustness;Encoding;Space exploration;Biological neural networks;electroencephalography;EEG classification;quantum machine learning;quantum algorithm;deep learning;brain-computer interface},
  doi={10.1109/SiPS62058.2024.00035}}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

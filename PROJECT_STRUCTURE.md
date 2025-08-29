# QuantumEEGNet Project Structure

## Overview

This document describes the reorganized project structure for QuantumEEGNet, which has been refactored to be more maintainable, modular, and engineering-friendly.

## Directory Structure

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
├── test_structure.py            # Structure verification script
├── requirements.txt              # Dependencies
├── setup.py                      # Installation script
├── README.md                     # Main documentation
└── PROJECT_STRUCTURE.md          # This file
```

## Key Improvements

### 1. Modular Architecture

**Before**: All code was in single files with mixed concerns
**After**: Separated into logical modules:
- `models/`: Neural network architectures
- `data/`: Dataset loading and preprocessing
- `training/`: Training utilities and management
- `configs/`: Configuration management
- `utils/`: Utility functions

### 2. Configuration Management

**Before**: Hard-coded parameters scattered throughout code
**After**: Centralized configuration system with:
- Predefined configurations for different use cases
- Easy customization and parameter management
- Configuration saving/loading capabilities

### 3. Data Handling

**Before**: Data loading code mixed with training logic
**After**: Dedicated dataset classes:
- `BCIDataset`: For BCI Competition IV 2a dataset
- `SyntheticDataset`: For synthetic data generation
- Consistent interface for both datasets

### 4. Training Infrastructure

**Before**: Basic training loops with limited features
**After**: Comprehensive training system with:
- Progress tracking and visualization
- Early stopping
- Model checkpointing
- Result saving and analysis
- Support for multiple optimizers

### 5. Code Organization

**Before**: Monolithic scripts
**After**: Clean separation of concerns:
- Model definitions in dedicated files
- Training logic in trainer class
- Configuration management separate from implementation
- Example scripts for different use cases

## File Descriptions

### Core Modules

#### `src/models/`
- `quantum_layer.py`: Implementation of quantum variational circuits using PennyLane
- `quantum_eegnet.py`: QuantumEEGNet and ClassicalEEGNet model architectures

#### `src/data/`
- `dataset.py`: Dataset classes for BCI and synthetic data with preprocessing

#### `src/training/`
- `trainer.py`: Training manager with comprehensive training utilities

#### `src/configs/`
- `config.py`: Configuration management system with predefined configs

### Scripts

#### `train_quantum_eegnet.py`
Main training script with command-line interface supporting:
- Model selection (quantum/classical)
- Dataset selection (BCI/synthetic)
- Hyperparameter configuration
- Device selection

#### `examples/`
- `train_synthetic.py`: Example for training on synthetic data
- `train_bci.py`: Example for training on BCI dataset

#### `test_structure.py`
Verification script to test the new project structure

## Usage Examples

### Quick Start with Synthetic Data
```bash
python train_quantum_eegnet.py --dataset synthetic --epochs 20
```

### Training on BCI Dataset
```bash
python train_quantum_eegnet.py --dataset bci --subject 1 --epochs 100
```

### Using Example Scripts
```bash
python examples/train_synthetic.py
python examples/train_bci.py
```

### Testing Structure
```bash
python test_structure.py
```

## Migration from Old Structure

The old training files are preserved for reference:
- `qeegnet.py`: Original model implementation
- `train.py`: Original training script
- `train_bcic2a.py`: Original BCI training script
- `train_bcic2a_*.py`: Various training variants

These files can be used as reference but the new structure is recommended for:
- Better maintainability
- Easier experimentation
- Cleaner code organization
- More robust training pipeline

## Benefits of New Structure

1. **Maintainability**: Clear separation of concerns makes code easier to understand and modify
2. **Reusability**: Modular components can be easily reused in different contexts
3. **Testability**: Individual components can be tested in isolation
4. **Extensibility**: Easy to add new models, datasets, or training features
5. **Documentation**: Better organized code is easier to document
6. **Collaboration**: Multiple developers can work on different modules simultaneously

## Future Enhancements

The new structure enables easy addition of:
- New quantum circuit architectures
- Additional datasets
- Advanced training techniques
- Model evaluation tools
- Visualization utilities
- Hyperparameter optimization
- Distributed training support

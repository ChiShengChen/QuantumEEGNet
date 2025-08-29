#!/usr/bin/env python3
"""
Main training script for QuantumEEGNet.

This script provides a clean interface for training QuantumEEGNet models
with different configurations and datasets.
"""

import torch
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src import (
    QuantumEEGNet, 
    ClassicalEEGNet,
    BCIDataset, 
    SyntheticDataset,
    Trainer,
    get_quantum_eegnet_config,
    get_classical_eegnet_config,
    get_synthetic_data_config
)


def get_device(device_config: str) -> torch.device:
    """Get the appropriate device for training."""
    if device_config == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_config)


def main():
    parser = argparse.ArgumentParser(description='Train QuantumEEGNet or Classical EEGNet')
    
    # Model selection
    parser.add_argument('--model', type=str, default='quantum', 
                       choices=['quantum', 'classical'],
                       help='Model type: quantum or classical')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['bci', 'synthetic'],
                       help='Dataset type: bci or synthetic')
    
    # BCI dataset specific arguments
    parser.add_argument('--subject', type=int, default=1,
                       help='BCI subject number (1-9)')
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to BCI dataset')
    
    # Synthetic data specific arguments
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of synthetic samples')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes for synthetic data')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Quantum specific arguments
    parser.add_argument('--n_qubits', type=int, default=4,
                       help='Number of qubits in quantum layer')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of layers in quantum circuit')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Get configuration
    if args.model == 'quantum':
        if args.dataset == 'synthetic':
            config = get_synthetic_data_config()
        else:
            config = get_quantum_eegnet_config()
    else:
        config = get_classical_eegnet_config()
    
    # Update configuration with command line arguments
    config.model_config.update({
        'num_classes': args.num_classes,
        'n_qubits': args.n_qubits,
        'n_layers': args.n_layers
    })
    
    config.training_config.update({
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience
    })
    
    config.data_config.update({
        'dataset_type': args.dataset,
        'data_path': args.data_path,
        'subject': args.subject,
        'num_samples': args.num_samples,
        'num_classes': args.num_classes
    })
    
    config.experiment_config.update({
        'device': args.device
    })
    
    print("Configuration:")
    print(config)
    
    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == 'bci':
        dataset = BCIDataset(
            data_path=args.data_path,
            subject=args.subject,
            validation_ratio=config.data_config['validation_ratio']
        )
    else:
        dataset = SyntheticDataset(
            num_samples=args.num_samples,
            num_channels=config.model_config['input_channels'],
            time_points=config.model_config['input_length'],
            num_classes=args.num_classes,
            validation_ratio=config.data_config['validation_ratio']
        )
    
    # Get dataloaders
    train_loader, valid_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.training_config['batch_size'],
        num_workers=config.training_config['num_workers']
    )
    
    print(f"Dataset info: {dataset.get_data_info()}")
    
    # Create model
    print(f"\nCreating {args.model} model...")
    if args.model == 'quantum':
        model = QuantumEEGNet(**config.model_config)
    else:
        model = ClassicalEEGNet(**config.model_config)
    
    print(f"Model info: {model.get_model_info()}")
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config=config.model_config
    )
    
    # Train model
    print("\nStarting training...")
    results = trainer.train()
    
    print("\nTraining completed successfully!")
    print(f"Best validation accuracy: {results['best_valid_accuracy']:.2f}%")
    print(f"Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Results saved to: {trainer.experiment_dir}")


if __name__ == "__main__":
    main()

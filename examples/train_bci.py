#!/usr/bin/env python3
"""
Example script for training QuantumEEGNet on BCI Competition IV 2a dataset.

This script demonstrates how to train the QuantumEEGNet model on the real
BCI Competition IV 2a dataset for motor imagery classification.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src import (
    QuantumEEGNet,
    ClassicalEEGNet,
    BCIDataset,
    Trainer,
    get_quantum_eegnet_config
)
import torch


def main():
    """Train QuantumEEGNet on BCI dataset."""
    
    # Configuration
    data_path = "data"  # Path to BCI dataset
    subject = 1  # Subject number (1-9)
    
    # Get configuration
    config = get_quantum_eegnet_config()
    
    # Update configuration for BCI dataset
    config.model_config.update({
        'num_classes': 4,  # BCI has 4 classes
        'n_qubits': 4,
        'n_layers': 2
    })
    
    config.training_config.update({
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'patience': 15
    })
    
    config.data_config.update({
        'data_path': data_path,
        'subject': subject,
        'validation_ratio': 0.2
    })
    
    print("Configuration:")
    print(config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load BCI dataset
    print(f"\nLoading BCI dataset for subject {subject}...")
    try:
        dataset = BCIDataset(
            data_path=data_path,
            subject=subject,
            validation_ratio=config.data_config['validation_ratio']
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the BCI dataset files are available in the data directory.")
        print("Expected files:")
        print(f"  - {data_path}/BCIC_S{subject:02d}_T.mat (training data)")
        print(f"  - {data_path}/BCIC_S{subject:02d}_E.mat (test data)")
        return
    
    # Get dataloaders
    train_loader, valid_loader, test_loader = dataset.get_dataloaders(
        batch_size=config.training_config['batch_size']
    )
    
    print(f"Dataset info: {dataset.get_data_info()}")
    
    # Create QuantumEEGNet model
    print("\nCreating QuantumEEGNet model...")
    model = QuantumEEGNet(**config.model_config)
    
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
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_valid_accuracy']:.2f}%")
    print(f"Test accuracy: {results['test_accuracy']:.2f}%")
    print(f"Results saved to: {trainer.experiment_dir}")
    
    # Compare with classical EEGNet
    print("\n" + "="*50)
    print("Comparing with Classical EEGNet...")
    
    # Create Classical EEGNet model
    classical_config = config.model_config.copy()
    classical_config['model_name'] = 'classical_eegnet'
    
    classical_model = ClassicalEEGNet(**classical_config)
    
    print(f"Classical model parameters: {classical_model.count_parameters():,}")
    
    # Create trainer for classical model
    classical_trainer = Trainer(
        model=classical_model,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        config=classical_config
    )
    
    # Train classical model
    print("\nTraining Classical EEGNet...")
    classical_results = classical_trainer.train()
    
    print("\nComparison Results:")
    print(f"QuantumEEGNet - Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Classical EEGNet - Test Accuracy: {classical_results['test_accuracy']:.2f}%")
    print(f"Improvement: {results['test_accuracy'] - classical_results['test_accuracy']:.2f}%")


if __name__ == "__main__":
    main()

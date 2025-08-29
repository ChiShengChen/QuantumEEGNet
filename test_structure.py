#!/usr/bin/env python3
"""
Test script to verify the new project structure works correctly.
"""

import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from src import (
            QuantumLayer,
            QuantumEEGNet,
            ClassicalEEGNet,
            BCIDataset,
            SyntheticDataset,
            Trainer,
            get_quantum_eegnet_config,
            get_classical_eegnet_config,
            get_synthetic_data_config
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_models():
    """Test model creation and forward pass."""
    print("\nTesting models...")
    
    try:
        from src import QuantumEEGNet, ClassicalEEGNet
        
        # Test QuantumEEGNet
        quantum_model = QuantumEEGNet(
            num_classes=2,
            n_qubits=4,
            n_layers=2,
            input_channels=2,
            input_length=128
        )
        
        # Test forward pass
        x = torch.randn(2, 1, 2, 128)  # batch_size=2, channels=1, electrodes=2, time_points=128
        output = quantum_model(x)
        print(f"✓ QuantumEEGNet forward pass successful, output shape: {output.shape}")
        
        # Test ClassicalEEGNet
        classical_model = ClassicalEEGNet(
            num_classes=2,
            input_channels=2,
            input_length=128
        )
        
        output = classical_model(x)
        print(f"✓ ClassicalEEGNet forward pass successful, output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_datasets():
    """Test dataset creation."""
    print("\nTesting datasets...")
    
    try:
        from src import SyntheticDataset
        
        # Test synthetic dataset
        dataset = SyntheticDataset(
            num_samples=100,
            num_channels=2,
            time_points=128,
            num_classes=2,
            validation_ratio=0.2
        )
        
        train_loader, valid_loader, test_loader = dataset.get_dataloaders(batch_size=16)
        
        # Test data loading
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx == 0:
                print(f"✓ Synthetic dataset loaded successfully")
                print(f"  Data shape: {data.shape}")
                print(f"  Target shape: {target.shape}")
                break
        
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def test_configs():
    """Test configuration system."""
    print("\nTesting configurations...")
    
    try:
        from src import get_quantum_eegnet_config, get_synthetic_data_config
        
        # Test quantum config
        quantum_config = get_quantum_eegnet_config()
        print(f"✓ Quantum config loaded: {quantum_config.model_config['model_name']}")
        
        # Test synthetic config
        synthetic_config = get_synthetic_data_config()
        print(f"✓ Synthetic config loaded: {synthetic_config.data_config['dataset_type']}")
        
        return True
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def test_trainer():
    """Test trainer creation."""
    print("\nTesting trainer...")
    
    try:
        from src import QuantumEEGNet, SyntheticDataset, Trainer, get_synthetic_data_config
        
        # Get config
        config = get_synthetic_data_config()
        
        # Create dataset
        dataset = SyntheticDataset(
            num_samples=50,  # Small dataset for testing
            num_channels=2,
            time_points=128,
            num_classes=2,
            validation_ratio=0.2
        )
        
        train_loader, valid_loader, test_loader = dataset.get_dataloaders(batch_size=8)
        
        # Create model - filter out model_name parameter
        model_params = {k: v for k, v in config.model_config.items() 
                       if k not in ['model_name']}
        model = QuantumEEGNet(**model_params)
        
        # Create trainer
        device = torch.device('cpu')  # Use CPU for testing
        trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            config=config.model_config
        )
        
        print("✓ Trainer created successfully")
        return True
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        return False

def test_command_line_interface():
    """Test command line interface simulation."""
    print("\nTesting command line interface...")
    
    try:
        from src import (
            QuantumEEGNet, 
            ClassicalEEGNet,
            SyntheticDataset, 
            Trainer, 
            get_synthetic_data_config
        )
        import torch
        
        # Simulate command line arguments
        args_model = 'quantum'
        args_dataset = 'synthetic'
        args_epochs = 5
        args_batch_size = 16
        args_learning_rate = 1e-3
        args_n_qubits = 4
        args_n_layers = 2
        args_num_classes = 2
        args_num_samples = 100
        args_device = 'cpu'
        
        # Get configuration
        config = get_synthetic_data_config()
        
        # Update configuration with command line arguments
        config.model_config.update({
            'num_classes': args_num_classes,
            'n_qubits': args_n_qubits,
            'n_layers': args_n_layers
        })
        
        config.training_config.update({
            'epochs': args_epochs,
            'batch_size': args_batch_size,
            'learning_rate': args_learning_rate
        })
        
        config.data_config.update({
            'dataset_type': args_dataset,
            'num_samples': args_num_samples,
            'num_classes': args_num_classes
        })
        
        # Load dataset
        dataset = SyntheticDataset(
            num_samples=args_num_samples,
            num_channels=config.model_config['input_channels'],
            time_points=config.model_config['input_length'],
            num_classes=args_num_classes,
            validation_ratio=config.data_config['validation_ratio']
        )
        
        train_loader, valid_loader, test_loader = dataset.get_dataloaders(
            batch_size=config.training_config['batch_size']
        )
        
        # Create model - filter out model_name parameter
        model_params = {k: v for k, v in config.model_config.items() 
                       if k not in ['model_name']}
        
        if args_model == 'quantum':
            model = QuantumEEGNet(**model_params)
        else:
            model = ClassicalEEGNet(**model_params)
        
        # Create trainer
        device = torch.device(args_device)
        trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            config=config.model_config
        )
        
        print("✓ Command line interface simulation successful")
        return True
    except Exception as e:
        print(f"✗ Command line interface test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing QuantumEEGNet project structure...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_models,
        test_datasets,
        test_configs,
        test_trainer,
        test_command_line_interface
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The project structure is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

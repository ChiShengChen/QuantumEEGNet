import os
from typing import Dict, Any
import json


class Config:
    """
    Configuration management for QuantumEEGNet experiments.
    
    This class provides a centralized way to manage all experiment configurations
    including model parameters, training settings, and data paths.
    """
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        """
        Initialize configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        if config_dict is None:
            config_dict = {}
        
        # Set default values
        self._set_defaults()
        
        # Update with provided values
        self._update_config(config_dict)
    
    def _set_defaults(self):
        """Set default configuration values."""
        self.model_config = {
            'model_name': 'quantum_eegnet',
            'F1': 8,
            'D': 2,
            'F2': 16,
            'dropout_rate': 0.25,
            'num_classes': 4,
            'n_qubits': 4,
            'n_layers': 2,
            'input_channels': 2,
            'input_length': 128
        }
        
        self.training_config = {
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'patience': 10,
            'num_workers': 0
        }
        
        self.data_config = {
            'dataset_type': 'bci',  # 'bci' or 'synthetic'
            'data_path': 'data',
            'subject': 1,
            'validation_ratio': 0.2,
            'num_samples': 1000,  # for synthetic data
            'time_points': 128
        }
        
        self.experiment_config = {
            'experiment_name': 'quantum_eegnet_experiment',
            'save_dir': 'experiments',
            'log_dir': 'logs',
            'device': 'auto'  # 'auto', 'cpu', 'cuda'
        }
    
    def _update_config(self, config_dict: Dict[str, Any]):
        """Update configuration with provided values."""
        for key, value in config_dict.items():
            if key == 'model_config':
                self.model_config.update(value)
            elif key == 'training_config':
                self.training_config.update(value)
            elif key == 'data_config':
                self.data_config.update(value)
            elif key == 'experiment_config':
                self.experiment_config.update(value)
            else:
                # Direct assignment for top-level keys
                setattr(self, key, value)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.model_config.copy()
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.training_config.copy()
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.data_config.copy()
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return self.experiment_config.copy()
    
    def save_config(self, filepath: str):
        """Save configuration to file."""
        config_dict = {
            'model_config': self.model_config,
            'training_config': self.training_config,
            'data_config': self.data_config,
            'experiment_config': self.experiment_config
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'Config':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_str = "Configuration:\n"
        config_str += f"  Model: {self.model_config}\n"
        config_str += f"  Training: {self.training_config}\n"
        config_str += f"  Data: {self.data_config}\n"
        config_str += f"  Experiment: {self.experiment_config}\n"
        return config_str


# Predefined configurations
def get_quantum_eegnet_config() -> Config:
    """Get default QuantumEEGNet configuration."""
    config_dict = {
        'model_config': {
            'model_name': 'quantum_eegnet',
            'F1': 8,
            'D': 2,
            'F2': 16,
            'dropout_rate': 0.25,
            'num_classes': 4,
            'n_qubits': 4,
            'n_layers': 2,
            'input_channels': 2,
            'input_length': 128
        },
        'training_config': {
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'patience': 10,
            'num_workers': 0
        },
        'data_config': {
            'dataset_type': 'bci',
            'data_path': 'data',
            'subject': 1,
            'validation_ratio': 0.2
        },
        'experiment_config': {
            'experiment_name': 'quantum_eegnet_experiment',
            'save_dir': 'experiments',
            'log_dir': 'logs',
            'device': 'auto'
        }
    }
    return Config(config_dict)


def get_classical_eegnet_config() -> Config:
    """Get default Classical EEGNet configuration."""
    config_dict = {
        'model_config': {
            'model_name': 'classical_eegnet',
            'F1': 8,
            'D': 2,
            'F2': 16,
            'dropout_rate': 0.25,
            'num_classes': 4,
            'input_channels': 2,
            'input_length': 128
        },
        'training_config': {
            'epochs': 100,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'patience': 10,
            'num_workers': 0
        },
        'data_config': {
            'dataset_type': 'bci',
            'data_path': 'data',
            'subject': 1,
            'validation_ratio': 0.2
        },
        'experiment_config': {
            'experiment_name': 'classical_eegnet_experiment',
            'save_dir': 'experiments',
            'log_dir': 'logs',
            'device': 'auto'
        }
    }
    return Config(config_dict)


def get_synthetic_data_config() -> Config:
    """Get configuration for synthetic data experiments."""
    config_dict = {
        'model_config': {
            'model_name': 'quantum_eegnet',
            'F1': 8,
            'D': 2,
            'F2': 16,
            'dropout_rate': 0.25,
            'num_classes': 2,
            'n_qubits': 4,
            'n_layers': 2,
            'input_channels': 2,
            'input_length': 128
        },
        'training_config': {
            'epochs': 50,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'optimizer': 'adamw',
            'weight_decay': 1e-4,
            'patience': 10,
            'num_workers': 0
        },
        'data_config': {
            'dataset_type': 'synthetic',
            'num_samples': 1000,
            'num_channels': 2,
            'time_points': 128,
            'num_classes': 2,
            'validation_ratio': 0.2
        },
        'experiment_config': {
            'experiment_name': 'synthetic_data_experiment',
            'save_dir': 'experiments',
            'log_dir': 'logs',
            'device': 'auto'
        }
    }
    return Config(config_dict)

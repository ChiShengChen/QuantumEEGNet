import torch
import torch.utils.data as Data
from scipy import io
import numpy as np
import os
from typing import Tuple, Optional


class BCIDataset:
    """
    BCI Competition IV 2a dataset loader.
    
    This class handles loading and preprocessing of the BCI Competition IV 2a dataset
    for motor imagery classification tasks.
    """
    
    def __init__(self, data_path: str, subject: int, validation_ratio: float = 0.2):
        """
        Initialize BCI dataset.
        
        Args:
            data_path (str): Path to the dataset directory
            subject (int): Subject number (1-9)
            validation_ratio (float): Ratio of training data to use for validation
        """
        self.data_path = data_path
        self.subject = subject
        self.validation_ratio = validation_ratio
        
        # Load data
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = \
            self._load_and_preprocess_data()
    
    def _load_and_preprocess_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                                torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load and preprocess the BCI dataset.
        
        Returns:
            Tuple of (x_train, y_train, x_valid, y_valid, x_test, y_test)
        """
        # Load training and test data
        train_file = os.path.join(self.data_path, f'BCIC_S{self.subject:02d}_T.mat')
        test_file = os.path.join(self.data_path, f'BCIC_S{self.subject:02d}_E.mat')
        
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        train_data = io.loadmat(train_file)
        test_data = io.loadmat(test_file)
        
        # Extract data
        x_train_raw = torch.Tensor(train_data['x_train']).unsqueeze(1)
        y_train_raw = torch.Tensor(train_data['y_train']).view(-1)
        x_test = torch.Tensor(test_data['x_test']).unsqueeze(1)
        y_test = torch.Tensor(test_data['y_test']).view(-1)
        
        # Split training data into train and validation
        x_train, y_train, x_valid, y_valid = self._split_train_valid(
            x_train_raw, y_train_raw, self.validation_ratio
        )
        
        # Preprocess data (time window selection)
        x_train = x_train[:, :, :, 124:562]  # Select time window
        x_valid = x_valid[:, :, :, 124:562]
        x_test = x_test[:, :, :, 124:562]
        
        # Convert labels to long type
        y_train = y_train.long()
        y_valid = y_valid.long()
        y_test = y_test.long()
        
        return x_train, y_train, x_valid, y_valid, x_test, y_test
    
    def _split_train_valid(self, x_train: torch.Tensor, y_train: torch.Tensor, 
                          ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split training data into training and validation sets.
        
        Args:
            x_train: Training features
            y_train: Training labels
            ratio: Validation ratio
            
        Returns:
            Tuple of (x_train, y_train, x_valid, y_valid)
        """
        # Sort by labels to ensure balanced split
        s = y_train.argsort()
        x_train = x_train[s]
        y_train = y_train[s]
        
        # Calculate samples per class
        cL = int(len(x_train) / 4)
        
        # Split each class
        class1_x = x_train[0 * cL:1 * cL]
        class2_x = x_train[1 * cL:2 * cL]
        class3_x = x_train[2 * cL:3 * cL]
        class4_x = x_train[3 * cL:4 * cL]
        
        class1_y = y_train[0 * cL:1 * cL]
        class2_y = y_train[1 * cL:2 * cL]
        class3_y = y_train[2 * cL:3 * cL]
        class4_y = y_train[3 * cL:4 * cL]
        
        # Calculate validation samples per class
        vL = int(len(class1_x) * ratio)
        
        # Create train and validation sets
        x_train = torch.cat((class1_x[:-vL], class2_x[:-vL], class3_x[:-vL], class4_x[:-vL]))
        y_train = torch.cat((class1_y[:-vL], class2_y[:-vL], class3_y[:-vL], class4_y[:-vL]))
        
        x_valid = torch.cat((class1_x[-vL:], class2_x[-vL:], class3_x[-vL:], class4_x[-vL:]))
        y_valid = torch.cat((class1_y[-vL:], class2_y[-vL:], class3_y[-vL:], class4_y[-vL:]))
        
        return x_train, y_train, x_valid, y_valid
    
    def get_dataloaders(self, batch_size: int = 64, num_workers: int = 0) -> Tuple[Data.DataLoader, Data.DataLoader, Data.DataLoader]:
        """
        Get PyTorch DataLoaders for training, validation, and test sets.
        
        Args:
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, valid_loader, test_loader)
        """
        # Create datasets
        train_dataset = Data.TensorDataset(self.x_train, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.y_test)
        
        # Create dataloaders
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        valid_loader = Data.DataLoader(
            dataset=valid_dataset,
            batch_size=1,  # Use batch size 1 for validation
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=1,  # Use batch size 1 for testing
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, valid_loader, test_loader
    
    def get_data_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            "subject": self.subject,
            "train_samples": len(self.x_train),
            "valid_samples": len(self.x_valid),
            "test_samples": len(self.x_test),
            "input_shape": self.x_train.shape[1:],
            "num_classes": len(torch.unique(self.y_train)),
            "class_distribution": {
                "train": torch.bincount(self.y_train).tolist(),
                "valid": torch.bincount(self.y_valid).tolist(),
                "test": torch.bincount(self.y_test).tolist()
            }
        }


class SyntheticDataset:
    """
    Synthetic dataset for testing and development.
    
    This class generates synthetic EEG-like data for testing the models
    without requiring the actual BCI dataset. The synthetic data simulates
    real EEG signals with class-dependent frequency patterns.
    
    Features:
    - Artificial EEG-like signals with configurable parameters
    - Class-dependent frequency patterns (10Hz, 15Hz, 20Hz, etc.)
    - Realistic data structure matching real EEG format
    - Fast generation for development and testing
    
    Usage:
        dataset = SyntheticDataset(num_samples=1000, num_classes=2)
        train_loader, valid_loader, test_loader = dataset.get_dataloaders()
    
    Note:
        This is for development purposes only. For final evaluation,
        use the real BCI Competition IV 2a dataset.
    """
    
    def __init__(self, num_samples: int = 1000, num_channels: int = 2, 
                 time_points: int = 128, num_classes: int = 2, 
                 validation_ratio: float = 0.2):
        """
        Initialize synthetic dataset.
        
        Args:
            num_samples: Number of samples to generate
            num_channels: Number of EEG channels
            time_points: Number of time points
            num_classes: Number of classes
            validation_ratio: Ratio for validation split
        """
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.time_points = time_points
        self.num_classes = num_classes
        self.validation_ratio = validation_ratio
        
        # Generate synthetic data
        self.x_train, self.y_train, self.x_valid, self.y_valid, self.x_test, self.y_test = \
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, 
                                               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate synthetic EEG-like data.
        
        Returns:
            Tuple of (x_train, y_train, x_valid, y_valid, x_test, y_test)
        """
        # Generate training data
        train_samples = int(self.num_samples * (1 - self.validation_ratio))
        valid_samples = self.num_samples - train_samples
        test_samples = int(self.num_samples * 0.2)
        
        # Generate features with some class-dependent patterns
        x_train = torch.randn(train_samples, 1, self.num_channels, self.time_points)
        x_valid = torch.randn(valid_samples, 1, self.num_channels, self.time_points)
        x_test = torch.randn(test_samples, 1, self.num_channels, self.time_points)
        
        # Generate labels
        y_train = torch.randint(0, self.num_classes, (train_samples,))
        y_valid = torch.randint(0, self.num_classes, (valid_samples,))
        y_test = torch.randint(0, self.num_classes, (test_samples,))
        
        # Add some class-dependent patterns to make the data more realistic
        for i in range(self.num_classes):
            # Add different frequency components for different classes
            freq = 10 + i * 5  # Different frequency for each class
            
            # Apply to training data
            mask_train = (y_train == i)
            if mask_train.sum() > 0:
                t = torch.linspace(0, 2*np.pi, self.time_points)
                pattern = torch.sin(freq * t).unsqueeze(0).unsqueeze(0)
                x_train[mask_train] += 0.1 * pattern
            
            # Apply to validation data
            mask_valid = (y_valid == i)
            if mask_valid.sum() > 0:
                t = torch.linspace(0, 2*np.pi, self.time_points)
                pattern = torch.sin(freq * t).unsqueeze(0).unsqueeze(0)
                x_valid[mask_valid] += 0.1 * pattern
            
            # Apply to test data
            mask_test = (y_test == i)
            if mask_test.sum() > 0:
                t = torch.linspace(0, 2*np.pi, self.time_points)
                pattern = torch.sin(freq * t).unsqueeze(0).unsqueeze(0)
                x_test[mask_test] += 0.1 * pattern
        
        return x_train, y_train, x_valid, y_valid, x_test, y_test
    
    def get_dataloaders(self, batch_size: int = 64, num_workers: int = 0) -> Tuple[Data.DataLoader, Data.DataLoader, Data.DataLoader]:
        """
        Get PyTorch DataLoaders for training, validation, and test sets.
        
        Args:
            batch_size: Batch size for training
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, valid_loader, test_loader)
        """
        # Create datasets
        train_dataset = Data.TensorDataset(self.x_train, self.y_train)
        valid_dataset = Data.TensorDataset(self.x_valid, self.y_valid)
        test_dataset = Data.TensorDataset(self.x_test, self.y_test)
        
        # Create dataloaders
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        valid_loader = Data.DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, valid_loader, test_loader
    
    def get_data_info(self) -> dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            "dataset_type": "synthetic",
            "train_samples": len(self.x_train),
            "valid_samples": len(self.x_valid),
            "test_samples": len(self.x_test),
            "input_shape": self.x_train.shape[1:],
            "num_classes": self.num_classes,
            "class_distribution": {
                "train": torch.bincount(self.y_train).tolist(),
                "valid": torch.bincount(self.y_valid).tolist(),
                "test": torch.bincount(self.y_test).tolist()
            }
        }

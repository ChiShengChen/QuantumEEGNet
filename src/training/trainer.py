import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:
    """
    Training manager for QuantumEEGNet models.
    
    This class handles the training, validation, and testing of quantum-classical
    hybrid neural networks for EEG signal classification.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 train_loader: DataLoader, valid_loader: DataLoader, 
                 test_loader: DataLoader, config: Dict):
        """
        Initialize the trainer.
        
        Args:
            model: Neural network model to train
            device: Device to run training on (CPU/GPU)
            train_loader: Training data loader
            valid_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration dictionary
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.config = config
        
        # Initialize optimizer and loss function
        self.optimizer = self._get_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.valid_losses = []
        self.train_accuracies = []
        self.valid_accuracies = []
        self.best_valid_accuracy = 0.0
        self.best_model_state = None
        
        # Create experiment directory
        self.experiment_dir = self._create_experiment_dir()
        
    def _get_optimizer(self) -> optim.Optimizer:
        """Initialize optimizer based on configuration."""
        optimizer_name = self.config.get('optimizer', 'adam')
        learning_rate = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 0.0)
        
        if optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_experiment_dir(self) -> str:
        """Create experiment directory for saving results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.config.get('model_name', 'quantum_eegnet')
        experiment_dir = os.path.join('experiments', f"{model_name}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.valid_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def test(self) -> Tuple[float, float, np.ndarray]:
        """
        Test the model.
        
        Returns:
            Tuple of (average_loss, accuracy, confusion_matrix)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                total_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_predictions.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        # Calculate confusion matrix
        num_classes = self.config.get('num_classes', 2)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        for pred, target in zip(all_predictions, all_targets):
            confusion_matrix[target, pred] += 1
        
        return avg_loss, accuracy, confusion_matrix
    
    def train(self) -> Dict:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Dictionary containing training results
        """
        epochs = self.config.get('epochs', 100)
        patience = self.config.get('patience', 10)
        early_stopping_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {self.model.count_parameters():,}")
        
        for epoch in range(1, epochs + 1):
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            valid_loss, valid_acc = self.validate()
            
            # Store history
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)
            self.train_accuracies.append(train_acc)
            self.valid_accuracies.append(valid_acc)
            
            # Print progress
            print(f'Epoch {epoch:3d}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.2f}%')
            
            # Save best model
            if valid_acc > self.best_valid_accuracy:
                self.best_valid_accuracy = valid_acc
                self.best_model_state = self.model.state_dict().copy()
                early_stopping_counter = 0
                print(f"New best validation accuracy: {valid_acc:.2f}%")
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Load best model for testing
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Final testing
        test_loss, test_acc, confusion_matrix = self.test()
        
        # Save results
        results = {
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'train_accuracies': self.train_accuracies,
            'valid_accuracies': self.valid_accuracies,
            'best_valid_accuracy': self.best_valid_accuracy,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'confusion_matrix': confusion_matrix.tolist(),
            'config': self.config
        }
        
        self._save_results(results)
        self._plot_training_curves()
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_valid_accuracy:.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%")
        
        return results
    
    def _save_results(self, results: Dict):
        """Save training results to files."""
        # Save results as JSON
        results_file = os.path.join(self.experiment_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save model
        model_file = os.path.join(self.experiment_dir, 'model.pth')
        torch.save({
            'model_state_dict': self.best_model_state,
            'config': self.config,
            'results': results
        }, model_file)
        
        print(f"Results saved to: {self.experiment_dir}")
    
    def _plot_training_curves(self):
        """Plot training and validation curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.valid_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.valid_accuracies, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plot_file = os.path.join(self.experiment_dir, 'training_curves.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {plot_file}")


def load_trained_model(model_path: str, model_class, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Load a trained model from file.
    
    Args:
        model_path: Path to the saved model file
        model_class: Model class to instantiate
        device: Device to load the model on
        
    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model instance
    model = model_class(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, config

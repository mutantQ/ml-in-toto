#!/usr/bin/env python3
"""
Enhanced Training Module for MLOps Pipeline
Includes experiment tracking, hyperparameter tuning, early stopping, and model registry
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
import structlog
import mlflow
import mlflow.pytorch
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

# Setup structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()


class MNISTDataset(Dataset):
    """Custom MNIST Dataset"""
    
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].float() / 255.0
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class EnhancedCNN(nn.Module):
    """Enhanced CNN with configurable architecture"""
    
    def __init__(self, config: DictConfig):
        super(EnhancedCNN, self).__init__()
        self.config = config
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(config.model.dropout_rate)
        self.dropout2 = nn.Dropout(config.model.dropout_rate * 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, config.model.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Pooling and dropout
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # First FC layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class ModelTrainer:
    """Enhanced model trainer with MLflow integration"""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = self._get_device()
        
        # Setup MLflow
        mlflow.set_tracking_uri(config.registry.tracking_uri)
        mlflow.set_experiment(config.registry.experiment_name)
        
        # Setup Weights & Biases if enabled
        if hasattr(config, 'wandb') and config.wandb.enabled:
            wandb.init(
                project=config.wandb.project_name,
                config=OmegaConf.to_container(config, resolve=True)
            )
    
    def _get_device(self):
        """Get optimal device for training"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def load_data(self, data_path: Path) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data loaders"""
        # Load processed data
        train_data = torch.load(data_path / "train" / "dataset.pt")
        test_data = torch.load(data_path / "test" / "dataset.pt")
        
        train_images = train_data['images']
        train_labels = train_data['labels']
        test_images = test_data['images']
        test_labels = test_data['labels']
        
        # Create datasets
        train_dataset = MNISTDataset(train_images, train_labels)
        test_dataset = MNISTDataset(test_images, test_labels)
        
        # Split training data for validation
        train_size = int((1 - self.config.data.validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(self.config.data.random_seed)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.testing.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True if self.device.type == "cuda" else False
        )
        
        return train_loader, val_loader, test_loader
    
    def create_model(self) -> nn.Module:
        """Create and initialize model"""
        model = EnhancedCNN(self.config).to(self.device)
        return model
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_name = self.config.model.optimizer.lower()
        
        if optimizer_name == "adam":
            return optim.Adam(model.parameters(), lr=self.config.model.learning_rate)
        elif optimizer_name == "sgd":
            return optim.SGD(model.parameters(), lr=self.config.model.learning_rate)
        elif optimizer_name == "adadelta":
            return optim.Adadelta(model.parameters(), lr=self.config.model.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def create_scheduler(self, optimizer: optim.Optimizer):
        """Create learning rate scheduler"""
        scheduler_name = self.config.model.scheduler.lower()
        
        if scheduler_name == "steplr":
            return StepLR(optimizer, step_size=self.config.model.step_size, 
                         gamma=self.config.model.gamma)
        elif scheduler_name == "reducelronplateau":
            return ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
        else:
            return None
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                   optimizer: optim.Optimizer, epoch: int) -> Dict[str, float]:
        """Train model for one epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        num_batches = len(train_loader)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if batch_idx % self.config.training.log_interval == 0:
                logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / num_batches
        accuracy = 100. * correct / len(train_loader.dataset)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def validate_epoch(self, model: nn.Module, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        model.eval()
        total_loss = 0.0
        correct = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = F.nll_loss(output, target, reduction='sum')
                total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = 100. * correct / len(val_loader.dataset)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    def train_model(self, data_path: Path) -> Dict[str, Any]:
        """Train model with MLflow tracking"""
        with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log parameters
            mlflow.log_params(OmegaConf.to_container(self.config, resolve=True))
            
            # Load data
            train_loader, val_loader, test_loader = self.load_data(data_path)
            
            # Create model, optimizer, and scheduler
            model = self.create_model()
            optimizer = self.create_optimizer(model)
            scheduler = self.create_scheduler(optimizer)
            
            # Early stopping
            early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping_patience,
                min_delta=self.config.training.min_delta
            )
            
            # Training loop
            train_losses = []
            val_losses = []
            train_accuracies = []
            val_accuracies = []
            
            for epoch in range(1, self.config.training.epochs + 1):
                # Train epoch
                train_metrics = self.train_epoch(model, train_loader, optimizer, epoch)
                
                # Validate epoch
                val_metrics = self.validate_epoch(model, val_loader)
                
                # Update scheduler
                if scheduler:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(val_metrics["loss"])
                    else:
                        scheduler.step()
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_f1": val_metrics["f1"],
                    "learning_rate": optimizer.param_groups[0]["lr"]
                }, step=epoch)
                
                # Store metrics
                train_losses.append(train_metrics["loss"])
                val_losses.append(val_metrics["loss"])
                train_accuracies.append(train_metrics["accuracy"])
                val_accuracies.append(val_metrics["accuracy"])
                
                # Log to Weights & Biases
                if hasattr(self.config, 'wandb') and self.config.wandb.enabled:
                    wandb.log({
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"]
                    })
                
                # Early stopping check
                if early_stopping(val_metrics["loss"], model):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                
                # Save checkpoint
                if epoch % self.config.training.checkpoint_interval == 0:
                    checkpoint_path = Path("checkpoints") / f"model_epoch_{epoch}.pt"
                    checkpoint_path.parent.mkdir(exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_metrics["loss"],
                        'val_loss': val_metrics["loss"]
                    }, checkpoint_path)
            
            # Final evaluation on test set
            test_metrics = self.validate_epoch(model, test_loader)
            
            # Log final test metrics
            mlflow.log_metrics({
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"]
            })
            
            # Save final model
            model_path = Path("models") / "final_model.pt"
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
            # Log model to MLflow
            mlflow.pytorch.log_model(model, "model")
            
            # Create and log training plots
            self.create_training_plots(train_losses, val_losses, train_accuracies, val_accuracies)
            
            # Register model in MLflow Model Registry
            if test_metrics["accuracy"] > self.config.testing.threshold:
                model_uri = f"runs:/{run.info.run_id}/model"
                mlflow.register_model(model_uri, self.config.registry.model_name)
                logger.info(f"Model registered successfully with accuracy: {test_metrics['accuracy']:.2f}%")
            
            results = {
                "run_id": run.info.run_id,
                "final_train_loss": train_losses[-1],
                "final_val_loss": val_losses[-1],
                "final_train_accuracy": train_accuracies[-1],
                "final_val_accuracy": val_accuracies[-1],
                "test_metrics": test_metrics,
                "model_path": str(model_path)
            }
            
            return results
    
    def create_training_plots(self, train_losses: List[float], val_losses: List[float],
                            train_accuracies: List[float], val_accuracies: List[float]):
        """Create and save training plots"""
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("reports/training") / "training_plots.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        # Log to MLflow
        mlflow.log_artifact(str(plot_path), "training_plots")
        
        plt.close()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Enhanced Model Training")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to processed data directory")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Train model
    results = trainer.train_model(Path(args.data_path))
    
    # Save results
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Training completed successfully!")
    print(f"Results saved to: training_results.json")


if __name__ == "__main__":
    main()
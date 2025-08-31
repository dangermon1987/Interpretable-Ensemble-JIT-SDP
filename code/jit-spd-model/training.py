"""
Training management for JIT-SPD model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from .config import TrainingConfig, ModelConfig
from .model import JITSPDModel


class EarlyStopping:
    """
    Early stopping mechanism for training.
    
    Code Reference: new_effort/EarlyStopping.py (lines 1-68)
    """
    
    def __init__(self, exp: str, meta: Dict[str, Any], patience: int = 8, 
                 min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            exp: Experiment directory path
            meta: Metadata dictionary
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in monitored quantity to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.exp = exp
        self.meta = meta
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.best_model = None
        
        # Create experiment directory
        os.makedirs(exp, exist_ok=True)
        
        # Save metadata
        torch.save(meta, os.path.join(exp, 'meta.txt'))
    
    def __call__(self, model: nn.Module, optimizer: optim.Optimizer, 
                 scheduler: Optional[optim.lr_scheduler._LRScheduler], 
                 epoch: int, val_loss: float, val_f1: float, 
                 val_mcc: float, val_auc: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            model: Model to monitor
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            epoch: Current epoch
            val_loss: Validation loss
            val_f1: Validation F1 score
            val_mcc: Validation MCC score
            val_auc: Validation AUC score
            
        Returns:
            True if training should stop, False otherwise
        """
        # Use F1 + AUC as the primary metric
        score = val_f1 + val_auc
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model = model.state_dict().copy()
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at epoch {epoch}")
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_model = model.state_dict().copy()
            self.counter = 0
        
        # Save best model
        if self.best_model is not None:
            torch.save({
                'model_state_dict': self.best_model,
                'epoch': self.best_epoch,
                'best_score': self.best_score,
                'meta': self.meta
            }, os.path.join(self.exp, 'model.bin'))
        
        return self.early_stop
    
    def get_best_model_state(self) -> Dict[str, Any]:
        """Get the best model state dictionary."""
        return self.best_model
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best performance."""
        return self.best_epoch


class TrainingManager:
    """
    Manages the training process for the JIT-SPD model.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the training manager.
        
        Args:
            config: Training configuration (uses default if None)
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {}
        self.training_history = []
    
    def setup_training(self, model: JITSPDModel, train_dataset, valid_dataset, test_dataset) -> None:
        """
        Setup training components.
        
        Args:
            model: JIT-SPD model to train
            train_dataset: Training dataset
            valid_dataset: Validation dataset
            test_dataset: Test dataset
        """
        self.model = model.to(self.device)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup loss function
        self._setup_criterion(train_dataset)
        
        # Setup scheduler
        if self.config.use_lr_scheduling:
            self._setup_scheduler()
        
        # Setup TensorBoard
        if self.config.use_tensorboard:
            self._setup_tensorboard()
        
        # Setup datasets
        self._setup_datasets(train_dataset, valid_dataset, test_dataset)
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer with weight decay."""
        no_decay = ['bias', 'layernorm.weight', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters, 
            lr=self.config.learning_rate
        )
    
    def _setup_criterion(self, train_dataset) -> None:
        """Setup loss function."""
        # Calculate positive weight for balanced loss
        if hasattr(train_dataset, 'neg_count') and hasattr(train_dataset, 'pos_count'):
            pos_weight = torch.tensor(train_dataset.neg_count / train_dataset.pos_count)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            patience=self.config.patience,
            min_lr=self.config.min_lr,
            factor=self.config.lr_factor
        )
    
    def _setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(self.config.result_path, 'tensorboard', timestamp)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
    
    def _setup_datasets(self, train_dataset, valid_dataset, test_dataset) -> None:
        """Setup datasets with configuration."""
        # Store datasets
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        
        # Get sample weights for balanced sampling
        if self.config.use_weighted_sampling and hasattr(train_dataset, 'get_sample_weights'):
            self.sample_weights = train_dataset.get_sample_weights().to(self.device)
        else:
            self.sample_weights = None
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        # Create data loader with balanced sampling if available
        if self.sample_weights is not None:
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(
                weights=self.sample_weights,
                num_samples=len(self.train_dataset),
                replacement=True
            )
            data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                sampler=sampler
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        
        epoch_start_time = datetime.now()
        
        for batch_idx, batch in enumerate(data_loader):
            # Unpack batch
            homogeneous_data, commit_message, code, features_embedding, labels, commits = batch
            
            # Get text embeddings
            text_embedding = torch.stack([
                self.train_text_embeddings[commit_id]['cls_embeddings'] for commit_id in commits
            ], dim=0)
            
            # Prepare data dictionary
            data = {
                'x_dict': homogeneous_data.x.to(self.device),
                'edge_index': homogeneous_data.edge_index.to(self.device),
                'batch': homogeneous_data.batch.to(self.device),
                'text_embedding': text_embedding.to(self.device),
                'features_embedding': features_embedding.to(self.device),
                'batch_size': len(labels)
            }
            
            # Forward pass
            logits, graph_embeds, graph_attn_weights = self.model(data)
            labels = labels.unsqueeze(-1)
            
            # Calculate loss
            loss = self.criterion(logits, labels.to(self.device))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            total_samples += len(labels)
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate average loss
        avg_loss = total_loss / len(data_loader)
        
        # Log to TensorBoard
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            if self.scheduler is not None:
                self.writer.add_scalar('LearningRate/train', 
                                     self.scheduler.get_last_lr()[0], epoch)
            self.writer.flush()
        
        epoch_time = datetime.now() - epoch_start_time
        
        return {
            'epoch': epoch,
            'train_loss': avg_loss,
            'epoch_time': epoch_time,
            'total_samples': total_samples
        }
    
    def evaluate(self, dataset, text_embeddings: Dict, dataset_name: str = "validation") -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Dataset to evaluate on
            text_embeddings: Text embeddings for the dataset
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_embeddings = []
        
        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in data_loader:
                # Unpack batch
                homogeneous_data, commit_message, code, features_embedding, labels, commits = batch
                
                # Get text embeddings
                text_embedding = torch.stack([
                    text_embeddings[commit_id]['cls_embeddings'] for commit_id in commits
                ], dim=0)
                
                # Prepare data
                data = {
                    'x_dict': homogeneous_data.x.to(self.device),
                    'edge_index': homogeneous_data.edge_index.to(self.device),
                    'batch': homogeneous_data.batch.to(self.device),
                    'text_embedding': text_embedding.to(self.device),
                    'features_embedding': features_embedding.to(self.device),
                    'batch_size': len(labels)
                }
                
                # Forward pass
                logits, embeddings, graph_attn_weights = self.model(data)
                labels = labels.unsqueeze(-1)
                
                # Calculate loss
                loss = self.criterion(logits, labels.to(self.device))
                total_loss += loss.item()
                
                # Store predictions and labels
                predictions = torch.sigmoid(logits).cpu().detach().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().detach().numpy())
                all_embeddings.extend(embeddings.cpu().detach().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_predictions, all_labels)
        metrics['loss'] = total_loss / len(data_loader)
        metrics['dataset'] = dataset_name
        
        # Log to TensorBoard
        if self.writer is not None:
            for metric_name, metric_value in metrics.items():
                if metric_name not in ['dataset']:
                    self.writer.add_scalar(f'{metric_name}/{dataset_name}', metric_value, self.current_epoch)
            self.writer.flush()
        
        return metrics
    
    def _calculate_metrics(self, predictions: list, labels: list) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            
        Returns:
            Dictionary containing metrics
        """
        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score, precision_score, roc_auc_score, matthews_corrcoef
        
        predictions = np.array(predictions).flatten()
        pred_labels = [1 if p > 0.5 else 0 for p in predictions]
        pred_labels = np.array(pred_labels).flatten()
        labels = np.array(labels).flatten()
        
        # Calculate metrics
        f1 = f1_score(labels, pred_labels)
        accuracy = accuracy_score(labels, pred_labels)
        precision = precision_score(labels, pred_labels, zero_division=0.0)
        auc = roc_auc_score(labels, predictions) if len(set(labels)) > 1 else 0.5
        mcc = matthews_corrcoef(labels, pred_labels)
        
        return {
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'auc': auc,
            'mcc': mcc
        }
    
    def update_scheduler(self, metrics: Dict[str, float]) -> None:
        """
        Update learning rate scheduler based on validation metrics.
        
        Args:
            metrics: Validation metrics
        """
        if self.scheduler is not None:
            # Use F1 + AUC as the primary metric for scheduling
            primary_metric = metrics.get('f1', 0) + metrics.get('auc', 0)
            self.scheduler.step(primary_metric)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False) -> None:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.result_path, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.result_path, 'best_checkpoint.pt')
            torch.save(checkpoint, best_path)
            print(f"New best checkpoint saved at epoch {epoch}")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch number of loaded checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def close(self) -> None:
        """Clean up training resources."""
        if self.writer is not None:
            self.writer.close()
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training process.
        
        Returns:
            Dictionary containing training summary
        """
        return {
            'current_epoch': self.current_epoch,
            'best_metrics': self.best_metrics,
            'training_history': self.training_history,
            'config': self.config.to_dict(),
            'model_info': self.model.get_model_info() if self.model else None
        }

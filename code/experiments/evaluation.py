"""
Model evaluation utilities for JIT-SPD model.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from .metrics import MetricsCalculator


class ModelEvaluator:
    """
    Evaluates trained JIT-SPD models.
    """
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to the trained model
            config_path: Path to model configuration (optional)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and configuration
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model and configuration."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model state and config
        if 'model_state_dict' in checkpoint:
            self.model_state = checkpoint['model_state_dict']
            self.config = checkpoint.get('config', {})
        else:
            # Handle legacy format
            self.model_state = checkpoint
            self.config = {}
        
        # Load configuration from separate file if available
        if self.config_path and os.path.exists(self.config_path):
            self.config = torch.load(self.config_path, map_location=self.device)
        
        print(f"Model loaded from: {self.model_path}")
        print(f"Configuration: {self.config}")
    
    def evaluate_model(self, test_dataset, text_embeddings: Dict) -> Dict[str, Any]:
        """
        Evaluate the model on test dataset.
        
        Args:
            test_dataset: Test dataset
            text_embeddings: Text embeddings for test dataset
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call setup_model() first.")
        
        # Setup model for evaluation
        self.model.eval()
        
        # Evaluation metrics
        all_predictions = []
        all_labels = []
        all_embeddings = []
        all_attention_weights = []
        
        # Create data loader
        from torch.utils.data import DataLoader
        data_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
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
                logits, embeddings, attention_weights = self.model(data)
                labels = labels.unsqueeze(-1)
                
                # Store results
                predictions = torch.sigmoid(logits).cpu().detach().numpy()
                all_predictions.extend(predictions)
                all_labels.extend(labels.cpu().detach().numpy())
                all_embeddings.extend(embeddings.cpu().detach().numpy())
                all_attention_weights.extend(attention_weights)
        
        # Calculate metrics
        metrics_calculator = MetricsCalculator()
        metrics = metrics_calculator.calculate_metrics(all_predictions, all_labels)
        
        # Additional analysis
        analysis = self._analyze_predictions(all_predictions, all_labels, all_embeddings)
        
        return {
            'metrics': metrics,
            'analysis': analysis,
            'predictions': all_predictions,
            'labels': all_labels,
            'embeddings': all_embeddings,
            'attention_weights': all_attention_weights
        }
    
    def _analyze_predictions(self, predictions: List[float], labels: List[int], 
                            embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze model predictions and embeddings.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            embeddings: Model embeddings
            
        Returns:
            Dictionary containing analysis results
        """
        predictions = np.array(predictions).flatten()
        labels = np.array(labels).flatten()
        embeddings = np.array(embeddings)
        
        # Prediction distribution
        pred_distribution = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'median': np.median(predictions)
        }
        
        # Confidence analysis
        confidence_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        confidence_analysis = {}
        
        for conf in confidence_levels:
            high_conf_mask = (predictions > conf) | (predictions < (1 - conf))
            if np.any(high_conf_mask):
                high_conf_preds = predictions[high_conf_mask]
                high_conf_labels = labels[high_conf_mask]
                
                # Calculate accuracy for high confidence predictions
                high_conf_accuracy = np.mean(
                    (high_conf_preds > 0.5) == high_conf_labels
                )
                
                confidence_analysis[f'confidence_{conf}'] = {
                    'count': np.sum(high_conf_mask),
                    'accuracy': high_conf_accuracy,
                    'percentage': np.sum(high_conf_mask) / len(predictions) * 100
                }
        
        # Embedding analysis
        embedding_analysis = {
            'dimension': embeddings.shape[1],
            'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
            'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
            'clustering_score': self._calculate_clustering_score(embeddings, labels)
        }
        
        return {
            'prediction_distribution': pred_distribution,
            'confidence_analysis': confidence_analysis,
            'embedding_analysis': embedding_analysis
        }
    
    def _calculate_clustering_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate clustering quality score for embeddings.
        
        Args:
            embeddings: Model embeddings
            labels: Ground truth labels
            
        Returns:
            Clustering quality score
        """
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(embeddings, labels)
        except ImportError:
            return 0.0
    
    def generate_evaluation_report(self, evaluation_results: Dict[str, Any], 
                                 output_path: str = None) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from evaluate_model()
            output_path: Path to save the report (optional)
            
        Returns:
            Path to the generated report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"evaluation_report_{timestamp}.txt"
        
        with open(output_path, 'w') as f:
            f.write("JIT-SPD Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Model information
            f.write("Model Information:\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Configuration: {self.config}\n\n")
            
            # Metrics
            f.write("Performance Metrics:\n")
            f.write("-" * 20 + "\n")
            metrics = evaluation_results['metrics']
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    f.write(f"{metric_name.upper()}: {metric_value:.4f}\n")
                else:
                    f.write(f"{metric_name.upper()}: {metric_value}\n")
            f.write("\n")
            
            # Analysis
            f.write("Prediction Analysis:\n")
            f.write("-" * 20 + "\n")
            analysis = evaluation_results['analysis']
            
            # Prediction distribution
            pred_dist = analysis['prediction_distribution']
            f.write("Prediction Distribution:\n")
            for key, value in pred_dist.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")
            
            # Confidence analysis
            f.write("Confidence Analysis:\n")
            for conf_key, conf_data in analysis['confidence_analysis'].items():
                f.write(f"  {conf_key}:\n")
                f.write(f"    Count: {conf_data['count']}\n")
                f.write(f"    Accuracy: {conf_data['accuracy']:.4f}\n")
                f.write(f"    Percentage: {conf_data['percentage']:.2f}%\n")
            f.write("\n")
            
            # Embedding analysis
            f.write("Embedding Analysis:\n")
            emb_analysis = analysis['embedding_analysis']
            f.write(f"  Dimension: {emb_analysis['dimension']}\n")
            f.write(f"  Mean Norm: {emb_analysis['mean_norm']:.4f}\n")
            f.write(f"  Std Norm: {emb_analysis['std_norm']:.4f}\n")
            f.write(f"  Clustering Score: {emb_analysis['clustering_score']:.4f}\n")
        
        print(f"Evaluation report saved to: {output_path}")
        return output_path
    
    def export_predictions(self, evaluation_results: Dict[str, Any], 
                          output_path: str, format: str = 'csv') -> str:
        """
        Export predictions and labels to file.
        
        Args:
            evaluation_results: Results from evaluate_model()
            output_path: Path to save the file
            format: Output format ('csv', 'json', 'excel')
            
        Returns:
            Path to the exported file
        """
        # Prepare data
        data = {
            'predictions': evaluation_results['predictions'],
            'labels': evaluation_results['labels']
        }
        
        # Export based on format
        if format == 'csv':
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
        elif format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'excel':
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Predictions exported to: {output_path}")
        return output_path
    
    def compare_with_baseline(self, baseline_results: Dict[str, Any], 
                             current_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare current model with baseline.
        
        Args:
            baseline_results: Baseline model results
            current_results: Current model results
            
        Returns:
            Dictionary containing comparison results
        """
        comparison = {}
        
        # Compare metrics
        baseline_metrics = baseline_results['metrics']
        current_metrics = current_results['metrics']
        
        for metric_name in baseline_metrics.keys():
            if metric_name in current_metrics:
                baseline_value = baseline_metrics[metric_name]
                current_value = current_metrics[metric_name]
                
                if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                    improvement = current_value - baseline_value
                    improvement_pct = (improvement / baseline_value * 100) if baseline_value != 0 else 0
                    
                    comparison[metric_name] = {
                        'baseline': baseline_value,
                        'current': current_value,
                        'improvement': improvement,
                        'improvement_pct': improvement_pct
                    }
        
        return comparison


def evaluate_existing_model(experiment_idx: int, result_location: str) -> Dict[str, Any]:
    """
    Evaluate an existing trained model (backward compatibility function).
    
    Args:
        experiment_idx: Index of the experiment
        result_location: Location of experiment results
        
    Returns:
        Dictionary containing evaluation results
    """
    # Construct paths
    runs_dir = os.path.join(result_location, str(experiment_idx))
    model_path = os.path.join(runs_dir, 'model.bin')
    meta_path = os.path.join(runs_dir, 'meta.txt')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load metadata
    meta = torch.load(meta_path, weights_only=False)
    
    # Create evaluator
    evaluator = ModelEvaluator(model_path)
    
    # Load datasets (this would need to be implemented based on your data loading logic)
    # For now, return basic information
    return {
        'experiment_idx': experiment_idx,
        'result_location': result_location,
        'model_path': model_path,
        'metadata': meta,
        'status': 'Model loaded successfully'
    }

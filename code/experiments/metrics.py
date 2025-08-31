"""
Metrics calculation for JIT-SPD model evaluation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, 
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)


class MetricsCalculator:
    """
    Calculates and analyzes evaluation metrics for the JIT-SPD model.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        pass
    
    def calculate_metrics(self, predictions: List[float], labels: List[int]) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            predictions: Model predictions (probabilities)
            labels: Ground truth labels
            
        Returns:
            Dictionary containing all metrics
        """
        # Convert to numpy arrays
        predictions = np.array(predictions).flatten()
        labels = np.array(labels).flatten()
        
        # Convert probabilities to binary predictions
        pred_labels = [1 if p > 0.5 else 0 for p in predictions]
        pred_labels = np.array(pred_labels).flatten()
        
        # Basic metrics
        f1 = f1_score(labels, pred_labels)
        accuracy = accuracy_score(labels, pred_labels)
        precision = precision_score(labels, pred_labels, zero_division=0.0)
        recall = f1_score(labels, pred_labels, average='binary', zero_division=0.0)
        
        # Advanced metrics
        auc = roc_auc_score(labels, predictions) if len(set(labels)) > 1 else 0.5
        mcc = matthews_corrcoef(labels, pred_labels)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F-beta scores
        f2 = f1_score(labels, pred_labels, beta=2, zero_division=0.0)
        f0_5 = f1_score(labels, pred_labels, beta=0.5, zero_division=0.0)
        
        return {
            'f1': f1,
            'f2': f2,
            'f0_5': f0_5,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc,
            'mcc': mcc,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    def calculate_confidence_intervals(self, results: List[Dict[str, float]], 
                                     confidence_level: float = 0.95) -> Dict[str, Dict[str, float]]:
        """
        Calculate confidence intervals for metrics across multiple runs.
        
        Args:
            results: List of metric dictionaries from multiple runs
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary containing confidence intervals for each metric
        """
        if not results:
            return {}
        
        # Extract metrics for each run
        metrics_data = {}
        for result in results:
            for metric_name, metric_value in result.items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(metric_value)
        
        # Calculate statistics
        confidence_intervals = {}
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%
        
        for metric_name, values in metrics_data.items():
            values = np.array(values)
            mean = np.mean(values)
            std = np.std(values)
            n = len(values)
            
            # Standard error
            se = std / np.sqrt(n)
            
            # Confidence interval
            ci = z_score * se
            
            # Coefficient of variation
            cv = std / mean if mean != 0 else 0
            
            confidence_intervals[metric_name] = {
                'mean': mean,
                'std': std,
                'se': se,
                'cv': cv,
                'ci_lower': mean - ci,
                'ci_upper': mean + ci,
                'confidence_level': confidence_level
            }
        
        return confidence_intervals
    
    def generate_classification_report(self, predictions: List[float], 
                                     labels: List[int]) -> str:
        """
        Generate detailed classification report.
        
        Args:
            predictions: Model predictions (probabilities)
            labels: Ground truth labels
            
        Returns:
            Formatted classification report
        """
        # Convert probabilities to binary predictions
        pred_labels = [1 if p > 0.5 else 0 for p in predictions]
        
        # Generate report
        report = classification_report(
            labels, pred_labels, 
            target_names=['Non-Buggy', 'Buggy'],
            digits=4
        )
        
        return report
    
    def calculate_roc_curve(self, predictions: List[float], 
                           labels: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ROC curve data.
        
        Args:
            predictions: Model predictions (probabilities)
            labels: Ground truth labels
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        return fpr, tpr, thresholds
    
    def calculate_pr_curve(self, predictions: List[float], 
                          labels: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate Precision-Recall curve data.
        
        Args:
            predictions: Model predictions (probabilities)
            labels: Ground truth labels
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        precision, recall, thresholds = precision_recall_curve(labels, predictions)
        return precision, recall, thresholds
    
    def calculate_threshold_metrics(self, predictions: List[float], 
                                  labels: List[int], 
                                  thresholds: List[float] = None) -> pd.DataFrame:
        """
        Calculate metrics for different classification thresholds.
        
        Args:
            predictions: Model predictions (probabilities)
            labels: Ground truth labels
            thresholds: List of thresholds to evaluate (default: 0.1 to 0.9)
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
        
        results = []
        predictions = np.array(predictions)
        labels = np.array(labels)
        
        for threshold in thresholds:
            # Convert to binary predictions
            pred_labels = (predictions > threshold).astype(int)
            
            # Calculate metrics
            metrics = self.calculate_metrics(predictions, labels)
            
            # Store results
            results.append({
                'threshold': threshold,
                **metrics
            })
        
        return pd.DataFrame(results)
    
    def compare_models(self, model_results: Dict[str, List[Dict[str, float]]]) -> pd.DataFrame:
        """
        Compare multiple models based on their metrics.
        
        Args:
            model_results: Dictionary mapping model names to results lists
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for model_name, results in model_results.items():
            # Calculate confidence intervals
            ci_data = self.calculate_confidence_intervals(results)
            
            # Extract key metrics
            for metric_name, metric_data in ci_data.items():
                if metric_name in ['f1', 'accuracy', 'precision', 'recall', 'auc', 'mcc']:
                    comparison_data.append({
                        'model': model_name,
                        'metric': metric_name,
                        'mean': metric_data['mean'],
                        'std': metric_data['std'],
                        'ci_lower': metric_data['ci_lower'],
                        'ci_upper': metric_data['ci_upper'],
                        'cv': metric_data['cv']
                    })
        
        return pd.DataFrame(comparison_data)
    
    def generate_summary_statistics(self, results: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Generate summary statistics for experiment results.
        
        Args:
            results: List of metric dictionaries
            
        Returns:
            Dictionary containing summary statistics
        """
        if not results:
            return {}
        
        # Calculate confidence intervals
        ci_data = self.calculate_confidence_intervals(results)
        
        # Summary statistics
        summary = {
            'total_runs': len(results),
            'confidence_level': 0.95,
            'metrics_summary': ci_data,
            'best_run': {},
            'worst_run': {}
        }
        
        # Find best and worst runs based on F1 score
        f1_scores = [r.get('f1', 0) for r in results]
        best_idx = np.argmax(f1_scores)
        worst_idx = np.argmin(f1_scores)
        
        summary['best_run'] = {
            'run_index': best_idx,
            'f1_score': f1_scores[best_idx],
            'metrics': results[best_idx]
        }
        
        summary['worst_run'] = {
            'run_index': worst_idx,
            'f1_score': f1_scores[worst_idx],
            'metrics': results[worst_idx]
        }
        
        return summary
    
    def export_metrics(self, results: List[Dict[str, float]], 
                      output_path: str, format: str = 'csv') -> str:
        """
        Export metrics to file.
        
        Args:
            results: List of metric dictionaries
            output_path: Path to save the file
            format: Output format ('csv', 'json', 'excel')
            
        Returns:
            Path to the exported file
        """
        if format == 'csv':
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        elif format == 'json':
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format == 'excel':
            df = pd.DataFrame(results)
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return output_path


# Utility functions for backward compatibility
def calculate_metrics(prods: List[float], labels: List[int]) -> Tuple[float, float, float, float, float]:
    """
    Calculate basic metrics (backward compatibility function).
    
    Args:
        prods: Model predictions (probabilities)
        labels: Ground truth labels
        
    Returns:
        Tuple of (f1, accuracy, precision, auc, mcc)
    """
    calculator = MetricsCalculator()
    metrics = calculator.calculate_metrics(prods, labels)
    
    return (
        metrics['f1'],
        metrics['accuracy'],
        metrics['precision'],
        metrics['auc'],
        metrics['mcc']
    )

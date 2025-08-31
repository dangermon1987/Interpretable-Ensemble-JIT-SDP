"""
Experiment runner for JIT-SPD model training and evaluation.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jit_spd_model import (
    JITSPDModel, ModelConfig, TrainingConfig,
    JITSPDDataset, FeatureProcessor, TrainingManager
)
from .metrics import MetricsCalculator
from .reporting import ExperimentReporter


class ExperimentRunner:
    """
    Runs experiments for the JIT-SPD model.
    
    Code Reference: new_effort/model_training_v4.py (lines 1-397)
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the experiment runner.
        
        Args:
            config: Training configuration (uses default if None)
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize components
        self.model = None
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.feature_processor = None
        self.training_manager = None
        
        # Experiment state
        self.experiment_results = []
        self.current_experiment = None
    
    def setup_experiment(self, experiment_config: Dict[str, Any]) -> None:
        """
        Setup experiment with specific configuration.
        
        Args:
            experiment_config: Configuration for the experiment
        """
        print("Setting up experiment...")
        
        # Create model configuration
        model_config = ModelConfig(
            with_graph_features=experiment_config.get('with_graph_features', True),
            with_text_features=experiment_config.get('with_text_features', True),
            with_manual_features=experiment_config.get('with_manual_features', True),
            manual_size=len(experiment_config.get('manual_features_columns', [])),
            graph_edges=experiment_config.get('graph_edges', ["CALL", "CFG", "PDG"])
        )
        
        # Create model
        self.model = JITSPDModel(model_config)
        
        # Load datasets
        self._load_datasets(model_config, experiment_config)
        
        # Setup feature processor
        self.feature_processor = FeatureProcessor(self.config)
        
        # Setup training manager
        self.training_manager = TrainingManager(self.config)
        self.training_manager.setup_training(
            self.model, 
            self.train_dataset, 
            self.valid_dataset, 
            self.test_dataset
        )
        
        print("Experiment setup complete!")
    
    def _load_datasets(self, model_config: ModelConfig, experiment_config: Dict[str, Any]) -> None:
        """
        Load and prepare datasets.
        
        Args:
            model_config: Model configuration
            experiment_config: Experiment configuration
        """
        print("Loading datasets...")
        
        # Load training dataset
        self.train_dataset = JITSPDDataset(
            data_dir=self.config.data_dir,
            projects=self.config.project_list,
            data_type='train',
            config=model_config
        )
        
        # Load validation dataset
        self.valid_dataset = JITSPDDataset(
            data_dir=self.config.data_dir,
            projects=self.config.project_list,
            data_type='validation',
            config=model_config
        )
        
        # Load test dataset
        self.test_dataset = JITSPDDataset(
            data_dir=self.config.data_dir,
            projects=self.config.project_list,
            data_type='test',
            config=model_config
        )
        
        # Set dataset configuration
        manual_features = experiment_config.get('manual_features_columns', [])
        graph_edges = experiment_config.get('graph_edges', ["CALL", "CFG", "PDG"])
        
        self.train_dataset.set_config(manual_features, graph_edges)
        self.valid_dataset.set_config(manual_features, graph_edges)
        self.test_dataset.set_config(manual_features, graph_edges)
        
        print(f"Datasets loaded: Train={len(self.train_dataset)}, "
              f"Valid={len(self.valid_dataset)}, Test={len(self.test_dataset)}")
    
    def run_experiment(self, experiment_idx: int, experiment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment.
        
        Args:
            experiment_idx: Index of the experiment
            experiment_config: Configuration for the experiment
            
        Returns:
            Dictionary containing experiment results
        """
        print(f"\n{'='*50}")
        print(f"Starting Experiment {experiment_idx}")
        print(f"{'='*50}")
        
        # Setup experiment
        self.setup_experiment(experiment_config)
        
        # Prepare text embeddings
        print("Preparing text embeddings...")
        train_text_embeddings, valid_text_embeddings, test_text_embeddings = (
            self.feature_processor.prepare_text_embeddings(
                self.train_dataset, self.valid_dataset, self.test_dataset
            )
        )
        
        # Store embeddings in training manager
        self.training_manager.train_text_embeddings = train_text_embeddings
        self.training_manager.valid_text_embeddings = valid_text_embeddings
        self.training_manager.test_text_embeddings = test_text_embeddings
        
        # Training loop
        best_metrics = {}
        early_stop = False
        
        for epoch in range(self.config.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            
            # Training
            train_metrics = self.training_manager.train_epoch(epoch)
            
            # Validation
            val_metrics = self.training_manager.evaluate(
                self.valid_dataset, valid_text_embeddings, "validation"
            )
            
            # Test evaluation
            test_metrics = self.training_manager.evaluate(
                self.test_dataset, test_text_embeddings, "test"
            )
            
            # Update scheduler
            self.training_manager.update_scheduler(val_metrics)
            
            # Check for best performance
            current_score = val_metrics['f1'] + val_metrics['auc']
            if not best_metrics or current_score > best_metrics.get('score', 0):
                best_metrics = {
                    'score': current_score,
                    'epoch': epoch,
                    'val_metrics': val_metrics.copy(),
                    'test_metrics': test_metrics.copy()
                }
                
                # Save best checkpoint
                self.training_manager.save_checkpoint(epoch, val_metrics, is_best=True)
            
            # Save regular checkpoint
            self.training_manager.save_checkpoint(epoch, val_metrics, is_best=False)
            
            # Log progress
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            print(f"Test F1: {test_metrics['f1']:.4f}, Test AUC: {test_metrics['auc']:.4f}")
            
            # Check early stopping
            if self.config.use_early_stopping:
                early_stop = self._check_early_stopping(epoch, val_metrics)
                if early_stop:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        # Final evaluation with best model
        final_results = self._final_evaluation(best_metrics, valid_text_embeddings, test_text_embeddings)
        
        # Store results
        experiment_result = {
            'experiment_idx': experiment_idx,
            'config': experiment_config,
            'best_metrics': best_metrics,
            'final_results': final_results,
            'early_stop': early_stop,
            'total_epochs': epoch + 1
        }
        
        self.experiment_results.append(experiment_result)
        
        # Cleanup
        self.training_manager.close()
        
        return experiment_result
    
    def _check_early_stopping(self, epoch: int, val_metrics: Dict[str, float]) -> bool:
        """
        Check if training should stop early.
        
        Args:
            epoch: Current epoch
            val_metrics: Validation metrics
            
        Returns:
            True if training should stop, False otherwise
        """
        # Simple early stopping based on F1 + AUC
        current_score = val_metrics['f1'] + val_metrics['auc']
        
        if not hasattr(self, '_best_score'):
            self._best_score = current_score
            self._patience_counter = 0
            return False
        
        if current_score > self._best_score:
            self._best_score = current_score
            self._patience_counter = 0
        else:
            self._patience_counter += 1
        
        return self._patience_counter >= self.config.patience
    
    def _final_evaluation(self, best_metrics: Dict[str, Any], 
                         valid_text_embeddings: Dict, test_text_embeddings: Dict) -> Dict[str, Any]:
        """
        Perform final evaluation with the best model.
        
        Args:
            best_metrics: Best metrics from training
            valid_text_embeddings: Validation text embeddings
            test_text_embeddings: Test text embeddings
            
        Returns:
            Dictionary containing final evaluation results
        """
        print("Performing final evaluation...")
        
        # Load best model
        best_epoch = best_metrics['epoch']
        checkpoint_path = os.path.join(
            self.config.result_path, 
            f'checkpoint_epoch_{best_epoch}.pt'
        )
        
        if os.path.exists(checkpoint_path):
            self.training_manager.load_checkpoint(checkpoint_path)
        
        # Final validation
        final_val_metrics = self.training_manager.evaluate(
            self.valid_dataset, valid_text_embeddings, "validation"
        )
        
        # Final test
        final_test_metrics = self.training_manager.evaluate(
            self.test_dataset, test_text_embeddings, "test"
        )
        
        return {
            'validation': final_val_metrics,
            'test': final_test_metrics
        }
    
    def run_multiple_experiments(self, experiment_configs: List[Dict[str, Any]], 
                                num_runs: int = 10) -> List[Dict[str, Any]]:
        """
        Run multiple experiments with different configurations.
        
        Args:
            experiment_configs: List of experiment configurations
            num_runs: Number of runs per configuration
            
        Returns:
            List of experiment results
        """
        all_results = []
        
        for config in experiment_configs:
            print(f"\n{'='*60}")
            print(f"Running experiments with configuration: {config.get('name', 'Unnamed')}")
            print(f"{'='*60}")
            
            config_results = []
            
            for run in range(num_runs):
                print(f"\nRun {run + 1}/{num_runs}")
                
                # Set random seed for reproducibility
                random.seed(42 + run)
                np.random.seed(42 + run)
                torch.manual_seed(42 + run)
                
                # Run experiment
                result = self.run_experiment(run, config)
                config_results.append(result)
            
            # Analyze results for this configuration
            config_analysis = self._analyze_config_results(config_results)
            all_results.extend(config_results)
            
            print(f"\nConfiguration Results Summary:")
            print(f"Average Val F1: {config_analysis['avg_val_f1']:.4f}")
            print(f"Average Val AUC: {config_analysis['avg_val_auc']:.4f}")
            print(f"Average Test F1: {config_analysis['avg_test_f1']:.4f}")
            print(f"Average Test AUC: {config_analysis['avg_test_auc']:.4f}")
        
        return all_results
    
    def _analyze_config_results(self, config_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze results for a specific configuration.
        
        Args:
            config_results: List of results for a configuration
            
        Returns:
            Dictionary containing analysis results
        """
        val_f1_scores = [r['best_metrics']['val_metrics']['f1'] for r in config_results]
        val_auc_scores = [r['best_metrics']['val_metrics']['auc'] for r in config_results]
        test_f1_scores = [r['best_metrics']['test_metrics']['f1'] for r in config_results]
        test_auc_scores = [r['best_metrics']['test_metrics']['auc'] for r in config_results]
        
        return {
            'avg_val_f1': np.mean(val_f1_scores),
            'std_val_f1': np.std(val_f1_scores),
            'avg_val_auc': np.mean(val_auc_scores),
            'std_val_auc': np.std(val_auc_scores),
            'avg_test_f1': np.mean(test_f1_scores),
            'std_test_f1': np.std(test_f1_scores),
            'avg_test_auc': np.mean(test_auc_scores),
            'std_test_auc': np.std(test_auc_scores)
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive experiment report.
        
        Args:
            output_path: Path to save the report (optional)
            
        Returns:
            Path to the generated report
        """
        if not self.experiment_results:
            print("No experiment results to report.")
            return ""
        
        reporter = ExperimentReporter()
        report_path = reporter.generate_report(
            self.experiment_results, 
            output_path or self.config.result_path
        )
        
        print(f"Experiment report generated: {report_path}")
        return report_path


def main():
    """Main function for running experiments."""
    parser = argparse.ArgumentParser(description='Run JIT-SPD model experiments')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate'], help='Experiment mode')
    parser.add_argument('--result_location', type=str, default='', 
                       help='Result location')
    parser.add_argument('--desc', type=str, default='', 
                       help='Experiment description')
    parser.add_argument('--num_runs', type=int, default=10, 
                       help='Number of runs per configuration')
    
    args = parser.parse_args()
    
    # Create experiment runner
    runner = ExperimentRunner()
    
    # Define experiment configurations
    experiment_configs = [
        {
            'name': 'Full Model',
            'with_graph_features': True,
            'with_text_features': True,
            'with_manual_features': True,
            'graph_edges': ["CALL", "CFG", "PDG"],
            'manual_features_columns': [
                'current_AST_blockstatement', 'delta_AST_memberreference',
                'parent_SM_class_nos_median', 'current_SM_method_nii_sum',
                'current_SM_method_nos_max', 'current_SM_method_mims_median',
                'current_SM_class_nos_median', 'current_SM_class_tlloc_min',
                'delta_SM_method_tloc_sum', 'parent_SM_class_nos_stdev'
            ]
        },
        {
            'name': 'Graph Only',
            'with_graph_features': True,
            'with_text_features': False,
            'with_manual_features': False,
            'graph_edges': ["CALL", "CFG", "PDG"],
            'manual_features_columns': []
        },
        {
            'name': 'Text Only',
            'with_graph_features': False,
            'with_text_features': True,
            'with_manual_features': False,
            'graph_edges': [],
            'manual_features_columns': []
        }
    ]
    
    if args.mode == 'train':
        # Run experiments
        results = runner.run_multiple_experiments(experiment_configs, args.num_runs)
        
        # Generate report
        runner.generate_report()
        
        print(f"\nCompleted {len(results)} experiments!")
        
    elif args.mode == 'evaluate':
        # Load and evaluate existing models
        print("Evaluation mode not yet implemented.")
        pass


if __name__ == '__main__':
    import torch
    main()

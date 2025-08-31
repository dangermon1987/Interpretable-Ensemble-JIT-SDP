"""
Experiment reporting utilities for JIT-SPD model.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime


class ExperimentReporter:
    """
    Generates comprehensive experiment reports.
    """
    
    def __init__(self):
        """Initialize the experiment reporter."""
        pass
    
    def generate_report(self, experiment_results: List[Dict[str, Any]], 
                       output_path: str) -> str:
        """
        Generate a comprehensive experiment report.
        
        Args:
            experiment_results: List of experiment results
            output_path: Directory to save the report
            
        Returns:
            Path to the generated report
        """
        if not experiment_results:
            print("No experiment results to report.")
            return ""
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate different report formats
        report_files = []
        
        # 1. Summary report
        summary_path = os.path.join(output_path, f"experiment_summary_{timestamp}.txt")
        self._generate_summary_report(experiment_results, summary_path)
        report_files.append(summary_path)
        
        # 2. Detailed results CSV
        csv_path = os.path.join(output_path, f"experiment_results_{timestamp}.csv")
        self._generate_csv_report(experiment_results, csv_path)
        report_files.append(csv_path)
        
        # 3. JSON report
        json_path = os.path.join(output_path, f"experiment_results_{timestamp}.json")
        self._generate_json_report(experiment_results, json_path)
        report_files.append(json_path)
        
        # 4. Configuration comparison
        config_path = os.path.join(output_path, f"configuration_comparison_{timestamp}.txt")
        self._generate_config_comparison(experiment_results, config_path)
        report_files.append(config_path)
        
        # 5. Performance analysis
        performance_path = os.path.join(output_path, f"performance_analysis_{timestamp}.txt")
        self._generate_performance_analysis(experiment_results, performance_path)
        report_files.append(performance_path)
        
        print(f"Generated {len(report_files)} report files in {output_path}")
        return output_path
    
    def _generate_summary_report(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Generate a summary report.
        
        Args:
            results: Experiment results
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("JIT-SPD Model Experiment Summary Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall statistics
            f.write(f"Total Experiments: {len(results)}\n")
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Group by configuration
            config_groups = {}
            for result in results:
                config_name = result['config'].get('name', 'Unnamed')
                if config_name not in config_groups:
                    config_groups[config_name] = []
                config_groups[config_name].append(result)
            
            # Summary for each configuration
            for config_name, config_results in config_groups.items():
                f.write(f"Configuration: {config_name}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Number of Runs: {len(config_results)}\n")
                
                # Calculate average metrics
                val_f1_scores = [r['best_metrics']['val_metrics']['f1'] for r in config_results]
                val_auc_scores = [r['best_metrics']['val_metrics']['auc'] for r in config_results]
                test_f1_scores = [r['best_metrics']['test_metrics']['f1'] for r in config_results]
                test_auc_scores = [r['best_metrics']['test_metrics']['auc'] for r in config_results]
                
                f.write(f"Average Validation F1: {np.mean(val_f1_scores):.4f} ± {np.std(val_f1_scores):.4f}\n")
                f.write(f"Average Validation AUC: {np.mean(val_auc_scores):.4f} ± {np.std(val_auc_scores):.4f}\n")
                f.write(f"Average Test F1: {np.mean(test_f1_scores):.4f} ± {np.std(test_f1_scores):.4f}\n")
                f.write(f"Average Test AUC: {np.mean(test_auc_scores):.4f} ± {np.std(test_auc_scores):.4f}\n")
                
                # Best run
                best_run = max(config_results, key=lambda x: x['best_metrics']['score'])
                f.write(f"Best Run: {best_run['experiment_idx']} (Score: {best_run['best_metrics']['score']:.4f})\n")
                f.write("\n")
    
    def _generate_csv_report(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Generate CSV report with detailed results.
        
        Args:
            results: Experiment results
            output_path: Path to save the CSV
        """
        # Flatten results for CSV
        csv_data = []
        
        for result in results:
            row = {
                'experiment_idx': result['experiment_idx'],
                'config_name': result['config'].get('name', 'Unnamed'),
                'with_graph_features': result['config'].get('with_graph_features', False),
                'with_text_features': result['config'].get('with_text_features', False),
                'with_manual_features': result['config'].get('with_manual_features', False),
                'graph_edges': ','.join(result['config'].get('graph_edges', [])),
                'manual_features_count': len(result['config'].get('manual_features_columns', [])),
                'best_epoch': result['best_metrics']['epoch'],
                'best_score': result['best_metrics']['score'],
                'val_f1': result['best_metrics']['val_metrics']['f1'],
                'val_accuracy': result['best_metrics']['val_metrics']['accuracy'],
                'val_precision': result['best_metrics']['val_metrics']['precision'],
                'val_auc': result['best_metrics']['val_metrics']['auc'],
                'val_mcc': result['best_metrics']['val_metrics']['mcc'],
                'test_f1': result['best_metrics']['test_metrics']['f1'],
                'test_accuracy': result['best_metrics']['test_metrics']['accuracy'],
                'test_precision': result['best_metrics']['test_metrics']['precision'],
                'test_auc': result['best_metrics']['test_metrics']['auc'],
                'test_mcc': result['best_metrics']['test_metrics']['mcc'],
                'early_stop': result['early_stop'],
                'total_epochs': result['total_epochs']
            }
            csv_data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(output_path, index=False)
    
    def _generate_json_report(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Generate JSON report with full results.
        
        Args:
            results: Experiment results
            output_path: Path to save the JSON
        """
        # Prepare data for JSON (remove non-serializable objects)
        json_data = []
        
        for result in results:
            # Deep copy and clean
            clean_result = self._clean_for_json(result)
            json_data.append(clean_result)
        
        # Save JSON
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
    
    def _clean_for_json(self, obj: Any) -> Any:
        """
        Clean object for JSON serialization.
        
        Args:
            obj: Object to clean
            
        Returns:
            Cleaned object
        """
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._clean_for_json(obj.__dict__)
        else:
            return obj
    
    def _generate_config_comparison(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Generate configuration comparison report.
        
        Args:
            results: Experiment results
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("Configuration Comparison Report\n")
            f.write("=" * 40 + "\n\n")
            
            # Group by configuration
            config_groups = {}
            for result in results:
                config_name = result['config'].get('name', 'Unnamed')
                if config_name not in config_groups:
                    config_groups[config_name] = []
                config_groups[config_name].append(result)
            
            # Compare configurations
            f.write("Configuration Details:\n")
            f.write("-" * 30 + "\n")
            
            for config_name, config_results in config_groups.items():
                f.write(f"\n{config_name}:\n")
                config = config_results[0]['config']
                
                f.write(f"  Graph Features: {config.get('with_graph_features', False)}\n")
                f.write(f"  Text Features: {config.get('with_text_features', False)}\n")
                f.write(f"  Manual Features: {config.get('with_manual_features', False)}\n")
                f.write(f"  Graph Edges: {config.get('graph_edges', [])}\n")
                f.write(f"  Manual Features Count: {len(config.get('manual_features_columns', []))}\n")
                
                # Performance summary
                val_f1_scores = [r['best_metrics']['val_metrics']['f1'] for r in config_results]
                val_auc_scores = [r['best_metrics']['val_metrics']['auc'] for r in config_results]
                test_f1_scores = [r['best_metrics']['test_metrics']['f1'] for r in config_results]
                test_auc_scores = [r['best_metrics']['test_metrics']['auc'] for r in config_results]
                
                f.write(f"  Validation F1: {np.mean(val_f1_scores):.4f} ± {np.std(val_f1_scores):.4f}\n")
                f.write(f"  Validation AUC: {np.mean(val_auc_scores):.4f} ± {np.std(val_auc_scores):.4f}\n")
                f.write(f"  Test F1: {np.mean(test_f1_scores):.4f} ± {np.std(test_f1_scores):.4f}\n")
                f.write(f"  Test AUC: {np.mean(test_auc_scores):.4f} ± {np.std(test_auc_scores):.4f}\n")
    
    def _generate_performance_analysis(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Generate performance analysis report.
        
        Args:
            results: Experiment results
            output_path: Path to save the report
        """
        with open(output_path, 'w') as f:
            f.write("Performance Analysis Report\n")
            f.write("=" * 30 + "\n\n")
            
            # Overall performance statistics
            all_val_f1 = [r['best_metrics']['val_metrics']['f1'] for r in results]
            all_val_auc = [r['best_metrics']['val_metrics']['auc'] for r in results]
            all_test_f1 = [r['best_metrics']['test_metrics']['f1'] for r in results]
            all_test_auc = [r['best_metrics']['test_metrics']['auc'] for r in results]
            
            f.write("Overall Performance:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Validation F1: {np.mean(all_val_f1):.4f} ± {np.std(all_val_f1):.4f}\n")
            f.write(f"Validation AUC: {np.mean(all_val_auc):.4f} ± {np.std(all_val_auc):.4f}\n")
            f.write(f"Test F1: {np.mean(all_test_f1):.4f} ± {np.std(all_test_f1):.4f}\n")
            f.write(f"Test AUC: {np.mean(all_test_auc):.4f} ± {np.std(all_test_auc):.4f}\n\n")
            
            # Best and worst performers
            best_val_idx = np.argmax(all_val_f1)
            worst_val_idx = np.argmin(all_val_f1)
            best_test_idx = np.argmax(all_test_f1)
            worst_test_idx = np.argmin(all_test_f1)
            
            f.write("Best Performers:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Best Validation F1: {results[best_val_idx]['config'].get('name', 'Unnamed')} "
                   f"(Run {results[best_val_idx]['experiment_idx']}): {all_val_f1[best_val_idx]:.4f}\n")
            f.write(f"Best Test F1: {results[best_test_idx]['config'].get('name', 'Unnamed')} "
                   f"(Run {results[best_test_idx]['experiment_idx']}): {all_test_f1[best_test_idx]:.4f}\n\n")
            
            f.write("Worst Performers:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Worst Validation F1: {results[worst_val_idx]['config'].get('name', 'Unnamed')} "
                   f"(Run {results[worst_val_idx]['experiment_idx']}): {all_val_f1[worst_val_idx]:.4f}\n")
            f.write(f"Worst Test F1: {results[worst_test_idx]['config'].get('name', 'Unnamed')} "
                   f"(Run {results[worst_test_idx]['experiment_idx']}): {all_test_f1[worst_test_idx]:.4f}\n\n")
            
            # Early stopping analysis
            early_stop_count = sum(1 for r in results if r['early_stop'])
            f.write("Early Stopping Analysis:\n")
            f.write("-" * 25 + "\n")
            f.write(f"Experiments with Early Stopping: {early_stop_count}/{len(results)} "
                   f"({early_stop_count/len(results)*100:.1f}%)\n")
            
            # Epoch analysis
            all_epochs = [r['total_epochs'] for r in results]
            f.write(f"Average Epochs: {np.mean(all_epochs):.1f} ± {np.std(all_epochs):.1f}\n")
            f.write(f"Min Epochs: {min(all_epochs)}\n")
            f.write(f"Max Epochs: {max(all_epochs)}\n")
    
    def generate_latex_report(self, results: List[Dict[str, Any]], output_path: str) -> str:
        """
        Generate LaTeX report for academic papers.
        
        Args:
            results: Experiment results
            output_path: Path to save the LaTeX file
            
        Returns:
            Path to the generated LaTeX file
        """
        with open(output_path, 'w') as f:
            f.write("\\documentclass{article}\n")
            f.write("\\usepackage{booktabs}\n")
            f.write("\\usepackage{multirow}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage{float}\n\n")
            
            f.write("\\title{JIT-SPD Model Experiment Results}\n")
            f.write("\\author{Experiment Report}\n")
            f.write("\\date{\\today}\n\n")
            
            f.write("\\begin{document}\n")
            f.write("\\maketitle\n\n")
            
            # Abstract
            f.write("\\begin{abstract}\n")
            f.write("This report presents the experimental results for the JIT-SPD model, ")
            f.write("a multi-modal deep learning approach for software defect prediction. ")
            f.write(f"The experiments include {len(results)} runs across different configurations, ")
            f.write("evaluating the model's performance on validation and test datasets.\n")
            f.write("\\end{abstract}\n\n")
            
            # Results table
            f.write("\\section{Experimental Results}\n\n")
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance Metrics Across Configurations}\n")
            f.write("\\label{tab:results}\n")
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\toprule\n")
            f.write("Config & Val F1 & Val AUC & Test F1 & Test AUC & Epochs & Early Stop \\\\\n")
            f.write("\\midrule\n")
            
            # Group by configuration
            config_groups = {}
            for result in results:
                config_name = result['config'].get('name', 'Unnamed')
                if config_name not in config_groups:
                    config_groups[config_name] = []
                config_groups[config_name].append(result)
            
            for config_name, config_results in config_groups.items():
                # Calculate averages
                val_f1_scores = [r['best_metrics']['val_metrics']['f1'] for r in config_results]
                val_auc_scores = [r['best_metrics']['val_metrics']['auc'] for r in config_results]
                test_f1_scores = [r['best_metrics']['test_metrics']['f1'] for r in config_results]
                test_auc_scores = [r['best_metrics']['test_metrics']['auc'] for r in config_results]
                epochs = [r['total_epochs'] for r in config_results]
                early_stop = sum(1 for r in config_results if r['early_stop'])
                
                f.write(f"{config_name} & "
                       f"{np.mean(val_f1_scores):.3f}±{np.std(val_f1_scores):.3f} & "
                       f"{np.mean(val_auc_scores):.3f}±{np.std(val_auc_scores):.3f} & "
                       f"{np.mean(test_f1_scores):.3f}±{np.std(test_f1_scores):.3f} & "
                       f"{np.mean(test_auc_scores):.3f}±{np.std(test_auc_scores):.3f} & "
                       f"{np.mean(epochs):.1f}±{np.std(epochs):.1f} & "
                       f"{early_stop}/{len(config_results)} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            f.write("\\end{document}\n")
        
        return output_path


# Import numpy for calculations
import numpy as np

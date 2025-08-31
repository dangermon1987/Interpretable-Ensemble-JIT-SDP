# JIT-SPD Experiments Package

A comprehensive package for running experiments and evaluating the JIT-SPD (Just-In-Time Software Defect Prediction) model.

## 🚀 Features

- **Experiment Runner**: Automated experiment execution with multiple configurations
- **Model Evaluation**: Comprehensive model performance assessment
- **Metrics Calculation**: Advanced evaluation metrics and statistical analysis
- **Report Generation**: Multiple report formats (TXT, CSV, JSON, LaTeX)
- **Configuration Management**: Flexible experiment configuration system
- **Statistical Analysis**: Confidence intervals, significance testing, and performance comparison

## 📦 Installation

### Prerequisites

- Python 3.8+
- JIT-SPD model package
- Required dependencies (see requirements.txt)

### Install Package

```bash
# Install dependencies
pip install -r requirements.txt

# Install JIT-SPD model package first
# pip install -e ../jit-spd-model/
```

## 🏗️ Architecture

The experiments package provides a modular architecture for running and analyzing experiments:

```
Experiments Package
├── ExperimentRunner      # Main experiment orchestration
├── ModelEvaluator        # Model evaluation utilities
├── MetricsCalculator     # Performance metrics calculation
└── ExperimentReporter    # Report generation
```

## 🚀 Quick Start

### Basic Experiment Execution

```python
from experiments import ExperimentRunner
from jit_spd_model import TrainingConfig

# Create configuration
config = TrainingConfig(
    batch_size=64,
    learning_rate=1e-2,
    max_epochs=200
)

# Create experiment runner
runner = ExperimentRunner(config)

# Define experiment configurations
experiment_configs = [
    {
        'name': 'Full Model',
        'with_graph_features': True,
        'with_text_features': True,
        'with_manual_features': True,
        'graph_edges': ["CALL", "CFG", "PDG"],
        'manual_features_columns': ['la', 'ld', 'nf', 'entropy']
    },
    {
        'name': 'Graph Only',
        'with_graph_features': True,
        'with_text_features': False,
        'with_manual_features': False,
        'graph_edges': ["CALL", "CFG", "PDG"],
        'manual_features_columns': []
    }
]

# Run experiments
results = runner.run_multiple_experiments(experiment_configs, num_runs=10)

# Generate reports
runner.generate_report()
```

### Model Evaluation

```python
from experiments import ModelEvaluator

# Create evaluator
evaluator = ModelEvaluator(
    model_path="/path/to/model.pt",
    config_path="/path/to/config.pt"
)

# Evaluate model
evaluation_results = evaluator.evaluate_model(test_dataset, text_embeddings)

# Generate evaluation report
evaluator.generate_evaluation_report(evaluation_results)

# Export predictions
evaluator.export_predictions(evaluation_results, "predictions.csv", format='csv')
```

### Metrics Analysis

```python
from experiments import MetricsCalculator

# Create calculator
calculator = MetricsCalculator()

# Calculate metrics
metrics = calculator.calculate_metrics(predictions, labels)

# Calculate confidence intervals
confidence_intervals = calculator.calculate_confidence_intervals(results_list)

# Generate classification report
report = calculator.generate_classification_report(predictions, labels)

# Compare models
comparison = calculator.compare_models({
    'Model A': results_a,
    'Model B': results_b
})
```

## 📊 Experiment Configurations

### Configuration Structure

Each experiment configuration defines:

```python
experiment_config = {
    'name': 'Configuration Name',
    'with_graph_features': True,           # Use graph neural networks
    'with_text_features': True,            # Use CodeBERT embeddings
    'with_manual_features': True,          # Use software metrics
    'graph_edges': ["CALL", "CFG", "PDG"], # Edge types to include
    'manual_features_columns': [           # Specific features to use
        'la', 'ld', 'nf', 'entropy'
    ]
}
```

### Predefined Configurations

The package includes several predefined configurations:

1. **Full Model**: Complete multi-modal approach
2. **Graph Only**: Graph neural networks only
3. **Text Only**: CodeBERT embeddings only
4. **Manual Only**: Software metrics only
5. **Hybrid**: Custom combinations

## 📈 Metrics and Evaluation

### Performance Metrics

The package calculates comprehensive metrics:

- **Basic Metrics**: F1, Accuracy, Precision, Recall
- **Advanced Metrics**: AUC, MCC, Specificity, Sensitivity
- **Statistical Metrics**: Confidence intervals, Standard error, Coefficient of variation

### Evaluation Pipeline

1. **Model Loading**: Load trained model and configuration
2. **Data Processing**: Prepare test dataset and embeddings
3. **Inference**: Generate predictions and extract embeddings
4. **Metrics Calculation**: Compute all performance metrics
5. **Analysis**: Analyze predictions, confidence, and embeddings
6. **Reporting**: Generate comprehensive evaluation reports

## 📋 Report Generation

### Report Types

The package generates multiple report formats:

1. **Summary Report** (TXT): High-level experiment overview
2. **Detailed Results** (CSV): Comprehensive metrics for each run
3. **JSON Report**: Complete results in structured format
4. **Configuration Comparison**: Side-by-side configuration analysis
5. **Performance Analysis**: Statistical analysis and insights
6. **LaTeX Report**: Academic paper-ready format

### Report Content

Each report includes:

- **Experiment Overview**: Configuration details and run statistics
- **Performance Metrics**: All calculated metrics with confidence intervals
- **Statistical Analysis**: Best/worst performers, early stopping analysis
- **Configuration Comparison**: Performance across different setups
- **Visualizations**: Charts and graphs (when applicable)

## 🔧 Advanced Usage

### Custom Experiment Configurations

```python
# Custom feature selection
custom_config = {
    'name': 'Custom Features',
    'with_graph_features': True,
    'with_text_features': True,
    'with_manual_features': True,
    'graph_edges': ["CALL", "CFG"],  # Limited edge types
    'manual_features_columns': [
        'la', 'ld', 'nf', 'ns', 'nd',  # Kamei features
        'current_AST_methodinvocation',  # AST features
        'current_SM_method_mi_avg'       # Software metrics
    ]
}

# Run custom experiment
result = runner.run_experiment(0, custom_config)
```

### Batch Experiment Execution

```python
# Run multiple configurations with different parameters
configurations = []

# Vary learning rates
for lr in [1e-3, 1e-2, 1e-1]:
    config = TrainingConfig(learning_rate=lr)
    runner = ExperimentRunner(config)
    
    for exp_config in experiment_configs:
        result = runner.run_experiment(0, exp_config)
        configurations.append(result)

# Analyze all results
all_results = runner.experiment_results
runner.generate_report()
```

### Statistical Analysis

```python
from experiments import MetricsCalculator

calculator = MetricsCalculator()

# Calculate confidence intervals
ci_data = calculator.calculate_confidence_intervals(results, confidence_level=0.95)

# Generate summary statistics
summary = calculator.generate_summary_statistics(results)

# Export metrics
calculator.export_metrics(results, "metrics.csv", format='csv')

# Compare multiple models
comparison = calculator.compare_models({
    'Baseline': baseline_results,
    'Improved': improved_results,
    'Best': best_results
})
```

## 📊 Output and Results

### Experiment Results Structure

Each experiment produces:

```python
experiment_result = {
    'experiment_idx': 0,
    'config': experiment_config,
    'best_metrics': {
        'score': 0.85,
        'epoch': 45,
        'val_metrics': {'f1': 0.82, 'auc': 0.88, ...},
        'test_metrics': {'f1': 0.80, 'auc': 0.86, ...}
    },
    'final_results': {
        'validation': val_metrics,
        'test': test_metrics
    },
    'early_stop': True,
    'total_epochs': 45
}
```

### Report Files

Generated reports include:

- `experiment_summary_YYYYMMDD_HHMMSS.txt`
- `experiment_results_YYYYMMDD_HHMMSS.csv`
- `experiment_results_YYYYMMDD_HHMMSS.json`
- `configuration_comparison_YYYYMMDD_HHMMSS.txt`
- `performance_analysis_YYYYMMDD_HHMMSS.txt`

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure JIT-SPD model package is installed
2. **Configuration Issues**: Verify experiment configuration format
3. **Data Loading**: Check dataset paths and format
4. **Memory Issues**: Reduce batch size or feature dimensions

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set specific logger
logger = logging.getLogger('experiments')
logger.setLevel(logging.DEBUG)
```

## 📚 API Reference

### Core Classes

- **ExperimentRunner**: Main experiment orchestration
- **ModelEvaluator**: Model evaluation and analysis
- **MetricsCalculator**: Performance metrics calculation
- **ExperimentReporter**: Report generation utilities

### Key Methods

- `run_experiment()`: Execute single experiment
- `run_multiple_experiments()`: Execute multiple experiments
- `evaluate_model()`: Evaluate trained model
- `generate_report()`: Generate comprehensive reports
- `calculate_metrics()`: Compute performance metrics

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd experiments

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Format code
black experiments/
```
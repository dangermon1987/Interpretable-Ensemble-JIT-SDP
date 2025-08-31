# JIT-SPD Model Package

A comprehensive deep learning package for Just-In-Time Software Defect Prediction using multi-modal fusion of graph neural networks, natural language processing, and software metrics.

## 🚀 Features

- **Multi-Modal Architecture**: Combines graph neural networks, CodeBERT embeddings, and traditional software metrics
- **Graph Processing**: Efficient processing of Code Property Graphs (CPGs) using GraphSAGE
- **Text Analysis**: Advanced commit message and code change analysis using CodeBERT
- **Feature Fusion**: Intelligent combination of multiple data modalities
- **Training Management**: Comprehensive training pipeline with early stopping and learning rate scheduling
- **Data Processing**: Efficient lazy loading and caching for large-scale datasets
- **Interpretability**: Attention mechanisms and feature importance analysis

## 📦 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Install Package

```bash
# Clone the repository
git clone <repository-url>
cd jit-spd-model

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## 🏗️ Architecture

The JIT-SPD model follows a multi-modal fusion architecture:

```
Input Data
├── Graph Data (CPGs)
├── Text Data (Commit Messages + Code Changes)
└── Feature Data (Software Metrics)

Processing Layers
├── Graph Neural Network (GraphSAGE)
├── Text Encoder (CodeBERT)
└── Feature Processor

Fusion Layer
├── Attention Pooling
├── Multi-Modal Concatenation
└── Normalization

Output
└── Defect Prediction (Binary Classification)
```

## 🚀 Quick Start

### Basic Usage

```python
from jit_spd_model import JITSPDModel, ModelConfig, TrainingConfig
from jit_spd_model.data import JITSPDDataset
from jit_spd_model.features import FeatureProcessor
from jit_spd_model.training import TrainingManager

# Initialize configuration
model_config = ModelConfig(
    with_graph_features=True,
    with_text_features=True,
    with_manual_features=True
)

training_config = TrainingConfig(
    batch_size=64,
    learning_rate=1e-2,
    max_epochs=200
)

# Create model
model = JITSPDModel(model_config)

# Load datasets
train_dataset = JITSPDDataset(
    data_dir="/path/to/data",
    data_type="train",
    config=model_config
)

valid_dataset = JITSPDDataset(
    data_dir="/path/to/data", 
    data_type="validation",
    config=model_config
)

# Setup training
training_manager = TrainingManager(training_config)
training_manager.setup_training(model, train_dataset, valid_dataset, None)

# Train model
for epoch in range(training_config.max_epochs):
    # Training
    train_metrics = training_manager.train_epoch(epoch)
    
    # Validation
    val_metrics = training_manager.evaluate(valid_dataset, text_embeddings, "validation")
    
    # Update scheduler
    training_manager.update_scheduler(val_metrics)
    
    # Save checkpoint
    is_best = val_metrics['f1'] + val_metrics['auc'] > best_score
    training_manager.save_checkpoint(epoch, val_metrics, is_best)
```

### Advanced Configuration

```python
# Custom model configuration
model_config = ModelConfig(
    node_feature_dimension=1024,  # Custom feature dimension
    out_channels=256,              # Custom output channels
    graph_edges=["CALL", "CFG", "PDG"],  # Selected edge types
    dropout_rate=0.2,              # Custom dropout
    manual_size=50                 # Custom manual feature size
)

# Custom training configuration
training_config = TrainingConfig(
    batch_size=32,                 # Smaller batch size
    learning_rate=5e-3,            # Lower learning rate
    weight_decay=1e-3,             # Higher weight decay
    patience=15,                   # More patience for early stopping
    use_tensorboard=True,          # Enable TensorBoard logging
    use_weighted_sampling=True     # Enable balanced sampling
)

# Environment variable configuration
import os
os.environ['JITSPD_BATCH_SIZE'] = '128'
os.environ['JITSPD_LEARNING_RATE'] = '1e-3'
os.environ['JITSPD_DATA_DIR'] = '/custom/data/path'

training_config = TrainingConfig.from_env()
```

## 📊 Data Format

### Graph Data (CPGs)

The package expects Code Property Graphs in PyTorch Geometric format:

```python
# Expected graph structure
graph = HeteroData(
    x=torch.tensor(...),           # Node features
    edge_index=torch.tensor(...),  # Edge indices
    edge_type=torch.tensor(...),   # Edge types
    # Additional attributes
)
```

### Text Data

Commit messages and code changes should be provided as:

```python
# Commit message
commit_message = "Fix bug in authentication logic"

# Code changes (dictionary format)
code_changes = {
    'added_code': ['if (user.isValid()) {', '    authenticate(user);', '}'],
    'removed_code': ['if (user != null) {', '    login(user);', '}']
}
```

### Feature Data

Software metrics should be provided as a pandas DataFrame with columns matching the configuration:

```python
# Example features
features = {
    'la': 10,           # Lines added
    'ld': 5,            # Lines deleted
    'nf': 2,            # Number of files
    'entropy': 0.8,     # Code entropy
    # ... additional metrics
}
```

## 🔧 Configuration

### Model Configuration

The `ModelConfig` class controls the model architecture:

```python
@dataclass
class ModelConfig:
    # Model dimensions
    node_feature_dimension: int = 896      # 768 (CodeBERT) + 128 (Node2Vec)
    out_channels: int = 128                # Graph embedding dimension
    in_channels: int = 897                 # Input feature dimension
    
    # Feature configuration
    with_manual_features: bool = True      # Use software metrics
    with_text_features: bool = True        # Use text embeddings
    with_graph_features: bool = True       # Use graph features
    
    # Graph processing
    graph_edges: List[str] = ["CALL", "CFG", "PDG"]  # Edge types to include
    
    # Architecture settings
    dropout_rate: float = 0.1              # Dropout rate
    activation: str = "gelu"               # Activation function
```

### Training Configuration

The `TrainingConfig` class controls the training process:

```python
@dataclass
class TrainingConfig:
    # Training parameters
    batch_size: int = 64                   # Batch size
    learning_rate: float = 1e-2            # Learning rate
    weight_decay: float = 1e-4             # Weight decay
    max_epochs: int = 200                  # Maximum epochs
    patience: int = 8                      # Early stopping patience
    
    # Data configuration
    data_dir: str = "/path/to/data"        # Data directory
    result_path: str = "/path/to/results"  # Results directory
    
    # Features
    use_weighted_sampling: bool = True     # Balanced sampling
    use_early_stopping: bool = True        # Early stopping
    use_lr_scheduling: bool = True         # Learning rate scheduling
    use_tensorboard: bool = True           # TensorBoard logging
```

## 📈 Training

### Training Pipeline

The training process follows this workflow:

1. **Data Preparation**: Load and preprocess datasets
2. **Feature Generation**: Generate text embeddings using CodeBERT
3. **Model Setup**: Initialize model, optimizer, and scheduler
4. **Training Loop**: Iterate through epochs with validation
5. **Checkpointing**: Save best models and training state
6. **Evaluation**: Comprehensive performance assessment

### Monitoring

Training progress can be monitored through:

- **Console Output**: Real-time metrics and progress
- **TensorBoard**: Loss curves, metrics, and embeddings
- **Checkpoints**: Regular model state saving
- **Logs**: Detailed training logs

### Early Stopping

The package includes intelligent early stopping:

```python
# Early stopping configuration
early_stopping = EarlyStopping(
    exp=experiment_path,
    meta=metadata,
    patience=8,                    # Wait 8 epochs
    min_delta=0.001               # Minimum improvement threshold
)

# In training loop
if early_stopping(model, optimizer, scheduler, epoch, 
                  val_loss, val_f1, val_mcc, val_auc):
    print("Early stopping triggered")
    break
```

## 🔍 Evaluation

### Metrics

The model provides comprehensive evaluation metrics:

- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **AUC**: Area under ROC curve
- **MCC**: Matthews Correlation Coefficient

### Evaluation Pipeline

```python
# Evaluate on test set
test_metrics = training_manager.evaluate(
    test_dataset, 
    test_text_embeddings, 
    "test"
)

print(f"Test F1: {test_metrics['f1']:.4f}")
print(f"Test AUC: {test_metrics['auc']:.4f}")
print(f"Test MCC: {test_metrics['mcc']:.4f}")
```

## 💾 Model Persistence

### Saving Models

```python
# Save model
model.save_model("/path/to/model.pt")

# Save checkpoint
training_manager.save_checkpoint(
    epoch=100,
    metrics=val_metrics,
    is_best=True
)
```

### Loading Models

```python
# Load model
model = JITSPDModel()
model.load_model("/path/to/model.pt")

# Load checkpoint
epoch = training_manager.load_checkpoint("/path/to/checkpoint.pt")
```

## 🔬 Interpretability

### Feature Importance

```python
# Get feature importance scores
importance_scores = model.get_feature_importance(data)

print("Graph feature importance:", importance_scores['graph'])
print("Text feature importance:", importance_scores['text'])
print("Manual feature importance:", importance_scores['manual'])
```

### Attention Weights

The model provides attention weights for both graph and text components:

```python
# Get attention weights
logits, graph_embedding, graph_attn_weights = model(data)

# Graph attention weights show which nodes are important
# Text attention weights show which tokens are important
```

## 🚀 Performance

### Computational Requirements

- **GPU Memory**: 2-4GB for batch size 64
- **CPU Memory**: 8-16GB for large datasets
- **Storage**: Variable based on dataset size

### Optimization Tips

1. **Batch Size**: Adjust based on available memory
2. **Feature Selection**: Use only relevant features
3. **Graph Filtering**: Limit edge types to essential ones
4. **Caching**: Enable feature caching for repeated access

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or feature dimensions
2. **Slow Training**: Check data loading and enable caching
3. **Poor Performance**: Verify data quality and feature relevance
4. **Model Not Converging**: Adjust learning rate and patience

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set specific logger
logger = logging.getLogger('jit_spd_model')
logger.setLevel(logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd jit-spd-model

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/

# Format code
black jit_spd_model/
```

## 📚 API Reference

### Core Classes

- `JITSPDModel`: Main model class
- `HomogeneousGraphConvolution`: Graph convolution layer
- `TrainingManager`: Training pipeline management
- `JITSPDDataset`: Dataset handling
- `FeatureProcessor`: Feature processing utilities

### Configuration Classes

- `ModelConfig`: Model architecture configuration
- `TrainingConfig`: Training process configuration

### Utility Classes

- `DataProcessor`: Data processing utilities
- `TextEmbeddingGenerator`: Text embedding generation
- `EarlyStopping`: Early stopping mechanism

# Interpretable Ensemble JIT-SDP Repository
## 🏗️ Repository Structure

```
Interpretable-Ensemble-JIT-SDP/
├── code/                           # Core implementation packages
│   ├── code_property_graph/        # CPG generation and processing
│   ├── jit-spd-model/             # Main JIT-SPD model implementation
│   ├── node_embeddings/           # Node embedding generation
│   └── experiments/               # Experiment framework and evaluation
├── data/                          # Dataset storage and management
│   ├── cpgs/                      # Generated Code Property Graphs
│   ├── enhanced_jit_defects4j/    # Enhanced Defects4J dataset
│   └── original_jit_defects4j/    # Original Defects4J dataset
└── results/                       # Experiment results and outputs
```

## 📦 Core Packages

### 1. Code Property Graph Package (`code/code_property_graph/`)

**Purpose**: Generates and processes Code Property Graphs from Java source code using the Joern static analysis tool.

**Key Features**:
- CPG generation from Java source code repositories
- GraphML export for analysis and visualization
- Cloud storage integration (Google Cloud Storage)
- Parallel processing for large-scale projects
- Code smell detection and complexity analysis

**Main Components**:
- `CPGGenerator`: Generates CPGs from source code
- `CPGProcessor`: Processes and exports CPGs to various formats
- `CPGAnalyzer`: Analyzes CPG structure and detects patterns
- `CPGConfig`: Manages configuration and settings

**Usage Example**:
```python
from code_property_graph import CPGGenerator, CPGProcessor

# Generate CPGs for a project
generator = CPGGenerator()
cpg_files = generator.generate_cpgs_for_project("commons-beanutils")

# Process CPGs to GraphML
processor = CPGProcessor()
graphml_files, zip_files = processor.process_and_compress_project("commons-beanutils")
```

### 2. JIT-SPD Model Package (`code/jit-spd-model/`)

**Purpose**: Implements the core multi-modal JIT-SPD model using PyTorch and PyTorch Geometric.

**Key Features**:
- Multi-modal architecture combining GNNs, CodeBERT, and software metrics
- Graph processing using GraphSAGE for CPG analysis
- Text analysis using CodeBERT for commit messages and code changes
- Feature fusion with attention mechanisms
- Comprehensive training pipeline with early stopping

**Main Components**:
- `JITSPDModel`: Main model class with multi-modal fusion
- `HomogeneousGraphConvolution`: Graph convolution layer
- `TrainingManager`: Training pipeline management
- `JITSPDDataset`: Dataset handling and preprocessing

**Usage Example**:
```python
from jit_spd_model import JITSPDModel, ModelConfig, TrainingConfig

# Initialize model
model_config = ModelConfig(
    with_graph_features=True,
    with_text_features=True,
    with_manual_features=True
)
model = JITSPDModel(model_config)

# Setup training
training_manager = TrainingManager(training_config)
training_manager.setup_training(model, train_dataset, valid_dataset, None)
```

### 3. Node Embeddings Package (`code/node_embeddings/`)

**Purpose**: Generates comprehensive node embeddings from Code Property Graphs using multiple embedding techniques.

**Key Features**:
- Multi-modal embeddings: Word2Vec, Node2Vec, and CodeBERT
- Intelligent k-distance filtering for graph processing
- Heterogeneous graph construction
- Batch processing for large graphs
- Parallel execution for multiple projects

**Main Components**:
- `EmbeddingGenerator`: Main orchestrator class
- `Word2VecEmbedder`: Semantic embeddings from node features
- `Node2VecEmbedder`: Structural embeddings from graph topology
- `CodeBERTEmbedder`: Advanced semantic embeddings
- `EmbeddingProcessor`: Embedding combination and graph construction

**Usage Example**:
```python
from node_embeddings import EmbeddingGenerator

# Initialize generator
generator = EmbeddingGenerator()

# Generate embeddings for a project
result = generator.generate_embeddings_for_project("commons-codec")

# Process multiple projects in parallel
all_results = generator.generate_embeddings_for_projects(parallel=True)
```

### 4. Experiments Package (`code/experiments/`)

**Purpose**: Provides a comprehensive framework for running experiments and evaluating JIT-SPD model performance.

**Key Features**:
- Automated experiment execution with multiple configurations
- Comprehensive model evaluation and metrics calculation
- Multiple report formats (TXT, CSV, JSON, LaTeX)
- Statistical analysis with confidence intervals
- Configuration comparison and performance analysis

**Main Components**:
- `ExperimentRunner`: Main experiment orchestration
- `ModelEvaluator`: Model evaluation utilities
- `MetricsCalculator`: Performance metrics calculation
- `ExperimentReporter`: Report generation

**Usage Example**:
```python
from experiments import ExperimentRunner

# Create experiment runner
runner = ExperimentRunner(config)

# Define experiment configurations
experiment_configs = [
    {
        'name': 'Full Model',
        'with_graph_features': True,
        'with_text_features': True,
        'with_manual_features': True
    },
    {
        'name': 'Graph Only',
        'with_graph_features': True,
        'with_text_features': False,
        'with_manual_features': False
    }
]

# Run experiments
results = runner.run_multiple_experiments(experiment_configs, num_runs=10)
```

## 📊 Data Management

### Dataset Structure

The repository works with the **Defects4J** dataset, which contains real-world Java bugs from open-source projects:

- **Original Defects4J**: Standard Defects4J dataset with bug information
- **Enhanced Defects4J**: Enhanced version with additional features and preprocessing
- **Generated CPGs**: Code Property Graphs generated from the source code

### Supported Projects

The system supports 20+ Java projects including:
- Apache Commons (codec, collections, compress, configuration, dbcp, digester, io, jcs, lang, math, net, scxml, validator, vfs)
- Apache Ant-IVY, BCEL, BeanUtils, Giraph, Gora, OpenNLP, Parquet-MR

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **PyTorch 1.12+**
3. **Joern** (for CPG generation)
4. **CUDA** (optional, for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Interpretable-Ensemble-JIT-SDP

# Install dependencies for each package
cd code/code_property_graph
pip install -r requirements.txt

cd ../jit-spd-model
pip install -r requirements.txt

cd ../node_embeddings
pip install -r requirements.txt

cd ../experiments
pip install -r requirements.txt
```

### Basic Workflow

1. **Generate Code Property Graphs**:
   ```bash
   cd code/code_property_graph
   python -m code_property_graph generate --project commons-beanutils
   ```

2. **Generate Node Embeddings**:
   ```python
   from node_embeddings import EmbeddingGenerator
   generator = EmbeddingGenerator()
   result = generator.generate_embeddings_for_project("commons-beanutils")
   ```

3. **Train JIT-SPD Model**:
   ```python
   from jit_spd_model import JITSPDModel, TrainingConfig
   # ... setup model and training
   ```

4. **Run Experiments**:
   ```python
   from experiments import ExperimentRunner
   # ... configure and run experiments
   ```

## 🔬 Research Applications

This repository is designed for research in:

- **Software Defect Prediction**: Predicting software defects at commit time
- **Code Representation Learning**: Learning meaningful representations of source code
- **Multi-Modal Machine Learning**: Combining different types of data for prediction
- **Graph Neural Networks**: Applying GNNs to software engineering tasks
- **Interpretable AI**: Making ML models more interpretable for software engineers

## 📈 Performance and Scalability

### Computational Requirements

- **GPU Memory**: 2-4GB for batch size 64
- **CPU Memory**: 8-16GB for large datasets
- **Storage**: Variable based on dataset size and CPG complexity

### Optimization Features

- Parallel processing for large-scale projects
- Efficient lazy loading and caching
- Configurable batch sizes and worker counts
- Memory management for large graphs

## 🤝 Contributing

We welcome contributions! Please see the individual package READMEs for specific contribution guidelines.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd Interpretable-Ensemble-JIT-SDP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

## 📚 Documentation

Each package contains detailed documentation:

- **Code Property Graph**: [README](code/code_property_graph/README.md)
- **JIT-SPD Model**: [README](code/jit-spd-model/README.md)
- **Node Embeddings**: [README](code/node_embeddings/README.md)
- **Experiments**: [README](code/experiments/README.md)

## 📄 License

This project is part of academic research and follows appropriate licensing terms.

## 🆘 Support

For issues and questions:

1. Check the individual package documentation
2. Review error messages and validation results
3. Verify configuration and file paths
4. Check system requirements and dependencies

## 🔬 Citation

If you use this repository in your research, please cite the relevant papers and acknowledge the contribution of this work to the software engineering and machine learning communities.

---

**Note**: This repository represents ongoing research in interpretable software defect prediction. The implementation is designed to be both research-friendly and practically applicable for real-world software quality assessment.

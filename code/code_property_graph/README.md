# Code Property Graph (CPG) Package

A comprehensive Python package for generating, processing, and analyzing Code Property Graphs from Java source code using the Joern static analysis tool.

## Features

- **CPG Generation**: Generate Code Property Graphs from Java source code repositories
- **GraphML Export**: Convert CPG binary files to GraphML format for analysis
- **Cloud Storage Integration**: Upload processed files to Google Cloud Storage
- **Parallel Processing**: Multi-threaded processing for large-scale projects
- **Comprehensive Analysis**: Code smell detection, complexity analysis, and dependency tracking
- **Flexible Configuration**: Environment-based and file-based configuration options
- **Command-Line Interface**: Easy-to-use CLI for common operations

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Joern**: Static analysis tool for CPG generation
3. **Git**: For repository operations
4. **Google Cloud SDK** (optional): For cloud storage integration

### Install Joern

```bash
# Install Joern
curl -L https://github.com/joernio/joern/releases/latest/download/joern-install.sh | bash

# Add to PATH
export PATH="$HOME/.local/share/joern:$PATH"
```

### Install the Package

```bash
# Clone the repository
git clone <repository-url>
cd ThesisDataset

# Install dependencies
pip install -r requirements.txt

# The package is ready to use
```

## Quick Start

### Basic Usage

```python
from code_property_graph import CPGGenerator, CPGProcessor, CPGAnalyzer

# Generate CPGs for a project
generator = CPGGenerator()
cpg_files = generator.generate_cpgs_for_project("commons-beanutils")

# Process CPGs to GraphML
processor = CPGProcessor()
graphml_files, zip_files = processor.process_and_compress_project("commons-beanutils")

# Analyze a CPG
analyzer = CPGAnalyzer()
graph = analyzer.load_cpg_from_graphml("commit_hash.graphml")
report = analyzer.generate_cpg_report(graph, "commons-beanutils")
```

### Command Line Interface

```bash
# Generate CPGs for a project
python -m code_property_graph generate --project commons-beanutils

# Process existing CPGs to GraphML
python -m code_property_graph process --project commons-beanutils

# Analyze a CPG file
python -m code_property_graph analyze --file commit_hash.graphml --project commons-beanutils

# Check project status
python -m code_property_graph status --project commons-beanutils

# Show help
python -m code_property_graph --help
```

## Configuration

### Environment Variables

```bash
export CPG_BASE_PATH="/workspace/s2156631-thesis"
export CPG_REPOS_PATH="/workspace/repos"
export CPG_OUTPUT_PATH="/workspace/s2156631-thesis/cpgs"
export CPG_JOERN_MEMORY="-J-Xmx25G"
export CPG_MAX_WORKERS="21"
export CPG_BUCKET_NAME="s2156631-thesis"
```

### Configuration File

```json
{
  "base_project_path": "/workspace/s2156631-thesis",
  "repos_path": "/workspace/repos",
  "cpgs_output_path": "/workspace/s2156631-thesis/cpgs",
  "joern_memory": "-J-Xmx25G",
  "max_workers": 21,
  "bucket_name": "s2156631-thesis"
}
```

## Architecture

### Core Components

1. **CPGGenerator**: Generates CPGs from source code
2. **CPGProcessor**: Processes and exports CPGs to various formats
3. **CPGAnalyzer**: Analyzes CPG structure and detects patterns
4. **CPGConfig**: Manages configuration and settings

### Data Flow

```
Source Code → Git Checkout → Joern Parse → CPG Binary → GraphML → ZIP → Cloud Storage
```

## Usage Examples

### Single Project Processing

```python
from code_property_graph import CPGGenerator, CPGProcessor

# Initialize components
generator = CPGGenerator()
processor = CPGProcessor()

# Generate CPGs
project_name = "commons-beanutils"
cpg_files = generator.generate_cpgs_for_project(project_name)
print(f"Generated {len(cpg_files)} CPG files")

# Process to GraphML
graphml_files, zip_files = processor.process_and_compress_project(project_name)
print(f"Created {len(zip_files)} ZIP files")
```

### Batch Processing

```python
# Process multiple projects in parallel
projects = ["commons-beanutils", "commons-codec", "commons-collections"]

# Generate CPGs
generation_results = generator.generate_cpgs_for_projects(projects, parallel=True)

# Process all projects
processing_results = processor.batch_process_projects(projects, upload_to_cloud=True, parallel=True)

# Check results
for project, results in processing_results.items():
    if results['success']:
        print(f"{project}: {len(results['zip_files'])} files processed")
    else:
        print(f"{project}: FAILED - {results['errors']}")
```

### Advanced Analysis

```python
from code_property_graph import CPGAnalyzer

analyzer = CPGAnalyzer()

# Load CPG
graph = analyzer.load_cpg_from_graphml("commit_hash.graphml")

# Generate comprehensive report
report = analyzer.generate_cpg_report(graph, "project_name")

# Access specific analyses
print(f"Methods: {report['graph_statistics']['method_count']}")
print(f"Code smells: {report['summary']['total_smells']}")
print(f"High complexity methods: {report['summary']['high_complexity_methods']}")

# Extract call graph
call_graph = analyzer.extract_call_graph(graph)

# Analyze method complexity
complexity_metrics = analyzer.analyze_method_complexity(graph)
```

## Supported Projects

The package is configured to work with the following Apache Commons projects:

- ant-ivy
- commons-bcel
- commons-beanutils
- commons-codec
- commons-collections
- commons-compress
- commons-configuration
- commons-dbcp
- commons-digester
- commons-io
- commons-jcs
- commons-lang
- commons-math
- commons-net
- commons-scxml
- commons-validator
- commons-vfs
- giraph
- gora
- opennlp
- parquet-mr

## File Formats

### Input Formats
- **Java Source Code**: Standard Java source files
- **Git Repositories**: Local clones of Java projects

### Output Formats
- **CPG Binary**: Joern's native binary format (.bin)
- **GraphML**: XML-based graph format (.graphml)
- **ZIP Archives**: Compressed GraphML files (.graphml.zip)

## Performance Considerations

### Memory Requirements
- **Joern**: 25GB+ heap memory recommended for large projects
- **Python**: 8GB+ RAM for processing large CPGs

### Parallel Processing
- **Default**: 21 concurrent workers
- **Configurable**: Adjust based on system resources
- **Memory**: Monitor memory usage during parallel processing

### Optimization Tips
1. Use SSD storage for better I/O performance
2. Adjust worker count based on available CPU cores
3. Monitor memory usage and adjust Joern heap size
4. Use cloud storage for large-scale deployments

## Error Handling

### Common Issues

1. **Memory Insufficient**
   ```bash
   # Increase Joern heap size
   export CPG_JOERN_MEMORY="-J-Xmx50G"
   ```

2. **Git Checkout Failures**
   ```bash
   # Ensure repository is clean
   git reset --hard HEAD
   git clean -fd
   ```

3. **Joern Parse Errors**
   ```bash
   # Verify source code compilation
   mvn clean compile
   ```

### Debugging

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check project status
status = generator.get_project_status("project_name")
print(json.dumps(status, indent=2))

# Validate generated CPGs
validation = generator.validate_generated_cpgs("project_name")
print(json.dumps(validation, indent=2))
```

## Integration

### With Existing Workflows

```python
# Integrate with existing data processing pipelines
from code_property_graph import CPGGenerator

def process_commit(project_name, commit_hash):
    generator = CPGGenerator()
    cpg_file = generator.generate_cpg_for_commit(project_name, commit_hash)
    return cpg_file

# Use in batch processing
commits = load_commit_list()
for commit in commits:
    cpg_file = process_commit(commit.project, commit.hash)
    # Continue with existing processing...
```

### With Machine Learning Pipelines

```python
# Extract features for ML models
from code_property_graph import CPGAnalyzer
import networkx as nx

def extract_cpg_features(graphml_file):
    analyzer = CPGAnalyzer()
    graph = analyzer.load_cpg_from_graphml(graphml_file)
    
    # Extract structural features
    stats = analyzer.analyze_cpg_structure(graph)
    
    # Extract complexity features
    complexity = analyzer.analyze_method_complexity(graph)
    
    # Extract code smell features
    smells = analyzer.find_code_smells(graph)
    
    return {
        'structural': stats,
        'complexity': complexity,
        'smells': smells
    }
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd ThesisDataset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
python -m flake8 code_property_graph/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for all function parameters and return values
- Include comprehensive docstrings
- Add unit tests for new functionality

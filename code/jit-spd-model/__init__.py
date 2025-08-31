"""
JIT-SPD Model Package: Just-In-Time Software Defect Prediction

This package provides a comprehensive deep learning system for predicting software defects
at the commit level using multiple data modalities including graph neural networks,
natural language processing, and traditional software metrics.

Main components:
- JITSPDModel: Core model architecture
- GraphConvolution: Graph neural network layers
- FeatureProcessor: Multi-modal feature processing
- TrainingManager: Training pipeline management
- DataLoader: Efficient data loading and preprocessing
"""

from .model import JITSPDModel, HomogeneousGraphConvolution
from .features import FeatureProcessor, TextEmbeddingGenerator
from .training import TrainingManager, EarlyStopping
from .data import JITSPDDataset, DataProcessor
from .config import ModelConfig, TrainingConfig

__version__ = "1.0.0"
__author__ = "Thesis Dataset Team"

__all__ = [
    'JITSPDModel',
    'HomogeneousGraphConvolution', 
    'FeatureProcessor',
    'TextEmbeddingGenerator',
    'TrainingManager',
    'EarlyStopping',
    'JITSPDDataset',
    'DataProcessor',
    'ModelConfig',
    'TrainingConfig'
]

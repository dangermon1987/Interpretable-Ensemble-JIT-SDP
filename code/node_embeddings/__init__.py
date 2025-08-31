"""
Node Embeddings Package for Code Property Graphs

This package provides utilities for generating, processing, and managing node embeddings
from Code Property Graphs using multiple embedding techniques including Word2Vec, 
Node2Vec, and CodeBERT.

Main components:
- EmbeddingGenerator: Core embedding generation pipeline
- Word2VecEmbedder: Word2Vec-based semantic embeddings
- Node2VecEmbedder: Node2Vec-based structural embeddings
- CodeBERTEmbedder: CodeBERT-based semantic embeddings
- EmbeddingProcessor: Embedding combination and processing
- EmbeddingConfig: Configuration management
"""

from .generator import EmbeddingGenerator
from .word2vec_embedder import Word2VecEmbedder
from .node2vec_embedder import Node2VecEmbedder
from .codebert_embedder import CodeBERTEmbedder
from .processor import EmbeddingProcessor
from .config import EmbeddingConfig
from .utils import (
    load_graph_from_graphml, 
    update_graph_to_k_distance,
    extract_node_features,
    combine_embeddings
)

__version__ = "1.0.0"
__author__ = "Thesis Dataset Team"

__all__ = [
    'EmbeddingGenerator',
    'Word2VecEmbedder',
    'Node2VecEmbedder', 
    'CodeBERTEmbedder',
    'EmbeddingProcessor',
    'EmbeddingConfig',
    'load_graph_from_graphml',
    'update_graph_to_k_distance',
    'extract_node_features',
    'combine_embeddings'
]

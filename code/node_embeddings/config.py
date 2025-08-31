"""
Configuration management for node embedding generation and processing.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Configuration class for node embedding generation and processing."""
    
    # Default project paths
    base_project_path: str = "/workspace/s2156631-thesis"
    data_dir: str = "/workspace/s2156631-thesis/data"
    output_dir: str = "/workspace/s2156631-thesis/embeddings"
    
    # Word2Vec settings
    word2vec_vector_size: int = 128
    word2vec_window: int = 10
    word2vec_min_count: int = 1
    word2vec_workers: int = 4
    word2vec_epochs: int = 5
    
    # Node2Vec settings
    node2vec_dimensions: int = 128
    node2vec_walk_length: int = 30
    node2vec_num_walks: int = 200
    node2vec_window: int = 10
    node2vec_min_count: int = 1
    node2vec_batch_words: int = 32
    node2vec_workers: int = 8
    node2vec_seed: int = 90
    
    # CodeBERT settings
    codebert_model_path: str = "/workspace/s2156631-thesis/data/codebert-base"
    codebert_max_tokens: int = 512
    codebert_batch_size: int = 32
    
    # Graph processing settings
    k_distance_levels: List[int] = field(default_factory=lambda: [0, 1])
    state_encoding: Dict[str, int] = field(default_factory=lambda: {
        'before': -1,
        'unchanged': 0,
        'after': 1
    })
    
    # Processing settings
    max_workers: int = 8
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Default projects list
    default_projects: List[str] = field(default_factory=lambda: [
        'ant-ivy', 'commons-bcel', 'commons-beanutils', 'commons-codec',
        'commons-collections', 'commons-compress', 'commons-configuration',
        'commons-dbcp', 'commons-digester', 'commons-io', 'commons-jcs',
        'commons-lang', 'commons-math', 'commons-net', 'commons-scxml',
        'commons-validator', 'commons-vfs', 'giraph', 'gora', 'opennlp', 'parquet-mr'
    ])
    
    # CPG properties for feature extraction
    cpg_properties: List[str] = field(default_factory=lambda: [
        'labelV', 'AST_PARENT_FULL_NAME', 'AST_PARENT_TYPE', 'CODE', 'CONTENT',
        'FULL_NAME', 'IS_EXTERNAL', 'NAME', 'PARSER_TYPE_NAME', 'CLASS_NAME',
        'CLASS_SHORT_NAME', 'METHOD_SHORT_NAME', 'NODE_LABEL', 'PACKAGE_NAME',
        'SYMBOL', 'VARIABLE', 'ARGUMENT_INDEX', 'ARGUMENT_NAME', 'DISPATCH_TYPE',
        'EVALUATION_STRATEGY', 'METHOD_FULL_NAME', 'CANONICAL_NAME',
        'CONTROL_STRUCTURE_TYPE', 'MODIFIER_TYPE', 'ALIAS_TYPE_FULL_NAME',
        'INHERITS_FROM_TYPE_FULL_NAME', 'TYPE_DECL_FULL_NAME', 'TYPE_FULL_NAME',
        'IS_VARIADIC', 'SIGNATURE', 'FILENAME'
    ])
    
    # Edge types for graph analysis
    edge_types: List[str] = field(default_factory=lambda: [
        "ALIAS_OF", "ARGUMENT", "AST", "BINDS", "BINDS_TO", "CALL", "CAPTURE",
        "CAPTURED_BY", "CDG", "CFG", "CONDITION", "CONTAINS", "DOMINATE",
        "EVAL_TYPE", "IMPORTS", "INHERITS_FROM", "IS_CALL_FOR_IMPORT",
        "PARAMETER_LINK", "POST_DOMINATE", "REACHING_DEF", "RECEIVER", "REF",
        "SOURCE_FILE", "TAGGED_BY"
    ])
    
    def __post_init__(self):
        """Validate and set default paths."""
        # Set device automatically if not specified
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_project_data_path(self, project_name: str) -> str:
        """Get the data path for a specific project."""
        return os.path.join(self.data_dir, project_name)
    
    def get_project_output_path(self, project_name: str) -> str:
        """Get the output path for a specific project."""
        return os.path.join(self.output_dir, project_name)
    
    def get_codebert_model_path(self) -> str:
        """Get the CodeBERT model path."""
        return self.codebert_model_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'base_project_path': self.base_project_path,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'word2vec_vector_size': self.word2vec_vector_size,
            'word2vec_window': self.word2vec_window,
            'word2vec_workers': self.word2vec_workers,
            'node2vec_dimensions': self.node2vec_dimensions,
            'node2vec_walk_length': self.node2vec_walk_length,
            'node2vec_num_walks': self.node2vec_num_walks,
            'node2vec_workers': self.node2vec_workers,
            'codebert_batch_size': self.codebert_batch_size,
            'k_distance_levels': self.k_distance_levels.copy(),
            'max_workers': self.max_workers,
            'device': self.device,
            'default_projects': self.default_projects.copy()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EmbeddingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'EmbeddingConfig':
        """Create configuration from environment variables."""
        return cls(
            base_project_path=os.getenv('EMBEDDING_BASE_PATH', "/workspace/s2156631-thesis"),
            data_dir=os.getenv('EMBEDDING_DATA_DIR', "/workspace/s2156631-thesis/data"),
            output_dir=os.getenv('EMBEDDING_OUTPUT_DIR', "/workspace/s2156631-thesis/embeddings"),
            word2vec_vector_size=int(os.getenv('EMBEDDING_WORD2VEC_SIZE', "128")),
            word2vec_workers=int(os.getenv('EMBEDDING_WORD2VEC_WORKERS', "4")),
            node2vec_dimensions=int(os.getenv('EMBEDDING_NODE2VEC_DIM', "128")),
            node2vec_workers=int(os.getenv('EMBEDDING_NODE2VEC_WORKERS', "8")),
            codebert_batch_size=int(os.getenv('EMBEDDING_CODEBERT_BATCH', "32")),
            max_workers=int(os.getenv('EMBEDDING_MAX_WORKERS', "8")),
            device=os.getenv('EMBEDDING_DEVICE', "auto")
        )


# Default configuration instance
default_config = EmbeddingConfig()

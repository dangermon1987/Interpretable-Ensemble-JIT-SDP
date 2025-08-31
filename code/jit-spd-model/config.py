"""
Configuration management for JIT-SPD model and training.
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for the JIT-SPD model architecture."""
    
    # Model dimensions
    node_feature_dimension: int = 896  # 768 (CodeBERT) + 128 (Node2Vec)
    out_channels: int = 128
    in_channels: int = 897  # node_feature_dimension + 1 (state vector)
    
    # Feature configuration
    with_manual_features: bool = True
    with_text_features: bool = True
    with_graph_features: bool = True
    manual_size: int = 14
    
    # Graph processing
    graph_edges: List[str] = field(default_factory=lambda: ["CALL", "CFG", "PDG"])
    
    # Architecture settings
    dropout_rate: float = 0.1
    activation: str = "gelu"  # "gelu", "relu", "leaky_relu"
    
    # Edge type mapping
    edge_type_mapping: Dict[str, str] = field(default_factory=lambda: {
        "AST": "AST",
        "CONDITION": "AST", 
        "REF": "AST",
        "CFG": "CFG",
        "CDG": "PDG",
        "REACHING_DEF": "PDG",
        "CALL": "CALL",
        "ARGUMENT": "CALL",
        "RECEIVER": "CALL"
    })
    
    # Feature categories
    kamei_features: List[str] = field(default_factory=lambda: [
        'la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
        'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix'
    ])
    
    ast_features: List[str] = field(default_factory=lambda: [
        'current_AST_compilationunit', 'current_AST_switchstatementcase',
        'delta_AST_localvariabledeclaration', 'current_AST_methodinvocation',
        'parent_AST_memberreference', 'parent_AST_variabledeclarator',
        'parent_AST_formalparameter', 'current_AST_blockstatement',
        'current_AST_statementexpression', 'delta_AST_memberreference'
    ])
    
    sm_features: List[str] = field(default_factory=lambda: [
        'current_SM_method_mism_median', 'current_SM_class_loc_stdev',
        'current_SM_method_mi_avg', 'current_SM_method_mi_max',
        'current_SM_class_nos_median', 'current_SM_method_nii_sum',
        'current_SM_method_nos_max', 'current_SM_method_mims_median',
        'current_SM_class_tlloc_min', 'delta_SM_method_tloc_sum'
    ])
    
    pmd_features: List[str] = field(default_factory=lambda: [
        'current_PMD_severity_minor', 'current_PMD_arp', 'current_PMD_dp',
        'current_PMD_rule_type_string and stringbuffer rules', 'current_PMD_gdl',
        'current_PMD_adl', 'current_PMD_atret', 'current_PMD_if',
        'current_PMD_gls', 'current_PMD_vnc', 'current_PMD_fdsbasoc'
    ])
    
    def get_all_manual_features(self) -> List[str]:
        """Get all manual features combined."""
        return (self.kamei_features + self.ast_features + 
                self.sm_features + self.pmd_features)
    
    def get_feature_size(self) -> int:
        """Get the total size of manual features."""
        return len(self.get_all_manual_features())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'node_feature_dimension': self.node_feature_dimension,
            'out_channels': self.out_channels,
            'in_channels': self.in_channels,
            'with_manual_features': self.with_manual_features,
            'with_text_features': self.with_text_features,
            'with_graph_features': self.with_graph_features,
            'manual_size': self.manual_size,
            'graph_edges': self.graph_edges.copy(),
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'edge_type_mapping': self.edge_type_mapping.copy(),
            'kamei_features': self.kamei_features.copy(),
            'ast_features': self.ast_features.copy(),
            'sm_features': self.sm_features.copy(),
            'pmd_features': self.pmd_features.copy()
        }


@dataclass
class TrainingConfig:
    """Configuration for training the JIT-SPD model."""
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 8
    min_lr: float = 1e-5
    lr_factor: float = 0.7
    
    # Data configuration
    data_dir: str = "/workspace/s2156631-thesis/new_before_after_data/output/"
    result_path: str = "/workspace/s2156631-thesis/results/"
    
    # Model paths
    codebert_model_path: str = "/workspace/s2156631-thesis/data/codebert-base"
    
    # Project list
    project_list: List[str] = field(default_factory=lambda: [
        'ant-ivy', 'commons-bcel', 'commons-beanutils', 'commons-codec',
        'commons-collections', 'commons-compress', 'commons-configuration',
        'commons-dbcp', 'commons-digester', 'commons-io', 'commons-jcs',
        'commons-lang', 'commons-math', 'commons-net', 'commons-scxml',
        'commons-validator', 'commons-vfs', 'giraph', 'gora', 'opennlp', 'parquet-mr'
    ])
    
    # Training features
    use_weighted_sampling: bool = True
    use_early_stopping: bool = True
    use_lr_scheduling: bool = True
    use_tensorboard: bool = True
    
    # Device configuration
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Text processing
    max_commit_message_tokens: int = 64
    max_total_tokens: int = 512
    
    # Special tokens
    add_token: str = "[ADD]"
    del_token: str = "[DEL]"
    
    def __post_init__(self):
        """Validate and set default paths."""
        # Set device automatically if not specified
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create result directory if it doesn't exist
        os.makedirs(self.result_path, exist_ok=True)
    
    def get_experiment_path(self, exp_str: str, exp_idx: int) -> str:
        """Get the path for a specific experiment."""
        return os.path.join(self.result_path, exp_str, str(exp_idx))
    
    def get_tensorboard_path(self, exp_str: str, exp_idx: int) -> str:
        """Get the TensorBoard log path for a specific experiment."""
        return os.path.join(self.result_path, exp_str, 'tensorboard', str(exp_idx))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'min_lr': self.min_lr,
            'lr_factor': self.lr_factor,
            'data_dir': self.data_dir,
            'result_path': self.result_path,
            'codebert_model_path': self.codebert_model_path,
            'project_list': self.project_list.copy(),
            'use_weighted_sampling': self.use_weighted_sampling,
            'use_early_stopping': self.use_early_stopping,
            'use_lr_scheduling': self.use_lr_scheduling,
            'use_tensorboard': self.use_tensorboard,
            'device': self.device,
            'max_commit_message_tokens': self.max_commit_message_tokens,
            'max_total_tokens': self.max_total_tokens
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'TrainingConfig':
        """Create configuration from environment variables."""
        return cls(
            batch_size=int(os.getenv('JITSPD_BATCH_SIZE', "64")),
            learning_rate=float(os.getenv('JITSPD_LEARNING_RATE', "1e-2")),
            weight_decay=float(os.getenv('JITSPD_WEIGHT_DECAY', "1e-4")),
            max_epochs=int(os.getenv('JITSPD_MAX_EPOCHS', "200")),
            patience=int(os.getenv('JITSPD_PATIENCE', "8")),
            data_dir=os.getenv('JITSPD_DATA_DIR', "/workspace/s2156631-thesis/new_before_after_data/output/"),
            result_path=os.getenv('JITSPD_RESULT_PATH', "/workspace/s2156631-thesis/results/"),
            codebert_model_path=os.getenv('JITSPD_CODEBERT_PATH', "/workspace/s2156631-thesis/data/codebert-base"),
            device=os.getenv('JITSPD_DEVICE', "auto")
        )


# Default configurations
default_model_config = ModelConfig()
default_training_config = TrainingConfig()

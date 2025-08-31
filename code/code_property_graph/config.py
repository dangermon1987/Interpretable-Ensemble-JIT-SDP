"""
Configuration management for Code Property Graph generation and processing.
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class CPGConfig:
    """Configuration class for CPG generation and processing."""
    
    # Default project paths
    base_project_path: str = "/Users/dangtran/workspace/study/workspace/"
    repos_path: str = "/Users/dangtran/workspace/study/data/repos/"
    cpgs_output_path: str = "/Users/dangtran/workspace/study/workspace/cpgs/"
    
    # Joern settings
    joern_memory: str = "-J-Xmx25G"
    joern_parse_cmd: str = "joern-parse"
    joern_export_cmd: str = "joern-export"
    
    # Processing settings
    max_workers: int = 21
    batch_size: int = 20
    
    # Cloud storage settings (optional)
    bucket_name: str = "s2156631-thesis"
    google_credentials_path: str = "keys/ml-80805-4893ba01f974.json"
    
    # Default projects list
    default_projects: List[str] = field(default_factory=lambda: [
        'ant-ivy', 'commons-bcel', 'commons-beanutils', 'commons-codec',
        'commons-collections', 'commons-compress', 'commons-configuration',
        'commons-dbcp', 'commons-digester', 'commons-io', 'commons-jcs',
        'commons-lang', 'commons-math', 'commons-net', 'commons-scxml',
        'commons-validator', 'commons-vfs', 'giraph', 'gora', 'opennlp', 'parquet-mr'
    ])
    
    # CPG properties for analysis
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
        # Set environment variable for Google credentials if provided
        if os.path.exists(self.google_credentials_path):
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.google_credentials_path
    
    def get_project_path(self, project_name: str) -> str:
        """Get the repository path for a specific project."""
        return os.path.join(self.repos_path, project_name)
    
    def get_cpg_output_path(self, project_name: str) -> str:
        """Get the CPG output path for a specific project."""
        return os.path.join(self.cpgs_output_path, project_name)
    
    def get_dataset_path(self, project_name: str) -> str:
        """Get the dataset path for a specific project."""
        return os.path.join(self.base_project_path, f"{project_name}_ces.csv")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'base_project_path': self.base_project_path,
            'repos_path': self.repos_path,
            'cpgs_output_path': self.cpgs_output_path,
            'joern_memory': self.joern_memory,
            'max_workers': self.max_workers,
            'batch_size': self.batch_size,
            'bucket_name': self.bucket_name,
            'default_projects': self.default_projects.copy()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CPGConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_env(cls) -> 'CPGConfig':
        """Create configuration from environment variables."""
        return cls(
            base_project_path=os.getenv('CPG_BASE_PATH', "/workspace/s2156631-thesis"),
            repos_path=os.getenv('CPG_REPOS_PATH', "/workspace/repos"),
            cpgs_output_path=os.getenv('CPG_OUTPUT_PATH', "/workspace/s2156631-thesis/cpgs"),
            joern_memory=os.getenv('CPG_JOERN_MEMORY', "-J-Xmx25G"),
            max_workers=int(os.getenv('CPG_MAX_WORKERS', "21")),
            bucket_name=os.getenv('CPG_BUCKET_NAME', "s2156631-thesis")
        )


# Default configuration instance
default_config = CPGConfig()

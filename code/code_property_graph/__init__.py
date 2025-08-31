"""
Code Property Graph (CPG) Generation and Processing Package

This package provides utilities for generating, processing, and analyzing Code Property Graphs
from Java source code repositories using Joern static analysis tool.

Main components:
- CPGGenerator: Core CPG generation from source code
- CPGProcessor: GraphML export and processing
- CPGAnalyzer: Analysis and filtering utilities
- CPGConfig: Configuration management
"""

from .generator import CPGGenerator
from .processor import CPGProcessor
from .analyzer import CPGAnalyzer
from .config import CPGConfig
from .utils import clean_java_qualified_name, load_cpg_from_graphml

__version__ = "1.0.0"
__author__ = "Thesis Dataset Team"

__all__ = [
    'CPGGenerator',
    'CPGProcessor', 
    'CPGAnalyzer',
    'CPGConfig',
    'clean_java_qualified_name',
    'load_cpg_from_graphml'
]

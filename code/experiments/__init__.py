"""
Experiments Package for JIT-SPD Model

This package contains experiment scripts and evaluation utilities for the JIT-SPD model.
It provides tools for running experiments, analyzing results, and generating reports.
"""

from .experiment_runner import ExperimentRunner
from .evaluation import ModelEvaluator
from .metrics import MetricsCalculator
from .reporting import ExperimentReporter

__version__ = "1.0.0"
__author__ = "Thesis Dataset Team"

__all__ = [
    'ExperimentRunner',
    'ModelEvaluator', 
    'MetricsCalculator',
    'ExperimentReporter'
]

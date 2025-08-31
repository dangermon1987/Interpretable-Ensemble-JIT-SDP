"""
Core Code Property Graph generation functionality.
"""

import os
import subprocess
import pandas as pd
import concurrent.futures
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from .config import CPGConfig
from .utils import clean_java_qualified_name


class CPGGenerator:
    """
    Core class for generating Code Property Graphs from Java source code.
    
    Code Reference: genCpgs.py (lines 1-50)
    """
    
    def __init__(self, config: Optional[CPGConfig] = None):
        """
        Initialize the CPG generator.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or CPGConfig()
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not os.path.exists(self.config.repos_path):
            raise ValueError(f"Repository path does not exist: {self.config.repos_path}")
        
        if not os.path.exists(self.config.base_project_path):
            raise ValueError(f"Base project path does not exist: {self.config.base_project_path}")
        
        # Create CPG output directory if it doesn't exist
        os.makedirs(self.config.cpgs_output_path, exist_ok=True)
    
    def generate_cpg_for_commit(self, project_name: str, commit_hash: str, 
                               project_path: Optional[str] = None) -> str:
        """
        Generate CPG for a specific commit.
        
        Args:
            project_name: Name of the project
            commit_hash: Git commit hash
            project_path: Optional custom project path
            
        Returns:
            Path to the generated CPG file
            
        Raises:
            subprocess.CalledProcessError: If git checkout or joern-parse fails
            FileNotFoundError: If project path doesn't exist
        """
        if project_path is None:
            project_path = self.config.get_project_path(project_name)
        
        if not os.path.exists(project_path):
            raise FileNotFoundError(f"Project path does not exist: {project_path}")
        
        # Create output directory
        cpg_output_path = self.config.get_cpg_output_path(project_name)
        os.makedirs(cpg_output_path, exist_ok=True)
        
        # Checkout specific commit
        print(f"Checking out commit {commit_hash} for project {project_name}")
        subprocess.run(['git', 'checkout', '-f', commit_hash], 
                      cwd=project_path, check=True, capture_output=True)
        
        # Generate CPG using Joern
        cpg_file = os.path.join(cpg_output_path, f'{commit_hash}.bin')
        print(f"Generating CPG for commit {commit_hash}")
        
        subprocess.run([
            self.config.joern_parse_cmd, 
            project_path, 
            self.config.joern_memory, 
            '-o', cpg_file
        ], check=True, capture_output=True)
        
        print(f"Generated CPG for commit {commit_hash}: {cpg_file}")
        return cpg_file
    
    def generate_cpgs_for_project(self, project_name: str, 
                                 dataset_path: Optional[str] = None,
                                 commit_column: str = 'commit_id') -> List[str]:
        """
        Generate CPGs for all commits in a project dataset.
        
        Args:
            project_name: Name of the project
            dataset_path: Optional custom dataset path
            commit_column: Name of the column containing commit hashes
            
        Returns:
            List of paths to generated CPG files
        """
        if dataset_path is None:
            dataset_path = self.config.get_dataset_path(project_name)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        if commit_column not in df.columns:
            raise ValueError(f"Column '{commit_column}' not found in dataset")
        
        print(f"Processing {len(df)} commits for project {project_name}")
        
        generated_files = []
        project_path = self.config.get_project_path(project_name)
        
        for index, row in df.iterrows():
            commit_hash = row[commit_column]
            try:
                cpg_file = self.generate_cpg_for_commit(project_name, commit_hash, project_path)
                generated_files.append(cpg_file)
            except Exception as e:
                print(f"Error processing commit {commit_hash} for project {project_name}: {e}")
                continue
        
        print(f"Successfully generated {len(generated_files)} CPGs for project {project_name}")
        return generated_files
    
    def generate_cpgs_for_projects(self, project_names: Optional[List[str]] = None,
                                  parallel: bool = True) -> Dict[str, List[str]]:
        """
        Generate CPGs for multiple projects.
        
        Args:
            project_names: List of project names (uses default if None)
            parallel: Whether to process projects in parallel
            
        Returns:
            Dictionary mapping project names to lists of generated CPG files
        """
        if project_names is None:
            project_names = self.config.default_projects
        
        results = {}
        
        if parallel:
            results = self._generate_cpgs_parallel(project_names)
        else:
            results = self._generate_cpgs_sequential(project_names)
        
        return results
    
    def _generate_cpgs_parallel(self, project_names: List[str]) -> Dict[str, List[str]]:
        """Generate CPGs for projects in parallel."""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all projects
            future_to_project = {
                executor.submit(self.generate_cpgs_for_project, project_name): project_name
                for project_name in project_names
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_project):
                project_name = future_to_project[future]
                try:
                    cpg_files = future.result()
                    results[project_name] = cpg_files
                except Exception as e:
                    print(f"Error processing project {project_name}: {e}")
                    results[project_name] = []
        
        return results
    
    def _generate_cpgs_sequential(self, project_names: List[str]) -> Dict[str, List[str]]:
        """Generate CPGs for projects sequentially."""
        results = {}
        
        for project_name in project_names:
            try:
                cpg_files = self.generate_cpgs_for_project(project_name)
                results[project_name] = cpg_files
            except Exception as e:
                print(f"Error processing project {project_name}: {e}")
                results[project_name] = []
        
        return results
    
    def get_project_status(self, project_name: str) -> Dict[str, any]:
        """
        Get the status of CPG generation for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary containing project status information
        """
        dataset_path = self.config.get_dataset_path(project_name)
        cpg_output_path = self.config.get_cpg_output_path(project_name)
        
        status = {
            'project_name': project_name,
            'dataset_exists': os.path.exists(dataset_path),
            'output_path_exists': os.path.exists(cpg_output_path),
            'total_commits': 0,
            'generated_cpgs': 0,
            'missing_cpgs': 0
        }
        
        if status['dataset_exists']:
            try:
                df = pd.read_csv(dataset_path)
                status['total_commits'] = len(df)
                
                if 'commit_id' in df.columns:
                    expected_commits = set(df['commit_id'].unique())
                else:
                    expected_commits = set()
            except Exception:
                expected_commits = set()
        else:
            expected_commits = set()
        
        if status['output_path_exists']:
            try:
                existing_cpgs = set()
                for file in os.listdir(cpg_output_path):
                    if file.endswith('.bin'):
                        commit_hash = file.replace('.bin', '')
                        existing_cpgs.add(commit_hash)
                
                status['generated_cpgs'] = len(existing_cpgs)
                status['missing_cpgs'] = len(expected_commits - existing_cpgs)
            except Exception:
                pass
        
        return status
    
    def cleanup_failed_generations(self, project_name: str, 
                                  min_file_size_mb: float = 0.1) -> List[str]:
        """
        Clean up failed CPG generation files (very small or corrupted files).
        
        Args:
            project_name: Name of the project
            min_file_size_mb: Minimum file size in MB to consider valid
            
        Returns:
            List of cleaned up file paths
        """
        cpg_output_path = self.config.get_cpg_output_path(project_name)
        if not os.path.exists(cpg_output_path):
            return []
        
        cleaned_files = []
        min_size_bytes = min_file_size_mb * 1024 * 1024
        
        for file in os.listdir(cpg_output_path):
            if file.endswith('.bin'):
                file_path = os.path.join(cpg_output_path, file)
                try:
                    if os.path.getsize(file_path) < min_size_bytes:
                        os.remove(file_path)
                        cleaned_files.append(file_path)
                        print(f"Cleaned up small/corrupted file: {file}")
                except Exception as e:
                    print(f"Error checking file {file}: {e}")
        
        return cleaned_files
    
    def validate_generated_cpgs(self, project_name: str) -> Dict[str, any]:
        """
        Validate generated CPG files for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary containing validation results
        """
        cpg_output_path = self.config.get_cpg_output_path(project_name)
        if not os.path.exists(cpg_output_path):
            return {'valid': False, 'error': 'Output path does not exist'}
        
        validation_results = {
            'project_name': project_name,
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'errors': []
        }
        
        for file in os.listdir(cpg_output_path):
            if file.endswith('.bin'):
                validation_results['total_files'] += 1
                file_path = os.path.join(cpg_output_path, file)
                
                try:
                    # Check file size
                    file_size = os.path.getsize(file_path)
                    if file_size < 1024:  # Less than 1KB
                        validation_results['invalid_files'] += 1
                        validation_results['errors'].append(f"File too small: {file} ({file_size} bytes)")
                    else:
                        validation_results['valid_files'] += 1
                except Exception as e:
                    validation_results['invalid_files'] += 1
                    validation_results['errors'].append(f"Error validating {file}: {e}")
        
        return validation_results

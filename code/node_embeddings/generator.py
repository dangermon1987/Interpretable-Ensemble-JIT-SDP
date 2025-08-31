"""
Main embedding generator class that orchestrates the entire CPG node embedding pipeline.
"""

import os
import networkx as nx
import torch
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .config import EmbeddingConfig
from .word2vec_embedder import Word2VecEmbedder
from .node2vec_embedder import Node2VecEmbedder
from .codebert_embedder import CodeBERTEmbedder
from .processor import EmbeddingProcessor
from .utils import (
    load_graph_from_graphml, 
    update_graph_to_k_distance,
    save_embeddings,
    get_graph_statistics
)


class EmbeddingGenerator:
    """
    Main class for generating CPG node embeddings using multiple embedding techniques.
    
    Code Reference: new_embedding/GenGraphData.py (lines 317-339)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding generator.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or EmbeddingConfig()
        
        # Initialize embedders
        self.word2vec_embedder = Word2VecEmbedder(self.config)
        self.node2vec_embedder = Node2VecEmbedder(self.config)
        self.codebert_embedder = CodeBERTEmbedder(self.config)
        self.processor = EmbeddingProcessor(self.config)
        
        print(f"EmbeddingGenerator initialized with device: {self.config.device}")
    
    def generate_embeddings_for_commit(self, before_graphml_file: str, after_graphml_file: str,
                                     output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate embeddings for a single commit (before/after comparison).
        
        Args:
            before_graphml_file: Path to before GraphML file
            after_graphml_file: Path to after GraphML file
            output_dir: Output directory (uses config default if None)
            
        Returns:
            Dictionary containing results and file paths
        """
        print(f"Generating embeddings for commit: {os.path.basename(before_graphml_file)}")
        
        if output_dir is None:
            output_dir = self.config.output_dir
        
        # Load and combine graphs
        before_graph = load_graph_from_graphml(before_graphml_file)
        after_graph = load_graph_from_graphml(after_graphml_file)
        commit_graph = nx.compose(before_graph, after_graph)
        
        # Apply k-distance filtering
        temp = commit_graph
        commit_graph = update_graph_to_k_distance(commit_graph, k=1)
        if len(commit_graph) == 0:
            commit_graph = temp
            print("Warning: K-distance filtering resulted in empty graph, using original graph")
        
        print(f"Processing graph with {commit_graph.number_of_nodes()} nodes and {commit_graph.number_of_edges()} edges")
        
        # Generate embeddings
        try:
            # Node2Vec structural embeddings
            print("Generating Node2Vec embeddings...")
            node2vec_embeddings = self.node2vec_embedder.generate_embeddings(commit_graph)
            
            # CodeBERT semantic embeddings
            print("Generating CodeBERT embeddings...")
            node_text_embeddings, node_indices = self.codebert_embedder.process_graph_nodes(commit_graph)
            
            # Combine embeddings
            print("Combining embeddings...")
            combined_embeddings = self.processor.combine_embeddings(node2vec_embeddings, node_text_embeddings)
            
            # Generate multi-level heterogeneous graphs
            hetero_data_files = []
            for k in self.config.k_distance_levels:
                print(f"Creating heterogeneous data for k={k}")
                hetero_data = self.processor.create_heterogeneous_data(commit_graph, combined_embeddings, k=k)
                
                # Save heterogeneous data
                base_name = os.path.splitext(os.path.basename(before_graphml_file))[0]
                hetero_data_file = os.path.join(output_dir, f"{base_name}_hetero_data_{k}.pt")
                self.processor.save_heterogeneous_data(hetero_data, hetero_data_file)
                hetero_data_files.append(hetero_data_file)
            
            # Save combined embeddings
            embeddings_file = os.path.join(output_dir, f"{base_name}_combined_embeddings.txt")
            save_embeddings(combined_embeddings, embeddings_file)
            
            # Generate statistics
            stats = self.processor.get_embedding_statistics(combined_embeddings)
            graph_stats = get_graph_statistics(commit_graph)
            
            results = {
                'status': 'success',
                'before_graphml': before_graphml_file,
                'after_graphml': after_graphml_file,
                'combined_embeddings_file': embeddings_file,
                'hetero_data_files': hetero_data_files,
                'embedding_stats': stats,
                'graph_stats': graph_stats,
                'total_nodes_processed': len(combined_embeddings)
            }
            
            print(f"Successfully generated embeddings for {len(combined_embeddings)} nodes")
            return results
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'before_graphml': before_graphml_file,
                'after_graphml': after_graphml_file
            }
    
    def generate_embeddings_for_project(self, project_name: str, 
                                      data_dir: Optional[str] = None,
                                      output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate embeddings for all commits in a project.
        
        Args:
            project_name: Name of the project
            data_dir: Data directory (uses config default if None)
            output_dir: Output directory (uses config default if None)
            
        Returns:
            Dictionary containing results summary
        """
        if data_dir is None:
            data_dir = self.config.get_project_data_path(project_name)
        
        if output_dir is None:
            output_dir = self.config.get_project_output_path(project_name)
        
        print(f"Generating embeddings for project: {project_name}")
        print(f"Data directory: {data_dir}")
        print(f"Output directory: {output_dir}")
        
        # Find GraphML files
        graphml_files = self._find_graphml_files(data_dir, project_name)
        
        if not graphml_files:
            print(f"No GraphML files found for project {project_name}")
            return {'status': 'no_files', 'project': project_name}
        
        print(f"Found {len(graphml_files)} GraphML files")
        
        # Process each file
        results = []
        successful = 0
        failed = 0
        
        for i, (before_file, after_file) in enumerate(graphml_files):
            print(f"\nProcessing file {i+1}/{len(graphml_files)}: {os.path.basename(before_file)}")
            
            try:
                result = self.generate_embeddings_for_commit(before_file, after_file, output_dir)
                results.append(result)
                
                if result['status'] == 'success':
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"Error processing file: {e}")
                failed += 1
                results.append({
                    'status': 'error',
                    'error': str(e),
                    'before_graphml': before_file,
                    'after_graphml': after_file
                })
        
        # Summary
        summary = {
            'status': 'completed',
            'project': project_name,
            'total_files': len(graphml_files),
            'successful': successful,
            'failed': failed,
            'results': results,
            'data_dir': data_dir,
            'output_dir': output_dir
        }
        
        print(f"\nProject {project_name} completed:")
        print(f"  Total files: {len(graphml_files)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        
        return summary
    
    def generate_embeddings_for_projects(self, project_names: Optional[List[str]] = None,
                                       parallel: bool = True) -> Dict[str, Any]:
        """
        Generate embeddings for multiple projects.
        
        Args:
            project_names: List of project names (uses config default if None)
            parallel: Whether to process projects in parallel
            
        Returns:
            Dictionary containing results for all projects
        """
        if project_names is None:
            project_names = self.config.default_projects
        
        print(f"Generating embeddings for {len(project_names)} projects: {project_names}")
        
        if parallel:
            return self._generate_embeddings_parallel(project_names)
        else:
            return self._generate_embeddings_sequential(project_names)
    
    def _generate_embeddings_sequential(self, project_names: List[str]) -> Dict[str, Any]:
        """Generate embeddings sequentially."""
        all_results = {}
        
        for project_name in project_names:
            print(f"\n{'='*50}")
            print(f"Processing project: {project_name}")
            print(f"{'='*50}")
            
            try:
                result = self.generate_embeddings_for_project(project_name)
                all_results[project_name] = result
            except Exception as e:
                print(f"Error processing project {project_name}: {e}")
                all_results[project_name] = {
                    'status': 'error',
                    'error': str(e),
                    'project': project_name
                }
        
        return {
            'status': 'completed',
            'total_projects': len(project_names),
            'results': all_results
        }
    
    def _generate_embeddings_parallel(self, project_names: List[str]) -> Dict[str, Any]:
        """Generate embeddings in parallel."""
        import concurrent.futures
        
        all_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all projects
            future_to_project = {
                executor.submit(self.generate_embeddings_for_project, project_name): project_name
                for project_name in project_names
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_project):
                project_name = future_to_project[future]
                try:
                    result = future.result()
                    all_results[project_name] = result
                    print(f"Project {project_name} completed with status: {result['status']}")
                except Exception as e:
                    print(f"Project {project_name} failed with error: {e}")
                    all_results[project_name] = {
                        'status': 'error',
                        'error': str(e),
                        'project': project_name
                    }
        
        return {
            'status': 'completed',
            'total_projects': len(project_names),
            'results': all_results
        }
    
    def _find_graphml_files(self, data_dir: str, project_name: str) -> List[Tuple[str, str]]:
        """
        Find GraphML files for a project.
        
        Args:
            data_dir: Data directory
            project_name: Project name
            
        Returns:
            List of (before_file, after_file) tuples
        """
        project_dir = os.path.join(data_dir, project_name)
        
        if not os.path.exists(project_dir):
            print(f"Project directory not found: {project_dir}")
            return []
        
        # Look for before/after GraphML files
        graphml_files = []
        
        for filename in os.listdir(project_dir):
            if filename.endswith('.xml') and '_before_graphml.xml' in filename:
                before_file = os.path.join(project_dir, filename)
                after_file = before_file.replace('_before_graphml.xml', '_after_graphml.xml')
                
                if os.path.exists(after_file):
                    graphml_files.append((before_file, after_file))
        
        return graphml_files
    
    def train_word2vec_model(self, file_paths: List[str], 
                           output_path: Optional[str] = None) -> str:
        """
        Train Word2Vec model on node features.
        
        Args:
            file_paths: List of file paths to process
            output_path: Output path for the model (uses config default if None)
            
        Returns:
            Path to the saved model
        """
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, 'word2vec_model.model')
        
        print(f"Training Word2Vec model on {len(file_paths)} files...")
        
        # Train the model
        model = self.word2vec_embedder.train_word2vec_model(file_paths)
        
        # Save the model
        self.word2vec_embedder.save_model(output_path)
        
        print(f"Word2Vec model trained and saved to: {output_path}")
        return output_path
    
    def load_word2vec_model(self, model_path: str) -> None:
        """
        Load a trained Word2Vec model.
        
        Args:
            model_path: Path to the model file
        """
        self.word2vec_embedder.load_model(model_path)
    
    def get_embedding_summary(self) -> Dict[str, Any]:
        """
        Get summary of all embedders and their status.
        
        Returns:
            Dictionary containing embedder summaries
        """
        summary = {
            'word2vec': self.word2vec_embedder.get_model_info(),
            'node2vec': self.node2vec_embedder.get_model_info(),
            'codebert': self.codebert_embedder.get_model_info(),
            'config': self.config.to_dict()
        }
        
        return summary
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate the setup and configuration.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        # Check data directory
        if not os.path.exists(self.config.data_dir):
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Data directory does not exist: {self.config.data_dir}")
        else:
            validation_results['checks']['data_dir'] = 'OK'
        
        # Check output directory
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            validation_results['checks']['output_dir'] = 'OK'
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Cannot create output directory: {e}")
        
        # Check CodeBERT model path
        if not os.path.exists(self.config.get_codebert_model_path()):
            validation_results['warnings'].append(f"CodeBERT model path may not exist: {self.config.get_codebert_model_path()}")
        else:
            validation_results['checks']['codebert_model'] = 'OK'
        
        # Check device availability
        if self.config.device == 'cuda' and not torch.cuda.is_available():
            validation_results['warnings'].append("CUDA requested but not available")
            validation_results['checks']['device'] = 'CPU fallback'
        else:
            validation_results['checks']['device'] = self.config.device
        
        return validation_results
    
    def cleanup_temp_files(self, output_dir: Optional[str] = None) -> None:
        """
        Clean up temporary files.
        
        Args:
            output_dir: Output directory to clean (uses config default if None)
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        print(f"Cleaning up temporary files in: {output_dir}")
        
        # Remove temporary files if they exist
        temp_extensions = ['.tmp', '.temp', '.cache']
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if any(file.endswith(ext) for ext in temp_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Removed temporary file: {file_path}")
                    except Exception as e:
                        print(f"Could not remove temporary file {file_path}: {e}")
    
    def get_project_status(self, project_name: str) -> Dict[str, Any]:
        """
        Get the status of embeddings for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary containing project status
        """
        output_dir = self.config.get_project_output_path(project_name)
        
        if not os.path.exists(output_dir):
            return {
                'project': project_name,
                'status': 'not_started',
                'output_dir': output_dir,
                'files': []
            }
        
        # Count output files
        files = []
        for filename in os.listdir(output_dir):
            if filename.endswith('.pt') or filename.endswith('.txt'):
                file_path = os.path.join(output_dir, filename)
                file_size = os.path.getsize(file_path)
                files.append({
                    'name': filename,
                    'size': file_size,
                    'path': file_path
                })
        
        return {
            'project': project_name,
            'status': 'completed' if files else 'in_progress',
            'output_dir': output_dir,
            'files': files,
            'total_files': len(files)
        }

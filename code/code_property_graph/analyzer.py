"""
Code Property Graph analysis and filtering functionality.
"""

import networkx as nx
import pandas as pd
from typing import List, Dict, Set, Optional, Any, Tuple
from collections import defaultdict

from .config import CPGConfig
from .utils import (
    clean_java_qualified_name, load_cpg_from_graphml, get_node_property,
    get_edge_property, filter_nodes_by_property, get_nodes_by_type
)


class CPGAnalyzer:
    """
    Class for analyzing and filtering Code Property Graphs.
    
    Code Reference: cpgs_script.py (lines 30-60)
    """
    
    def __init__(self, config: Optional[CPGConfig] = None):
        """
        Initialize the CPG analyzer.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or CPGConfig()
    
    def filter_cpg_by_long_names(self, graph: nx.Graph, long_names: List[str]) -> nx.Graph:
        """
        Filter CPG to include only nodes related to specified long names.
        
        Args:
            graph: NetworkX graph object
            long_names: List of Java qualified names to filter by
            
        Returns:
            Filtered subgraph containing only related nodes
            
        Code Reference: cpgs_script.py (lines 30-45)
        """
        # Clean the long_names to match the graph's naming convention
        ces_names = [clean_java_qualified_name(name) for name in long_names]
        
        # Find nodes that match the cleaned long_names
        start_nodes = [n for n, d in graph.nodes(data=True) 
                      if d.get('name') in ces_names]
        
        if not start_nodes:
            print(f"Warning: No nodes found matching the provided long names: {long_names}")
            return nx.create_empty_copy(graph)
        
        # Use DFS to find all related nodes
        related_nodes = set()
        for start_node in start_nodes:
            for node in nx.dfs_preorder_nodes(graph, source=start_node):
                related_nodes.add(node)
        
        # Create a subgraph with the related nodes
        subgraph = graph.subgraph(related_nodes)
        print(f"Filtered CPG: {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges")
        
        return subgraph
    
    def filter_cpg_by_commit_changes(self, graph: nx.Graph, 
                                   added_files: Optional[List[str]] = None,
                                   removed_files: Optional[List[str]] = None,
                                   modified_files: Optional[List[str]] = None) -> nx.Graph:
        """
        Filter CPG based on commit changes (added, removed, modified files).
        
        Args:
            graph: NetworkX graph object
            added_files: List of added file paths
            removed_files: List of removed file paths
            modified_files: List of modified file paths
            
        Returns:
            Filtered subgraph containing only nodes from specified files
        """
        target_files = set()
        if added_files:
            target_files.update(added_files)
        if removed_files:
            target_files.update(removed_files)
        if modified_files:
            target_files.update(modified_files)
        
        if not target_files:
            return graph
        
        # Filter nodes by filename
        related_nodes = set()
        for node, data in graph.nodes(data=True):
            filename = data.get('FILENAME', '')
            if filename in target_files:
                related_nodes.add(node)
        
        if not related_nodes:
            print(f"Warning: No nodes found for the specified files: {target_files}")
            return nx.create_empty_copy(graph)
        
        # Create subgraph
        subgraph = graph.subgraph(related_nodes)
        print(f"Filtered CPG by file changes: {len(subgraph.nodes())} nodes, {len(subgraph.edges())} edges")
        
        return subgraph
    
    def analyze_cpg_structure(self, graph: nx.Graph) -> Dict[str, any]:
        """
        Analyze the structure of a CPG graph.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary containing structural analysis
        """
        analysis = {
            'total_nodes': graph.number_of_nodes(),
            'total_edges': graph.number_of_edges(),
            'node_types': defaultdict(int),
            'edge_types': defaultdict(int),
            'file_coverage': set(),
            'package_coverage': set(),
            'method_count': 0,
            'class_count': 0,
            'variable_count': 0
        }
        
        # Analyze nodes
        for node, data in graph.nodes(data=True):
            node_type = data.get('labelV', 'UNKNOWN')
            analysis['node_types'][node_type] += 1
            
            # Count specific types
            if node_type == 'METHOD':
                analysis['method_count'] += 1
            elif node_type == 'TYPE_DECL':
                analysis['class_count'] += 1
            elif node_type == 'IDENTIFIER':
                analysis['variable_count'] += 1
            
            # Collect file and package information
            filename = data.get('FILENAME', '')
            if filename:
                analysis['file_coverage'].add(filename)
            
            package_name = data.get('PACKAGE_NAME', '')
            if package_name:
                analysis['package_coverage'].add(package_name)
        
        # Analyze edges
        for source, target, data in graph.edges(data=True):
            edge_type = data.get('labelE', 'UNKNOWN')
            analysis['edge_types'][edge_type] += 1
        
        # Convert sets to lists for JSON serialization
        analysis['file_coverage'] = list(analysis['file_coverage'])
        analysis['package_coverage'] = list(analysis['package_coverage'])
        
        return analysis
    
    def find_code_smells(self, graph: nx.Graph) -> Dict[str, List[Dict[str, any]]]:
        """
        Detect potential code smells in the CPG.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary mapping smell types to detected instances
        """
        smells = {
            'long_methods': [],
            'large_classes': [],
            'complex_conditions': [],
            'duplicate_code': [],
            'unused_variables': []
        }
        
        # Detect long methods
        for node, data in graph.nodes(data=True):
            if data.get('labelV') == 'METHOD':
                method_name = data.get('NAME', 'unknown')
                method_code = data.get('CODE', '')
                
                if method_code:
                    lines = method_code.count('\n') + 1
                    if lines > 50:  # Threshold for long method
                        smells['long_methods'].append({
                            'node_id': node,
                            'method_name': method_name,
                            'line_count': lines,
                            'code_preview': method_code[:200] + '...' if len(method_code) > 200 else method_code
                        })
        
        # Detect large classes
        for node, data in graph.nodes(data=True):
            if data.get('labelV') == 'TYPE_DECL':
                class_name = data.get('NAME', 'unknown')
                
                # Count methods in this class
                method_count = 0
                for neighbor in graph.predecessors(node):
                    neighbor_data = graph.nodes[neighbor]
                    if neighbor_data.get('labelV') == 'METHOD':
                        method_count += 1
                
                if method_count > 20:  # Threshold for large class
                    smells['large_classes'].append({
                        'node_id': node,
                        'class_name': class_name,
                        'method_count': method_count
                    })
        
        # Detect complex conditions
        for node, data in graph.nodes(data=True):
            if data.get('labelV') == 'CONTROL_STRUCTURE':
                control_type = data.get('CONTROL_STRUCTURE_TYPE', '')
                if control_type in ['IF', 'WHILE', 'FOR']:
                    # Count conditions
                    condition_count = 0
                    for neighbor in graph.successors(node):
                        neighbor_data = graph.nodes[neighbor]
                        if neighbor_data.get('labelV') == 'CALL':
                            method_name = neighbor_data.get('NAME', '')
                            if method_name in ['equals', 'contains', 'startsWith', 'endsWith']:
                                condition_count += 1
                    
                    if condition_count > 3:  # Threshold for complex condition
                        smells['complex_conditions'].append({
                            'node_id': node,
                            'control_type': control_type,
                            'condition_count': condition_count
                        })
        
        return smells
    
    def extract_call_graph(self, graph: nx.Graph) -> nx.DiGraph:
        """
        Extract call graph from the CPG.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Directed graph representing method calls
        """
        call_graph = nx.DiGraph()
        
        # Find all method nodes
        method_nodes = get_nodes_by_type(graph, 'METHOD')
        
        for method_node in method_nodes:
            method_data = graph.nodes[method_node]
            method_name = method_data.get('NAME', 'unknown')
            class_name = method_data.get('CLASS_NAME', 'unknown')
            full_name = f"{class_name}.{method_name}"
            
            # Add method to call graph
            call_graph.add_node(method_node, 
                              name=method_name, 
                              class_name=class_name, 
                              full_name=full_name)
        
        # Find call relationships
        for source, target, data in graph.edges(data=True):
            if data.get('labelE') == 'CALL':
                source_node = source
                target_node = target
                
                # Check if both are method nodes
                if (source_node in method_nodes and target_node in method_nodes):
                    call_graph.add_edge(source_node, target_node)
        
        return call_graph
    
    def analyze_method_complexity(self, graph: nx.Graph) -> Dict[str, Dict[str, any]]:
        """
        Analyze cyclomatic complexity of methods.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary mapping method IDs to complexity metrics
        """
        complexity_metrics = {}
        
        method_nodes = get_nodes_by_type(graph, 'METHOD')
        
        for method_node in method_nodes:
            method_data = graph.nodes[method_node]
            method_name = method_data.get('NAME', 'unknown')
            
            # Calculate cyclomatic complexity
            complexity = 1  # Base complexity
            
            # Count control structures
            for node in nx.descendants(graph, method_node):
                node_data = graph.nodes[node]
                if node_data.get('labelV') == 'CONTROL_STRUCTURE':
                    control_type = node_data.get('CONTROL_STRUCTURE_TYPE', '')
                    if control_type in ['IF', 'WHILE', 'FOR', 'CATCH', 'CASE']:
                        complexity += 1
            
            # Count logical operators
            for node in nx.descendants(graph, method_node):
                node_data = graph.nodes[node]
                if node_data.get('labelV') == 'CALL':
                    method_call = node_data.get('NAME', '')
                    if method_call in ['and', 'or', 'not']:
                        complexity += 1
            
            complexity_metrics[method_node] = {
                'method_name': method_name,
                'cyclomatic_complexity': complexity,
                'complexity_level': self._get_complexity_level(complexity)
            }
        
        return complexity_metrics
    
    def _get_complexity_level(self, complexity: int) -> str:
        """Get complexity level based on cyclomatic complexity value."""
        if complexity <= 5:
            return 'LOW'
        elif complexity <= 10:
            return 'MEDIUM'
        elif complexity <= 20:
            return 'HIGH'
        else:
            return 'VERY_HIGH'
    
    def find_data_flow_paths(self, graph: nx.Graph, 
                            start_node: str, 
                            end_node: str) -> List[List[str]]:
        """
        Find data flow paths between two nodes.
        
        Args:
            graph: NetworkX graph object
            start_node: Starting node ID
            end_node: Ending node ID
            
        Returns:
            List of paths between the nodes
        """
        if not (graph.has_node(start_node) and graph.has_node(end_node)):
            return []
        
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(graph, start_node, end_node))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def analyze_dependencies(self, graph: nx.Graph) -> Dict[str, any]:
        """
        Analyze dependencies in the CPG.
        
        Args:
            graph: NetworkX graph object
            
        Returns:
            Dictionary containing dependency analysis
        """
        dependencies = {
            'imports': [],
            'external_calls': [],
            'inheritance': [],
            'implementations': []
        }
        
        # Find imports
        for node, data in graph.nodes(data=True):
            if data.get('labelV') == 'IMPORT':
                import_name = data.get('CODE', '')
                if import_name:
                    dependencies['imports'].append({
                        'node_id': node,
                        'import_name': import_name
                    })
        
        # Find external calls
        for source, target, data in graph.edges(data=True):
            if data.get('labelE') == 'CALL':
                target_data = graph.nodes[target]
                if target_data.get('IS_EXTERNAL', False):
                    dependencies['external_calls'].append({
                        'source': source,
                        'target': target,
                        'method_name': target_data.get('NAME', 'unknown')
                    })
        
        # Find inheritance relationships
        for source, target, data in graph.edges(data=True):
            if data.get('labelE') == 'INHERITS_FROM':
                source_data = graph.nodes[source]
                target_data = graph.nodes[target]
                dependencies['inheritance'].append({
                    'child_class': source_data.get('NAME', 'unknown'),
                    'parent_class': target_data.get('NAME', 'unknown')
                })
        
        return dependencies
    
    def generate_cpg_report(self, graph: nx.Graph, 
                           project_name: str = "unknown") -> Dict[str, any]:
        """
        Generate a comprehensive CPG analysis report.
        
        Args:
            graph: NetworkX graph object
            project_name: Name of the project
            
        Returns:
            Dictionary containing the complete analysis report
        """
        report = {
            'project_name': project_name,
            'timestamp': pd.Timestamp.now().isoformat(),
            'graph_statistics': self.analyze_cpg_structure(graph),
            'code_smells': self.find_code_smells(graph),
            'method_complexity': self.analyze_method_complexity(graph),
            'dependencies': self.analyze_dependencies(graph),
            'summary': {}
        }
        
        # Generate summary
        total_methods = report['graph_statistics']['method_count']
        high_complexity_methods = sum(
            1 for metrics in report['method_complexity'].values()
            if metrics['complexity_level'] in ['HIGH', 'VERY_HIGH']
        )
        
        report['summary'] = {
            'total_methods': total_methods,
            'high_complexity_methods': high_complexity_methods,
            'complexity_ratio': high_complexity_methods / total_methods if total_methods > 0 else 0,
            'total_smells': sum(len(smells) for smells in report['code_smells'].values()),
            'file_coverage': len(report['graph_statistics']['file_coverage']),
            'package_coverage': len(report['graph_statistics']['package_coverage'])
        }
        
        return report

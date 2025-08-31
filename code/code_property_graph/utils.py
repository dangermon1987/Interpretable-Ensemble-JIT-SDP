"""
Utility functions for Code Property Graph processing and analysis.
"""

import re
import networkx as nx
from typing import List, Set, Optional, Any


def clean_java_qualified_name(name: str) -> str:
    """
    Clean Java qualified names for better readability and matching.
    
    Args:
        name: Java qualified name string
        
    Returns:
        Cleaned Java qualified name
        
    Code Reference: cpgs_script.py (lines 20-30)
    """
    # Remove return type character if it's there
    name = re.sub(r'\)[VZBSIFJD]$', ')', name)
    
    # Replace type signatures in parameters
    def clean_params(match):
        params = match.group(0)
        # Replace class type identifiers and slashes for readability
        params = re.sub(r'L(.*?);', r'\1', params)
        params = params.replace('/', '.')
        return params
    
    # Apply cleaning to parameters
    cleaned_name = re.sub(r'\(.*?\)', clean_params, name)
    
    return cleaned_name


def load_cpg_from_graphml(file_path: str) -> nx.Graph:
    """
    Load a CPG from a GraphML file.
    
    Args:
        file_path: Path to the GraphML file
        
    Returns:
        NetworkX graph object
        
    Code Reference: cpgs_script.py (lines 15-17)
    """
    return nx.read_graphml(file_path)


def save_cpg_to_graphml(graph: nx.Graph, file_path: str) -> None:
    """
    Save a CPG to a GraphML file.
    
    Args:
        graph: NetworkX graph object to save
        file_path: Output file path
    """
    nx.write_graphml(graph, file_path)


def get_node_property(graph: nx.Graph, node_id: Any, property_name: str, default: Any = None) -> Any:
    """
    Safely get a node property from the graph.
    
    Args:
        graph: NetworkX graph object
        node_id: Node identifier
        property_name: Name of the property to retrieve
        default: Default value if property doesn't exist
        
    Returns:
        Property value or default
    """
    if graph.has_node(node_id):
        return graph.nodes[node_id].get(property_name, default)
    return default


def get_edge_property(graph: nx.Graph, source: Any, target: Any, property_name: str, default: Any = None) -> Any:
    """
    Safely get an edge property from the graph.
    
    Args:
        graph: NetworkX graph object
        source: Source node identifier
        target: Target node identifier
        property_name: Name of the property to retrieve
        default: Default value if property doesn't exist
        
    Returns:
        Property value or default
    """
    if graph.has_edge(source, target):
        return graph.edges[source, target].get(property_name, default)
    return default


def filter_nodes_by_property(graph: nx.Graph, property_name: str, property_value: Any) -> List[Any]:
    """
    Filter nodes by a specific property value.
    
    Args:
        graph: NetworkX graph object
        property_name: Name of the property to filter by
        property_value: Value to filter for
        
    Returns:
        List of node IDs that match the criteria
    """
    return [node for node, data in graph.nodes(data=True) 
            if data.get(property_name) == property_value]


def filter_nodes_by_property_contains(graph: nx.Graph, property_name: str, substring: str) -> List[Any]:
    """
    Filter nodes by a property containing a substring.
    
    Args:
        graph: NetworkX graph object
        property_name: Name of the property to filter by
        substring: Substring to search for
        
    Returns:
        List of node IDs that match the criteria
    """
    return [node for node, data in graph.nodes(data=True) 
            if substring.lower() in str(data.get(property_name, '')).lower()]


def get_nodes_by_type(graph: nx.Graph, node_type: str) -> List[Any]:
    """
    Get all nodes of a specific type.
    
    Args:
        graph: NetworkX graph object
        node_type: Type of nodes to retrieve
        
    Returns:
        List of node IDs of the specified type
    """
    return filter_nodes_by_property(graph, 'labelV', node_type)


def get_method_nodes(graph: nx.Graph) -> List[Any]:
    """
    Get all method nodes from the graph.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        List of method node IDs
    """
    return get_nodes_by_type(graph, 'METHOD')


def get_class_nodes(graph: nx.Graph) -> List[Any]:
    """
    Get all class nodes from the graph.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        List of class node IDs
    """
    return get_nodes_by_type(graph, 'TYPE_DECL')


def get_variable_nodes(graph: nx.Graph) -> List[Any]:
    """
    Get all variable nodes from the graph.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        List of variable node IDs
    """
    return get_nodes_by_type(graph, 'IDENTIFIER')


def count_nodes_by_type(graph: nx.Graph) -> dict:
    """
    Count nodes by their type.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        Dictionary mapping node types to counts
    """
    type_counts = {}
    for _, data in graph.nodes(data=True):
        node_type = data.get('labelV', 'UNKNOWN')
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    return type_counts


def get_graph_statistics(graph: nx.Graph) -> dict:
    """
    Get comprehensive statistics about the graph.
    
    Args:
        graph: NetworkX graph object
        
    Returns:
        Dictionary containing graph statistics
    """
    return {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'node_types': count_nodes_by_type(graph),
        'is_directed': graph.is_directed(),
        'is_multigraph': graph.is_multigraph(),
        'density': nx.density(graph) if graph.number_of_nodes() > 1 else 0.0
    }


def validate_cpg_structure(graph: nx.Graph, required_properties: Optional[List[str]] = None) -> dict:
    """
    Validate the structure of a CPG graph.
    
    Args:
        graph: NetworkX graph object to validate
        required_properties: List of required node properties (optional)
        
    Returns:
        Dictionary containing validation results
    """
    if required_properties is None:
        required_properties = ['labelV', 'CODE', 'FULL_NAME']
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'missing_properties': []
    }
    
    # Check if graph has nodes
    if graph.number_of_nodes() == 0:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Graph has no nodes")
    
    # Check required properties
    for prop in required_properties:
        missing_count = 0
        for _, data in graph.nodes(data=True):
            if prop not in data:
                missing_count += 1
        
        if missing_count > 0:
            validation_results['warnings'].append(f"Property '{prop}' missing from {missing_count} nodes")
            validation_results['missing_properties'].append(prop)
    
    return validation_results

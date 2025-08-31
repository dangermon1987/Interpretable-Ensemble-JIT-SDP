"""
Utility functions for node embedding generation and processing.
"""

import os
import networkx as nx
import torch
from typing import List, Dict, Set, Optional, Any, Tuple
from pathlib import Path

from .config import EmbeddingConfig


def load_graph_from_graphml(graphml_file: str) -> nx.Graph:
    """
    Load a CPG from a GraphML file.
    
    Args:
        graphml_file: Path to the GraphML file
        
    Returns:
        NetworkX graph object
        
    Code Reference: new_embedding/GenGraphData.py (lines 62-65)
    """
    if not os.path.exists(graphml_file):
        raise FileNotFoundError(f"GraphML file does not exist: {graphml_file}")
    
    G = nx.read_graphml(graphml_file, force_multigraph=True)
    return G


def update_graph_to_k_distance(G: nx.Graph, k: int = 1) -> nx.Graph:
    """
    Filter graph to include only nodes within k-distance of changed nodes.
    
    Args:
        G: NetworkX graph object
        k: Distance threshold for filtering
        
    Returns:
        Filtered subgraph containing relevant nodes
        
    Code Reference: new_embedding/GenGraphData.py (lines 40-60)
    """
    # Identify all changed nodes (state = before or after)
    changed_nodes = set(node_id for node_id, node_data in G.nodes(data=True)
                        if node_data.get('state', 'unchanged') in ['before', 'after'])
    
    if not changed_nodes:
        # If no changed nodes, return original graph
        return G.copy()
    
    # Find all nodes within k-distance from changed nodes
    nodes_to_keep = set(changed_nodes)  # Start with the changed nodes themselves
    for node_id in changed_nodes:
        # Find nodes within k-distance from this node
        nodes_within_k_distance = nx.single_source_shortest_path_length(G, node_id, cutoff=k).keys()
        nodes_to_keep.update(nodes_within_k_distance)
    
    # Create a subgraph containing only the relevant nodes and edges
    G_subgraph = G.subgraph(nodes_to_keep).copy()
    
    return G_subgraph


def extract_node_features(G: nx.Graph, config: Optional[EmbeddingConfig] = None) -> Tuple[List[List[str]], Dict[Any, List[str]]]:
    """
    Extract node features from CPG graph.
    
    Args:
        G: NetworkX graph object
        config: Configuration object (uses default if None)
        
    Returns:
        Tuple of (list of feature lists, dictionary mapping node IDs to feature lists)
        
    Code Reference: word2vectrain.py (lines 79-101)
    """
    if config is None:
        config = EmbeddingConfig()
    
    node_features = []
    node_feature_dict = {}
    
    for node in G.nodes(data=True):
        node_id, attributes = node
        state = attributes.get('state', 'unchanged')
        node_type = attributes.get('labelV', 'default_type')
        code = attributes.get('CODE', '')
        
        # Combine all CPG properties
        other_features = ' '.join([str(attributes.get(prop, '')) for prop in config.cpg_properties])
        
        # Create feature string and split into tokens
        node_feature = f"{state} {node_type} {code} {other_features}".split()
        node_features.append(node_feature)
        node_feature_dict[node_id] = node_feature
    
    return node_features, node_feature_dict


def combine_embeddings(node2vec_embeddings: Dict[Any, List[float]], 
                      node_text_embeddings: Dict[Any, Tuple[str, torch.Tensor]]) -> Dict[Any, Tuple[str, torch.Tensor]]:
    """
    Combine Node2Vec and text embeddings.
    
    Args:
        node2vec_embeddings: Dictionary of Node2Vec embeddings
        node_text_embeddings: Dictionary of text embeddings with node type
        
    Returns:
        Dictionary of combined embeddings
        
    Code Reference: new_embedding/GenGraphData.py (lines 196-203)
    """
    combined_embeddings = {}
    
    for node in node2vec_embeddings.keys():
        if node in node_text_embeddings:
            node2vec_emb = torch.tensor(node2vec_embeddings[node], dtype=torch.float)
            text_emb = node_text_embeddings[node]
            
            # Concatenate Node2Vec and text embeddings
            combined_embedding = torch.cat([node2vec_emb, text_emb[1]], dim=0)
            combined_embeddings[node] = (text_emb[0], combined_embedding)
    
    return combined_embeddings


def add_state_vector(embedding: torch.Tensor, state: str, config: Optional[EmbeddingConfig] = None) -> torch.Tensor:
    """
    Add state vector to embedding.
    
    Args:
        embedding: Input embedding tensor
        state: Node state ('before', 'unchanged', 'after')
        config: Configuration object (uses default if None)
        
    Returns:
        Embedding tensor with state vector appended
        
    Code Reference: new_embedding/GenGraphData.py (lines 170-180)
    """
    if config is None:
        config = EmbeddingConfig()
    
    state_vector = torch.zeros(1)
    state_value = config.state_encoding.get(state, 0)
    state_vector[0] = state_value
    
    return torch.cat([embedding, state_vector], dim=0)


def save_embeddings(embeddings: Dict[Any, torch.Tensor], output_path: str) -> None:
    """
    Save embeddings to file.
    
    Args:
        embeddings: Dictionary of embeddings
        output_path: Output file path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for node, embedding in embeddings.items():
            if isinstance(embedding, torch.Tensor):
                embedding_str = ' '.join(map(str, embedding.tolist()))
            else:
                embedding_str = ' '.join(map(str, embedding))
            f.write(f"{node} {embedding_str}\n")


def load_embeddings(file_path: str) -> Dict[str, List[float]]:
    """
    Load embeddings from file.
    
    Args:
        file_path: Path to embeddings file
        
    Returns:
        Dictionary mapping node IDs to embedding vectors
    """
    embeddings = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                node_id = parts[0]
                embedding_vector = [float(x) for x in parts[1:]]
                embeddings[node_id] = embedding_vector
    
    return embeddings


def validate_graph_structure(G: nx.Graph) -> Dict[str, Any]:
    """
    Validate CPG graph structure.
    
    Args:
        G: NetworkX graph object
        
    Returns:
        Dictionary containing validation results
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'node_count': G.number_of_nodes(),
        'edge_count': G.number_of_edges(),
        'missing_attributes': []
    }
    
    # Check required node attributes
    required_attrs = ['state', 'labelV', 'CODE']
    for node_id, node_data in G.nodes(data=True):
        for attr in required_attrs:
            if attr not in node_data:
                validation_results['missing_attributes'].append(f"Node {node_id} missing {attr}")
    
    if validation_results['missing_attributes']:
        validation_results['warnings'].append(f"Missing attributes in {len(validation_results['missing_attributes'])} nodes")
    
    # Check if graph has nodes
    if G.number_of_nodes() == 0:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Graph has no nodes")
    
    return validation_results


def get_graph_statistics(G: nx.Graph) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the graph.
    
    Args:
        G: NetworkX graph object
        
    Returns:
        Dictionary containing graph statistics
    """
    stats = {
        'total_nodes': G.number_of_nodes(),
        'total_edges': G.number_of_edges(),
        'node_types': {},
        'state_distribution': {},
        'is_directed': G.is_directed(),
        'is_multigraph': G.is_multigraph()
    }
    
    # Count node types and states
    for node_id, node_data in G.nodes(data=True):
        node_type = node_data.get('labelV', 'UNKNOWN')
        state = node_data.get('state', 'UNKNOWN')
        
        stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        stats['state_distribution'][state] = stats['state_distribution'].get(state, 0) + 1
    
    # Calculate density if graph has nodes
    if G.number_of_nodes() > 1:
        stats['density'] = nx.density(G)
    else:
        stats['density'] = 0.0
    
    return stats


def filter_nodes_by_state(G: nx.Graph, states: List[str]) -> nx.Graph:
    """
    Filter graph to include only nodes with specified states.
    
    Args:
        G: NetworkX graph object
        states: List of states to include
        
    Returns:
        Filtered subgraph
    """
    nodes_to_keep = set()
    
    for node_id, node_data in G.nodes(data=True):
        if node_data.get('state', 'unchanged') in states:
            nodes_to_keep.add(node_id)
    
    if not nodes_to_keep:
        return nx.create_empty_copy(G)
    
    return G.subgraph(nodes_to_keep).copy()


def get_neighborhood_subgraph(G: nx.Graph, center_nodes: List[Any], radius: int = 1) -> nx.Graph:
    """
    Get subgraph containing nodes within specified radius of center nodes.
    
    Args:
        G: NetworkX graph object
        center_nodes: List of center node IDs
        radius: Neighborhood radius
        
    Returns:
        Subgraph containing neighborhood
    """
    nodes_to_keep = set(center_nodes)
    
    for center_node in center_nodes:
        if center_node in G:
            neighborhood = nx.single_source_shortest_path_length(G, center_node, cutoff=radius).keys()
            nodes_to_keep.update(neighborhood)
    
    if not nodes_to_keep:
        return nx.create_empty_copy(G)
    
    return G.subgraph(nodes_to_keep).copy()


def normalize_embeddings(embeddings: Dict[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
    """
    Normalize embeddings using L2 normalization.
    
    Args:
        embeddings: Dictionary of embeddings
        
    Returns:
        Dictionary of normalized embeddings
    """
    normalized_embeddings = {}
    
    for node_id, embedding in embeddings.items():
        if isinstance(embedding, torch.Tensor):
            norm = torch.norm(embedding, p=2)
            if norm > 0:
                normalized_embeddings[node_id] = embedding / norm
            else:
                normalized_embeddings[node_id] = embedding
        else:
            # Convert to tensor if not already
            embedding_tensor = torch.tensor(embedding, dtype=torch.float)
            norm = torch.norm(embedding_tensor, p=2)
            if norm > 0:
                normalized_embeddings[node_id] = embedding_tensor / norm
            else:
                normalized_embeddings[node_id] = embedding_tensor
    
    return normalized_embeddings


def create_embedding_summary(embeddings: Dict[Any, torch.Tensor], 
                           G: nx.Graph) -> Dict[str, Any]:
    """
    Create summary of embeddings and graph.
    
    Args:
        embeddings: Dictionary of embeddings
        G: NetworkX graph object
        
    Returns:
        Dictionary containing embedding summary
    """
    summary = {
        'total_nodes': G.number_of_nodes(),
        'nodes_with_embeddings': len(embeddings),
        'embedding_coverage': len(embeddings) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'embedding_dimensions': {},
        'node_type_coverage': {},
        'state_coverage': {}
    }
    
    # Analyze embedding dimensions
    for node_id, embedding in embeddings.items():
        if isinstance(embedding, torch.Tensor):
            dim = embedding.shape[0]
        else:
            dim = len(embedding)
        
        summary['embedding_dimensions'][dim] = summary['embedding_dimensions'].get(dim, 0) + 1
    
    # Analyze coverage by node type and state
    for node_id in embeddings.keys():
        if node_id in G:
            node_data = G.nodes[node_id]
            node_type = node_data.get('labelV', 'UNKNOWN')
            state = node_data.get('state', 'UNKNOWN')
            
            summary['node_type_coverage'][node_type] = summary['node_type_coverage'].get(node_type, 0) + 1
            summary['state_coverage'][state] = summary['state_coverage'].get(state, 0) + 1
    
    return summary

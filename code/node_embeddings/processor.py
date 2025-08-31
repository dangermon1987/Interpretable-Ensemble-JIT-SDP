"""
Embedding processing and combination utilities.
"""

import os
import torch
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from torch_geometric.data import HeteroData

from .config import EmbeddingConfig
from .utils import update_graph_to_k_distance


class EmbeddingProcessor:
    """
    Class for processing and combining different types of embeddings.
    
    Code Reference: new_embedding/GenGraphData.py (lines 196-280)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the embedding processor.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or EmbeddingConfig()
    
    def combine_embeddings(self, node2vec_embeddings: Dict[Any, List[float]], 
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
        
        print(f"Combined embeddings for {len(combined_embeddings)} nodes")
        return combined_embeddings
    
    def create_heterogeneous_data(self, G: nx.Graph, node_embeddings: Dict[Any, Tuple[str, torch.Tensor]], 
                                 k: int = 0) -> HeteroData:
        """
        Create heterogeneous graph data from embeddings.
        
        Args:
            G: NetworkX graph object
            node_embeddings: Dictionary of node embeddings with type information
            k: K-distance for graph filtering
            
        Returns:
            PyTorch Geometric HeteroData object
            
        Code Reference: new_embedding/GenGraphData.py (lines 205-280)
        """
        hetero_data = HeteroData()
        print(f"Start building heterogeneous data at k={k}")
        
        # Filter graph to k-distance
        G_filtered = update_graph_to_k_distance(G, k=k)
        
        # Keep track of all nodes that should be included in the graph
        nodes_to_include = set()
        
        # First pass: Collect nodes involved in edges where at least one node is changed
        for src, dst, edge_data in G_filtered.edges(data=True):
            if src in node_embeddings and dst in node_embeddings:
                src_embedding = node_embeddings[src][1]
                dst_embedding = node_embeddings[dst][1]
                
                src_state = src_embedding[-1].item()  # Last element is the state vector
                dst_state = dst_embedding[-1].item()
                
                # If either node is changed, add both src and dst to the set of nodes to include
                if src_state != 0 or dst_state != 0:
                    nodes_to_include.add(src)
                    nodes_to_include.add(dst)
        
        new_node_indices = {}  # Remap the indices for the included nodes
        type_specific_counters = {}  # Counter to track indices for each node type
        
        # Process nodes: Add node features for nodes that are part of the edges
        print(f"Processing {len(nodes_to_include)} nodes")
        for node_id in nodes_to_include:
            if node_id in node_embeddings:
                node_type, embedding = node_embeddings[node_id]
                
                if node_type not in hetero_data.node_types:
                    hetero_data[node_type].x = []
                    hetero_data[node_type].id = []
                    type_specific_counters[node_type] = 0
                
                hetero_data[node_type].x.append(embedding)
                hetero_data[node_type].id.append(node_id)
                
                # Assign new index for this node in the node type
                new_node_indices[node_id] = type_specific_counters[node_type]
                type_specific_counters[node_type] += 1
        
        # Convert lists to tensors for node features
        for node_type in hetero_data.node_types:
            if len(hetero_data[node_type].x) > 0:
                hetero_data[node_type].x = torch.stack(hetero_data[node_type].x, dim=0)
        
        # Second pass: Process edges after nodes are added
        print("Processing edges")
        for src, dst, edge_data in G_filtered.edges(data=True):
            edge_type = edge_data.get('labelE', 'default_edge_type')
            
            if src in node_embeddings and dst in node_embeddings:
                src_embedding = node_embeddings[src][1]
                dst_embedding = node_embeddings[dst][1]
                
                src_state = src_embedding[-1].item()
                dst_state = dst_embedding[-1].item()
                
                # Add edge only if at least one of the nodes is changed
                if src_state != 0 or dst_state != 0:
                    src_type, _ = node_embeddings[src]
                    dst_type, _ = node_embeddings[dst]
                    edge_key = (src_type, edge_type, dst_type)
                    
                    if edge_key not in hetero_data.edge_types:
                        hetero_data[edge_key].edge_index = torch.empty((2, 0), dtype=torch.long)
                    
                    src_index = new_node_indices[src]
                    dst_index = new_node_indices[dst]
                    
                    edge_index = torch.tensor([[src_index], [dst_index]], dtype=torch.long)
                    hetero_data[edge_key].edge_index = torch.cat([hetero_data[edge_key].edge_index, edge_index], dim=1)
        
        print(f"Finished building heterogeneous data with {len(nodes_to_include)} nodes")
        return hetero_data
    
    def create_simple_heterogeneous_data(self, G: nx.Graph, 
                                       node_embeddings: Dict[Any, Tuple[str, torch.Tensor]]) -> HeteroData:
        """
        Create simple heterogeneous graph data without k-distance filtering.
        
        Args:
            G: NetworkX graph object
            node_embeddings: Dictionary of node embeddings with type information
            
        Returns:
            PyTorch Geometric HeteroData object
        """
        hetero_data = HeteroData()
        print("Start building simple heterogeneous data")
        
        # Add node features to HeteroData for relevant nodes
        for node_id, (node_type, embedding) in node_embeddings.items():
            if node_type not in hetero_data.node_types:
                hetero_data[node_type].x = []
                hetero_data[node_type].id = []
            hetero_data[node_type].x.append(embedding)
            hetero_data[node_type].id.append(node_id)
        
        # Convert lists to tensors for node features
        for node_type in hetero_data.node_types:
            if len(hetero_data[node_type].x) > 0:
                hetero_data[node_type].x = torch.stack(hetero_data[node_type].x, dim=0)
        
        # Process edges for the relevant edge types only
        print("Processing edges")
        for src, dst, edge_data in G.edges(data=True):
            edge_type = edge_data.get('labelE', 'default_edge_type')
            
            if src in node_embeddings and dst in node_embeddings:
                src_type, _ = node_embeddings[src]
                dst_type, _ = node_embeddings[dst]
                edge_key = (src_type, edge_type, dst_type)
                
                if edge_key not in hetero_data.edge_types:
                    hetero_data[edge_key].edge_index = torch.empty((2, 0), dtype=torch.long)
                
                # Use list index for node mapping
                src_index = list(node_embeddings.keys()).index(src)
                dst_index = list(node_embeddings.keys()).index(dst)
                
                edge_index = torch.tensor([[src_index], [dst_index]], dtype=torch.long)
                hetero_data[edge_key].edge_index = torch.cat([hetero_data[edge_key].edge_index, edge_index], dim=1)
        
        print("Finished building simple heterogeneous data")
        return hetero_data
    
    def create_multi_level_heterogeneous_data(self, G: nx.Graph, 
                                            node_embeddings: Dict[Any, Tuple[str, torch.Tensor]]) -> List[HeteroData]:
        """
        Create heterogeneous data at multiple k-distance levels.
        
        Args:
            G: NetworkX graph object
            node_embeddings: Dictionary of node embeddings with type information
            
        Returns:
            List of HeteroData objects for different k levels
        """
        hetero_data_list = []
        
        for k in self.config.k_distance_levels:
            print(f"Creating heterogeneous data for k={k}")
            hetero_data = self.create_heterogeneous_data(G, node_embeddings, k=k)
            hetero_data_list.append(hetero_data)
        
        return hetero_data_list
    
    def save_heterogeneous_data(self, hetero_data: HeteroData, output_path: str) -> None:
        """
        Save heterogeneous data to file.
        
        Args:
            hetero_data: HeteroData object to save
            output_path: Output file path
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(hetero_data, output_path)
        print(f"Heterogeneous data saved to: {output_path}")
    
    def load_heterogeneous_data(self, file_path: str) -> HeteroData:
        """
        Load heterogeneous data from file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Loaded HeteroData object
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        hetero_data = torch.load(file_path)
        print(f"Heterogeneous data loaded from: {file_path}")
        return hetero_data
    
    def get_embedding_statistics(self, node_embeddings: Dict[Any, Tuple[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Get statistics about the embeddings.
        
        Args:
            node_embeddings: Dictionary of node embeddings
            
        Returns:
            Dictionary containing embedding statistics
        """
        stats = {
            'total_nodes': len(node_embeddings),
            'node_types': {},
            'embedding_dimensions': {},
            'state_distribution': {}
        }
        
        for node_id, (node_type, embedding) in node_embeddings.items():
            # Count node types
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
            
            # Count embedding dimensions
            if isinstance(embedding, torch.Tensor):
                dim = embedding.shape[0]
            else:
                dim = len(embedding)
            stats['embedding_dimensions'][dim] = stats['embedding_dimensions'].get(dim, 0) + 1
            
            # Count states (last element of embedding)
            if isinstance(embedding, torch.Tensor) and embedding.shape[0] > 0:
                state_value = embedding[-1].item()
                if state_value == -1:
                    state = 'before'
                elif state_value == 0:
                    state = 'unchanged'
                elif state_value == 1:
                    state = 'after'
                else:
                    state = 'unknown'
                stats['state_distribution'][state] = stats['state_distribution'].get(state, 0) + 1
        
        return stats
    
    def validate_embeddings(self, node_embeddings: Dict[Any, Tuple[str, torch.Tensor]]) -> Dict[str, Any]:
        """
        Validate the structure and content of embeddings.
        
        Args:
            node_embeddings: Dictionary of node embeddings
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'total_nodes': len(node_embeddings),
            'valid_nodes': 0
        }
        
        for node_id, (node_type, embedding) in node_embeddings.items():
            try:
                # Check if embedding is a tensor
                if not isinstance(embedding, torch.Tensor):
                    validation_results['warnings'].append(f"Node {node_id}: embedding is not a tensor")
                    continue
                
                # Check if embedding has the expected structure
                if embedding.dim() != 1:
                    validation_results['errors'].append(f"Node {node_id}: embedding is not 1-dimensional")
                    validation_results['is_valid'] = False
                    continue
                
                # Check if embedding has state information (last element)
                if embedding.shape[0] < 1:
                    validation_results['errors'].append(f"Node {node_id}: embedding has no dimensions")
                    validation_results['is_valid'] = False
                    continue
                
                # Check state value
                state_value = embedding[-1].item()
                if state_value not in [-1, 0, 1]:
                    validation_results['warnings'].append(f"Node {node_id}: invalid state value {state_value}")
                
                validation_results['valid_nodes'] += 1
                
            except Exception as e:
                validation_results['errors'].append(f"Node {node_id}: error during validation - {e}")
                validation_results['is_valid'] = False
        
        return validation_results
    
    def filter_embeddings_by_state(self, node_embeddings: Dict[Any, Tuple[str, torch.Tensor]], 
                                  states: List[str]) -> Dict[Any, Tuple[str, torch.Tensor]]:
        """
        Filter embeddings to include only nodes with specified states.
        
        Args:
            node_embeddings: Dictionary of node embeddings
            states: List of states to include
            
        Returns:
            Filtered dictionary of embeddings
        """
        state_values = [self.config.state_encoding.get(state, 0) for state in states]
        
        filtered_embeddings = {}
        for node_id, (node_type, embedding) in node_embeddings.items():
            if isinstance(embedding, torch.Tensor) and embedding.shape[0] > 0:
                state_value = embedding[-1].item()
                if state_value in state_values:
                    filtered_embeddings[node_id] = (node_type, embedding)
        
        print(f"Filtered embeddings: {len(filtered_embeddings)} nodes with states {states}")
        return filtered_embeddings
    
    def normalize_embeddings(self, node_embeddings: Dict[Any, Tuple[str, torch.Tensor]]) -> Dict[Any, Tuple[str, torch.Tensor]]:
        """
        Normalize embeddings using L2 normalization.
        
        Args:
            node_embeddings: Dictionary of node embeddings
            
        Returns:
            Dictionary of normalized embeddings
        """
        normalized_embeddings = {}
        
        for node_id, (node_type, embedding) in node_embeddings.items():
            if isinstance(embedding, torch.Tensor):
                # Separate state vector from embedding
                state_vector = embedding[-1:]
                feature_embedding = embedding[:-1]
                
                # Normalize feature embedding
                norm = torch.norm(feature_embedding, p=2)
                if norm > 0:
                    normalized_feature = feature_embedding / norm
                else:
                    normalized_feature = feature_embedding
                
                # Recombine with state vector
                normalized_embedding = torch.cat([normalized_feature, state_vector], dim=0)
                normalized_embeddings[node_id] = (node_type, normalized_embedding)
            else:
                normalized_embeddings[node_id] = (node_type, embedding)
        
        print("Embeddings normalized using L2 normalization")
        return normalized_embeddings

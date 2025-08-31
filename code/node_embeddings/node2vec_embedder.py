"""
Node2Vec-based structural embedding generation for CPG nodes.
"""

import networkx as nx
import torch
from typing import List, Dict, Any, Optional
from node2vec import Node2Vec

from .config import EmbeddingConfig


class Node2VecEmbedder:
    """
    Node2Vec-based structural embedding generator for CPG nodes.
    
    Code Reference: new_embedding/GenGraphData.py (lines 68-74)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the Node2Vec embedder.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or EmbeddingConfig()
        self.model = None
    
    def generate_embeddings(self, G: nx.Graph,
                           dimensions: Optional[int] = None,
                           walk_length: Optional[int] = None,
                           num_walks: Optional[int] = None,
                           window: Optional[int] = None,
                           min_count: Optional[int] = None,
                           batch_words: Optional[int] = None,
                           workers: Optional[int] = None,
                           seed: Optional[int] = None) -> Dict[Any, List[float]]:
        """
        Generate Node2Vec embeddings for nodes in a graph.
        
        Args:
            G: NetworkX graph object
            dimensions: Embedding dimensions (uses config default if None)
            walk_length: Length of random walks (uses config default if None)
            num_walks: Number of walks per node (uses config default if None)
            window: Context window size (uses config default if None)
            min_count: Minimum word count (uses config default if None)
            batch_words: Batch size for word processing (uses config default if None)
            workers: Number of workers (uses config default if None)
            seed: Random seed (uses config default if None)
            
        Returns:
            Dictionary mapping node IDs to embedding vectors
        """
        # Use config defaults if not specified
        dimensions = dimensions or self.config.node2vec_dimensions
        walk_length = walk_length or self.config.node2vec_walk_length
        num_walks = num_walks or self.config.node2vec_num_walks
        window = window or self.config.node2vec_window
        min_count = min_count or self.config.node2vec_min_count
        batch_words = batch_words or self.config.node2vec_batch_words
        workers = workers or self.config.node2vec_workers
        seed = seed or self.config.node2vec_seed
        
        print(f"Generating Node2Vec embeddings with parameters:")
        print(f"  Dimensions: {dimensions}")
        print(f"  Walk length: {walk_length}")
        print(f"  Number of walks: {num_walks}")
        print(f"  Window: {window}")
        print(f"  Min count: {min_count}")
        print(f"  Batch words: {batch_words}")
        print(f"  Workers: {workers}")
        print(f"  Seed: {seed}")
        
        # Initialize Node2Vec
        node2vec = Node2Vec(
            G, 
            dimensions=dimensions, 
            walk_length=walk_length, 
            num_walks=num_walks, 
            workers=workers,
            seed=seed
        )
        
        # Fit the model
        print("Fitting Node2Vec model...")
        self.model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
        
        # Generate embeddings for each node
        print("Extracting embeddings...")
        node2vec_embeddings = {}
        for node in G.nodes():
            try:
                embedding = self.model.wv[str(node)]
                node2vec_embeddings[node] = embedding.tolist()
            except Exception as e:
                print(f"Error extracting embedding for node {node}: {e}")
                # Fallback: zero vector
                node2vec_embeddings[node] = [0.0] * dimensions
        
        print(f"Generated Node2Vec embeddings for {len(node2vec_embeddings)} nodes")
        return node2vec_embeddings
    
    def generate_embeddings_with_custom_params(self, G: nx.Graph, **kwargs) -> Dict[Any, List[float]]:
        """
        Generate embeddings with custom parameters.
        
        Args:
            G: NetworkX graph object
            **kwargs: Custom parameters for Node2Vec
            
        Returns:
            Dictionary mapping node IDs to embedding vectors
        """
        return self.generate_embeddings(G, **kwargs)
    
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensions of the embeddings.
        
        Returns:
            Embedding dimensions
        """
        if self.model is None:
            return self.config.node2vec_dimensions
        return self.model.vector_size
    
    def get_node_embedding(self, node_id: Any) -> Optional[List[float]]:
        """
        Get embedding for a specific node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Node embedding vector or None if not found
        """
        if self.model is None:
            return None
        
        try:
            embedding = self.model.wv[str(node_id)]
            return embedding.tolist()
        except KeyError:
            return None
    
    def get_similar_nodes(self, node_id: Any, topn: int = 10) -> List[tuple]:
        """
        Find nodes most similar to the given node.
        
        Args:
            node_id: Input node identifier
            topn: Number of similar nodes to return
            
        Returns:
            List of (node_id, similarity_score) tuples
        """
        if self.model is None:
            return []
        
        try:
            return self.model.wv.most_similar(str(node_id), topn=topn)
        except KeyError:
            return []
    
    def compute_node_similarity(self, node1: Any, node2: Any) -> Optional[float]:
        """
        Compute similarity between two nodes.
        
        Args:
            node1: First node identifier
            node2: Second node identifier
            
        Returns:
            Similarity score or None if nodes not found
        """
        if self.model is None:
            return None
        
        try:
            return self.model.wv.similarity(str(node1), str(node2))
        except KeyError:
            return None
    
    def get_embedding_matrix(self, node_ids: List[Any]) -> Optional[torch.Tensor]:
        """
        Get embedding matrix for a list of nodes.
        
        Args:
            node_ids: List of node identifiers
            
        Returns:
            Embedding matrix tensor or None if model not available
        """
        if self.model is None:
            return None
        
        embeddings = []
        valid_node_ids = []
        
        for node_id in node_ids:
            embedding = self.get_node_embedding(node_id)
            if embedding is not None:
                embeddings.append(embedding)
                valid_node_ids.append(node_id)
        
        if not embeddings:
            return None
        
        return torch.tensor(embeddings, dtype=torch.float), valid_node_ids
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained Node2Vec model.
        
        Args:
            file_path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save. Generate embeddings first.")
        
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save(file_path)
        print(f"Node2Vec model saved to: {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """
        Load a trained Node2Vec model.
        
        Args:
            file_path: Path to the model file
        """
        import os
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        from gensim.models import Word2Vec
        self.model = Word2Vec.load(file_path)
        print(f"Node2Vec model loaded from: {file_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {'status': 'No model available'}
        
        return {
            'status': 'Model loaded',
            'vector_size': self.model.vector_size,
            'vocabulary_size': len(self.model.wv),
            'window': getattr(self.model, 'window', 'N/A'),
            'min_count': getattr(self.model, 'min_count', 'N/A')
        }
    
    def validate_graph(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Validate graph for Node2Vec processing.
        
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
            'edge_count': G.number_of_edges()
        }
        
        # Check if graph has nodes
        if G.number_of_nodes() == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Graph has no nodes")
        
        # Check if graph has edges
        if G.number_of_edges() == 0:
            validation_results['warnings'].append("Graph has no edges - Node2Vec may not work well")
        
        # Check for isolated nodes
        isolated_nodes = list(nx.isolates(G))
        if isolated_nodes:
            validation_results['warnings'].append(f"Graph has {len(isolated_nodes)} isolated nodes")
        
        # Check graph connectivity
        if not nx.is_connected(G):
            validation_results['warnings'].append("Graph is not connected - some nodes may be unreachable")
        
        return validation_results
    
    def optimize_parameters(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Suggest optimal parameters based on graph characteristics.
        
        Args:
            G: NetworkX graph object
            
        Returns:
            Dictionary containing suggested parameters
        """
        node_count = G.number_of_nodes()
        edge_count = G.number_of_edges()
        avg_degree = edge_count / node_count if node_count > 0 else 0
        
        # Suggest walk length based on graph diameter
        try:
            diameter = nx.diameter(G)
            suggested_walk_length = min(diameter, 30)
        except:
            suggested_walk_length = 20
        
        # Suggest number of walks based on graph size
        if node_count < 1000:
            suggested_num_walks = 100
        elif node_count < 10000:
            suggested_num_walks = 200
        else:
            suggested_num_walks = 300
        
        # Suggest workers based on available cores
        import multiprocessing
        available_cores = multiprocessing.cpu_count()
        suggested_workers = min(available_cores, 8)
        
        return {
            'suggested_walk_length': suggested_walk_length,
            'suggested_num_walks': suggested_num_walks,
            'suggested_workers': suggested_workers,
            'graph_stats': {
                'node_count': node_count,
                'edge_count': edge_count,
                'avg_degree': avg_degree,
                'diameter': suggested_walk_length
            }
        }
    
    def batch_generate_embeddings(self, graphs: List[nx.Graph], 
                                 **kwargs) -> List[Dict[Any, List[float]]]:
        """
        Generate embeddings for multiple graphs in batch.
        
        Args:
            graphs: List of NetworkX graph objects
            **kwargs: Parameters for embedding generation
            
        Returns:
            List of embedding dictionaries
        """
        embeddings_list = []
        
        for i, G in enumerate(graphs):
            print(f"Processing graph {i+1}/{len(graphs)}...")
            try:
                embeddings = self.generate_embeddings(G, **kwargs)
                embeddings_list.append(embeddings)
            except Exception as e:
                print(f"Error processing graph {i+1}: {e}")
                embeddings_list.append({})
        
        return embeddings_list

"""
Word2Vec-based semantic embedding generation for CPG nodes.
"""

import os
import zipfile
import networkx as nx
import pandas as pd
import torch
from typing import List, Dict, Any, Optional, Generator
from gensim.models import Word2Vec

from .config import EmbeddingConfig
from .utils import extract_node_features


class Word2VecEmbedder:
    """
    Word2Vec-based semantic embedding generator for CPG nodes.
    
    Code Reference: word2vectrain.py (lines 79-112)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the Word2Vec embedder.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or EmbeddingConfig()
        self.model = None
    
    def unzip_without_folders(self, zip_file_path: str, extract_to_path: str) -> None:
        """
        Unzip file without preserving folder structure.
        
        Args:
            zip_file_path: Path to ZIP file
            extract_to_path: Path to extract to
        """
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                # Skip directories
                if member.endswith('/'):
                    continue
                # Create a path to extract the file, ignoring folder structure
                filename = os.path.basename(member)
                source = zip_ref.open(member)
                target = open(extract_to_path, "wb")
                with source, target:
                    target.write(source.read())
    
    def node_features_generator(self, file_paths: List[str], data_dir: Optional[str] = None) -> Generator[List[str], None, None]:
        """
        Generate node features from GraphML files.
        
        Args:
            file_paths: List of file paths to process
            data_dir: Optional custom data directory
            
        Yields:
            List of feature tokens for each node
        """
        if data_dir is None:
            data_dir = self.config.data_dir
        
        for file_path in file_paths:
            try:
                # Parse project and commit from filename
                if '_' in file_path:
                    project = file_path.split('_')[0]
                    commit_hash = file_path.split('_')[1]
                else:
                    project = "unknown"
                    commit_hash = file_path
                
                # Construct path to GraphML ZIP file
                zip_path = os.path.join(data_dir, project, f"{commit_hash}.graphml.zip")
                
                if not os.path.exists(zip_path):
                    print(f"File not found: {zip_path}")
                    continue
                
                # Unzip the file
                xml_path = os.path.join(os.path.dirname(zip_path), f"{commit_hash}.xml")
                self.unzip_without_folders(zip_path, xml_path)
                
                try:
                    # Load and process GraphML
                    G = nx.read_graphml(xml_path)
                    
                    # Extract features for each node
                    for node in G.nodes(data=True):
                        node_id, attributes = node
                        state = attributes.get('state', 'unchanged')
                        node_type = attributes.get('labelV', 'default_type')
                        code = attributes.get('CODE', '')
                        
                        # Combine all CPG properties
                        other_features = ' '.join([str(attributes.get(prop, '')) 
                                                 for prop in self.config.cpg_properties])
                        
                        # Create feature string and split into tokens
                        node_feature = f"{state} {node_type} {code} {other_features}".split()
                        yield node_feature
                
                finally:
                    # Clean up temporary XML file
                    if os.path.exists(xml_path):
                        os.remove(xml_path)
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue
    
    def train_word2vec_model(self, file_paths: List[str], 
                            vector_size: Optional[int] = None,
                            window: Optional[int] = None,
                            min_count: Optional[int] = None,
                            workers: Optional[int] = None,
                            epochs: Optional[int] = None) -> Word2Vec:
        """
        Train Word2Vec model on node features.
        
        Args:
            file_paths: List of file paths to process
            vector_size: Vector size (uses config default if None)
            window: Window size (uses config default if None)
            min_count: Minimum count (uses config default if None)
            workers: Number of workers (uses config default if None)
            epochs: Number of epochs (uses config default if None)
            
        Returns:
            Trained Word2Vec model
        """
        # Use config defaults if not specified
        vector_size = vector_size or self.config.word2vec_vector_size
        window = window or self.config.word2vec_window
        min_count = min_count or self.config.word2vec_min_count
        workers = workers or self.config.word2vec_workers
        epochs = epochs or self.config.word2vec_epochs
        
        print(f"Training Word2Vec model with parameters:")
        print(f"  Vector size: {vector_size}")
        print(f"  Window: {window}")
        print(f"  Min count: {min_count}")
        print(f"  Workers: {workers}")
        print(f"  Epochs: {epochs}")
        
        # Initialize model
        model = Word2Vec(vector_size=vector_size, window=window, 
                        min_count=min_count, workers=workers)
        
        # Build vocabulary from generator
        print("Building vocabulary...")
        model.build_vocab(self.node_features_generator(file_paths))
        
        # Train the model incrementally
        print("Training model...")
        for epoch in range(epochs):
            print(f"  Epoch {epoch + 1}/{epochs}")
            model.train(self.node_features_generator(file_paths), 
                       total_examples=model.corpus_count, epochs=1)
        
        self.model = model
        print("Word2Vec model training completed!")
        return model
    
    def generate_embeddings(self, G: nx.Graph, 
                           fallback_dimensions: Optional[int] = None) -> Dict[Any, torch.Tensor]:
        """
        Generate Word2Vec embeddings for nodes in a graph.
        
        Args:
            G: NetworkX graph object
            fallback_dimensions: Dimensions for fallback embeddings (uses config default if None)
            
        Returns:
            Dictionary mapping node IDs to embedding tensors
        """
        if self.model is None:
            raise RuntimeError("Word2Vec model not trained. Call train_word2vec_model() first.")
        
        fallback_dimensions = fallback_dimensions or self.config.word2vec_vector_size
        
        # Extract node features
        node_features, node_feature_dict = extract_node_features(G, self.config)
        
        # Generate embeddings for each node
        node_embeddings = {}
        for node_id, feature_list in node_feature_dict.items():
            try:
                # Get feature vectors for words that exist in vocabulary
                feature_vectors = []
                for word in feature_list:
                    if word in self.model.wv:
                        feature_vectors.append(self.model.wv[word])
                
                if feature_vectors:
                    # Average the feature vectors
                    avg_embedding = sum(feature_vectors) / len(feature_vectors)
                    node_embeddings[node_id] = torch.tensor(avg_embedding, dtype=torch.float)
                else:
                    # Fallback: zero vector
                    node_embeddings[node_id] = torch.zeros(fallback_dimensions, dtype=torch.float)
                    print(f"Warning: No features found for node {node_id}, using zero vector")
                    
            except Exception as e:
                print(f"Error generating embedding for node {node_id}: {e}")
                # Fallback: zero vector
                node_embeddings[node_id] = torch.zeros(fallback_dimensions, dtype=torch.float)
        
        print(f"Generated Word2Vec embeddings for {len(node_embeddings)} nodes")
        return node_embeddings
    
    def save_model(self, file_path: str) -> None:
        """
        Save the trained Word2Vec model.
        
        Args:
            file_path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train a model first.")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.model.save(file_path)
        print(f"Word2Vec model saved to: {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """
        Load a trained Word2Vec model.
        
        Args:
            file_path: Path to the model file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        self.model = Word2Vec.load(file_path)
        print(f"Word2Vec model loaded from: {file_path}")
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the model vocabulary.
        
        Returns:
            Number of words in vocabulary
        """
        if self.model is None:
            return 0
        return len(self.model.wv)
    
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensions of the embeddings.
        
        Returns:
            Embedding dimensions
        """
        if self.model is None:
            return 0
        return self.model.vector_size
    
    def find_similar_words(self, word: str, topn: int = 10) -> List[tuple]:
        """
        Find words most similar to the given word.
        
        Args:
            word: Input word
            topn: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if self.model is None:
            raise RuntimeError("No model available. Train or load a model first.")
        
        if word not in self.model.wv:
            return []
        
        return self.model.wv.most_similar(word, topn=topn)
    
    def get_word_vector(self, word: str) -> Optional[List[float]]:
        """
        Get the vector for a specific word.
        
        Args:
            word: Input word
            
        Returns:
            Word vector or None if word not in vocabulary
        """
        if self.model is None:
            return None
        
        if word in self.model.wv:
            return self.model.wv[word].tolist()
        return None
    
    def update_model(self, additional_sentences: List[List[str]], 
                    epochs: int = 1) -> None:
        """
        Update the model with additional training data.
        
        Args:
            additional_sentences: List of additional sentences
            epochs: Number of training epochs
        """
        if self.model is None:
            raise RuntimeError("No model available. Train or load a model first.")
        
        print(f"Updating model with {len(additional_sentences)} additional sentences...")
        
        # Update vocabulary if needed
        self.model.build_vocab(additional_sentences, update=True)
        
        # Train on additional data
        self.model.train(additional_sentences, total_examples=len(additional_sentences), epochs=epochs)
        
        print("Model update completed!")
    
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
            'vocabulary_size': len(self.model.wv),
            'vector_size': self.model.vector_size,
            'window': self.model.window,
            'min_count': self.model.min_count,
            'workers': self.model.workers,
            'epochs': self.model.epochs
        }

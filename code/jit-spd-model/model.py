"""
Core JIT-SPD model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE
from torch.nn import LayerNorm, BatchNorm1d
from torch_geometric.nn.aggr import AttentionalAggregation
from typing import Dict, Any, Tuple, Optional

from .config import ModelConfig


class HomogeneousGraphConvolution(nn.Module):
    """
    Homogeneous graph convolution layer using GraphSAGE.
    
    Code Reference: new_effort/SimplifiedHomoGraphClassifierV2.py (lines 7-18)
    """
    
    def __init__(self, in_channels: int, out_channels: int, config: Optional[ModelConfig] = None):
        """
        Initialize the graph convolution layer.
        
        Args:
            in_channels: Input feature dimensions
            out_channels: Output feature dimensions
            config: Model configuration (uses default if None)
        """
        super().__init__()
        self.config = config or ModelConfig()
        
        # Normalization layers
        self.layernorm = LayerNorm(normalized_shape=out_channels)
        self.batchnorm = BatchNorm1d(out_channels)
        
        # Graph convolution
        self.conv = GraphSAGE(in_channels, out_channels, num_layers=1, act='gelu')
        
        # Activation function
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the graph convolution layer.
        
        Args:
            x: Node features tensor
            edge_index: Edge index tensor
            
        Returns:
            Transformed node features
        """
        # Apply graph convolution
        x = self.conv(x, edge_index)
        
        # Apply normalization
        x = self.layernorm(x)
        
        # Apply activation
        return self.gelu(x)


class JITSPDModel(nn.Module):
    """
    Main JIT-SPD model for software defect prediction.
    
    Code Reference: new_effort/SimplifiedHomoGraphClassifierV2.py (lines 20-88)
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the JIT-SPD model.
        
        Args:
            config: Model configuration (uses default if None)
        """
        super().__init__()
        self.config = config or ModelConfig()
        
        # Store configuration
        self.out_channels = self.config.out_channels
        self.with_manual_features = self.config.with_manual_features
        self.with_text_features = self.config.with_text_features
        self.with_graph_features = self.config.with_graph_features
        self.manual_size = self.config.manual_size
        
        # Graph convolution layer
        self.homo_conv = HomogeneousGraphConvolution(
            self.config.in_channels, 
            self.config.out_channels, 
            self.config
        )
        
        # Attention pooling layer for graph-level embeddings
        self.global_attention = AttentionalAggregation(
            gate_nn=nn.Linear(self.config.out_channels, 1)
        )
        
        # Input normalization
        self.project_norm = LayerNorm(normalized_shape=self.config.in_channels)
        self.batchnorm = BatchNorm1d(self.config.in_channels)
        
        # Regularization
        self.dropout = nn.Dropout(self.config.dropout_rate)
        
        # Feature processing layers
        self.msg_fc = nn.Linear(768, self.config.out_channels)  # Text features
        self.features_fc = nn.Linear(self.manual_size, self.config.out_channels)  # Manual features
        
        # Learnable weights
        self.graph_weight = nn.Parameter(torch.tensor(1.0))
        
        # Count components for fusion
        self.num_components = 0
        if self.with_graph_features:
            self.num_components += 1
        if self.with_manual_features:
            self.num_components += 1
        if self.with_text_features:
            self.num_components += 1
        
        # Multi-modal fusion
        if self.num_components > 1:
            self.mixed_norm = LayerNorm(normalized_shape=self.num_components * self.config.out_channels)
        
        # Final classification layer
        self.fc1 = nn.Linear(self.num_components * self.config.out_channels, 1)
    
    def graph_embedding_gen(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate graph embeddings using attention pooling.
        
        Args:
            data: Input data dictionary containing graph information
            
        Returns:
            Tuple of (graph_embedding, attention_weights)
        """
        # Extract graph data
        x = data['x_dict']
        edge_index = data['edge_index']
        batch = data['batch']
        batch_size = data['batch_size']
        
        # Normalize input features
        x = self.project_norm(x)
        
        # Apply graph convolution
        x = self.homo_conv(x, edge_index)
        
        # Global attention pooling
        graph_embedding, attn_weights = self.global_attention(x, index=batch, dim_size=batch_size)
        
        return graph_embedding, attn_weights
    
    def forward(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            data: Input data dictionary containing:
                - x_dict: Node features
                - edge_index: Edge indices
                - batch: Batch indices
                - text_embedding: Text embeddings
                - features_embedding: Manual feature embeddings
                - batch_size: Batch size
                
        Returns:
            Tuple of (logits, graph_embedding, graph_attention_weights)
        """
        embedding_features = []
        
        # Process graph features
        if self.with_graph_features:
            graph_embedding, graph_attn_weights = self.graph_embedding_gen(data)
            graph_embedding = self.dropout(graph_embedding)
            weighted_graph_embedding = self.graph_weight * graph_embedding
            embedding_features.append(weighted_graph_embedding)
        else:
            graph_attn_weights = []
            graph_embedding = torch.empty((0,), dtype=torch.long)
        
        # Process text features
        if self.with_text_features:
            commit_msg_data = data['text_embedding']
            commit_msg_embeddings = F.gelu(self.msg_fc(commit_msg_data))
            embedding_features.append(commit_msg_embeddings)
        
        # Process manual features
        if self.with_manual_features:
            features_embedding = data['features_embedding']
            features_embedding = F.gelu(self.features_fc(features_embedding))
            embedding_features.append(features_embedding)
        
        # Concatenate all features
        embeddings = torch.cat(embedding_features, dim=1)
        
        # Apply normalization if multiple components
        if self.num_components > 1:
            embeddings = self.mixed_norm(embeddings)
        
        # Final classification
        logits = self.fc1(embeddings)
        
        return logits, graph_embedding, graph_attn_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model architecture.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': 'JITSPDModel',
            'in_channels': self.config.in_channels,
            'out_channels': self.config.out_channels,
            'manual_size': self.manual_size,
            'num_components': self.num_components,
            'with_graph_features': self.with_graph_features,
            'with_text_features': self.with_text_features,
            'with_manual_features': self.with_manual_features,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    def save_model(self, file_path: str) -> None:
        """
        Save the model to file.
        
        Args:
            file_path: Path to save the model
        """
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'model_info': self.get_model_info()
        }, file_path)
        
        print(f"Model saved to: {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """
        Load the model from file.
        
        Args:
            file_path: Path to the model file
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        checkpoint = torch.load(file_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from: {file_path}")
    
    def get_feature_importance(self, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Get feature importance scores for interpretability.
        
        Args:
            data: Input data dictionary
            
        Returns:
            Dictionary containing feature importance scores
        """
        importance_scores = {}
        
        # Graph feature importance
        if self.with_graph_features:
            graph_embedding, _ = self.graph_embedding_gen(data)
            importance_scores['graph'] = torch.norm(graph_embedding, dim=1)
        
        # Text feature importance
        if self.with_text_features:
            text_embedding = data['text_embedding']
            importance_scores['text'] = torch.norm(text_embedding, dim=1)
        
        # Manual feature importance
        if self.with_manual_features:
            features_embedding = data['features_embedding']
            importance_scores['manual'] = torch.norm(features_embedding, dim=1)
        
        return importance_scores


# Legacy class name for backward compatibility
class HomogeneousGraphSequentialClassifierV2(JITSPDModel):
    """
    Legacy class name for backward compatibility.
    This class inherits from JITSPDModel and provides the same functionality.
    """
    
    def __init__(self, in_channels: int, out_channels: int, device: str = 'cpu',
                 with_manual_features: bool = True, with_text_features: bool = True,
                 with_graph_features: bool = True, manual_size: int = 14):
        """
        Initialize with legacy parameters.
        
        Args:
            in_channels: Input feature dimensions
            out_channels: Output feature dimensions
            device: Device to use (for backward compatibility)
            with_manual_features: Whether to use manual features
            with_text_features: Whether to use text features
            with_graph_features: Whether to use graph features
            manual_size: Size of manual features
        """
        # Create config from legacy parameters
        config = ModelConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            with_manual_features=with_manual_features,
            with_text_features=with_text_features,
            with_graph_features=with_graph_features,
            manual_size=manual_size
        )
        
        super().__init__(config)
        
        # Store device for backward compatibility
        self.device = device

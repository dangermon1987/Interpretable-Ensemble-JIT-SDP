"""
Data processing for JIT-SPD model.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch_geometric.data import Dataset, HeteroData, Data
import torch_geometric.transforms as T
from sklearn.utils import resample
from typing import List, Dict, Any, Optional, Tuple

from .config import ModelConfig, TrainingConfig


class DataProcessor:
    """
    Processes and prepares data for the JIT-SPD model.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Model configuration (uses default if None)
        """
        self.config = config or ModelConfig()
    
    def get_edge_type(self, edge_type: str) -> str:
        """
        Map edge types to unified categories.
        
        Args:
            edge_type: Original edge type
            
        Returns:
            Unified edge type category
        """
        return self.config.edge_type_mapping.get(edge_type, "OTHER")
    
    def to_homogeneous(self, graph_data: HeteroData) -> Data:
        """
        Convert heterogeneous graph to homogeneous format.
        
        Args:
            graph_data: Heterogeneous graph data
            
        Returns:
            Homogeneous graph data
        """
        if graph_data.num_nodes == 0 or graph_data.num_edges == 0 or len(graph_data.edge_types) == 0:
            return self._create_empty_data()
        
        homogeneous_data = graph_data.to_homogeneous(
            node_attrs=['x', 'id'],
            add_node_type=True,
            add_edge_type=True,
            dummy_values=True
        )
        return homogeneous_data
    
    def _create_empty_data(self) -> Data:
        """Create empty data structure."""
        data = Data()
        data.x = torch.empty((0, self.config.node_feature_dimension))
        data.edge_index = torch.empty((2, 0), dtype=torch.long)
        data.node_type = torch.empty((0,), dtype=torch.long)
        data.edge_type = torch.empty((0,), dtype=torch.long)
        data.id = torch.empty((0,), dtype=torch.long)
        return data
    
    def filter_graph_by_edges(self, graph: HeteroData, edge_types: List[str]) -> HeteroData:
        """
        Filter graph to include only specified edge types.
        
        Args:
            graph: Input graph
            edge_types: List of edge types to include
            
        Returns:
            Filtered graph
        """
        if edge_types is None:
            return graph
        
        # Map edge types to unified categories
        unified_edge_types = [self.get_edge_type(et) for et in edge_types]
        
        # Filter edges
        filtered_edge_types = []
        for edge_type in graph.edge_types:
            if self.get_edge_type(edge_type[1]) in unified_edge_types:
                filtered_edge_types.append(edge_type)
        
        # Create subgraph
        if filtered_edge_types:
            return graph.edge_type_subgraph(edge_types=filtered_edge_types)
        else:
            return self._create_empty_data()
    
    def transform_sample(self, sample: Tuple) -> Tuple:
        """
        Transform a data sample.
        
        Args:
            sample: Input sample tuple
            
        Returns:
            Transformed sample tuple
        """
        graph, commit_message, code, feature_embedding, label, commit = sample
        
        # Normalize features
        graph = T.NormalizeFeatures()(graph)
        
        # Convert to homogeneous
        graph = self.to_homogeneous(graph)
        
        return graph, commit_message, code, feature_embedding, label, commit
    
    def convert_dtype_dataframe(self, df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
        """
        Convert dataframe column data types.
        
        Args:
            df: Input dataframe
            feature_names: List of feature column names
            
        Returns:
            Dataframe with converted data types
        """
        # Convert fix column to boolean then float
        if 'fix' in df.columns:
            df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
        
        # Convert feature columns to float32
        for feature in feature_names:
            if feature in df.columns:
                df[feature] = df[feature].astype('float32')
        
        return df


class JITSPDDataset(Dataset):
    """
    Dataset class for JIT-SPD model training and evaluation.
    
    Code Reference: new_effort/NewLazyLoadDatasetV4.py (lines 100-396)
    """
    
    def __init__(self, data_dir: str, projects: Optional[List[str]] = None, 
                 data_type: str = 'train', merge: bool = True, old: bool = True,
                 tokenizer=None, model=None, device: str = 'cpu', 
                 changes_data=None, features_data=None, need_attentions: bool = True,
                 under_sampling: bool = False, scaler=None, config: Optional[ModelConfig] = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing data files
            projects: List of project names to include
            data_type: Type of dataset ('train', 'validation', 'test')
            merge: Whether to merge data
            old: Whether to use old data format
            tokenizer: BERT tokenizer
            model: BERT model
            device: Device to use
            changes_data: Changes data
            features_data: Features data
            need_attentions: Whether attention weights are needed
            under_sampling: Whether to apply undersampling
            scaler: Feature scaler
            config: Model configuration
        """
        super().__init__()
        self.config = config or ModelConfig()
        self.data_processor = DataProcessor(self.config)
        
        # Store parameters
        self.data_dir = data_dir
        self.projects = projects
        self.data_type = data_type
        self.merge = merge
        self.old = old
        self.bert_tokenizer = tokenizer
        self.bert_model = model
        self.device = device
        self.changes_data = changes_data
        self.features_data = features_data
        self.need_attentions = need_attentions
        self.under_sampling = under_sampling
        self.scaler = scaler
        
        # Initialize data structures
        self.labels = None
        self.cached_data = {}
        self.path_suffix = '_hetero_data.pt'
        
        # Configuration
        self.graph_edges = None
        self.manual_features_columns = None
        
        # Load data
        self._load_data()
        
        # Apply undersampling if requested
        if self.under_sampling:
            self._apply_undersampling()
    
    def _load_data(self) -> None:
        """Load and prepare data."""
        # Load labels
        self._load_labels()
        
        # Verify data integrity
        self.verify_data()
        
        # Set default configuration
        if self.manual_features_columns is None:
            self.manual_features_columns = self.config.get_all_manual_features()
        
        if self.graph_edges is None:
            self.graph_edges = self.config.graph_edges.copy()
    
    def _load_labels(self) -> None:
        """Load labels and metadata."""
        # Load buggy line data
        buggy_line_filepath = os.path.join(self.data_dir, 'changes_complete_buggy_line_level.pkl')
        
        if os.path.exists(buggy_line_filepath):
            self.labels = pd.read_pickle(buggy_line_filepath)
        else:
            # Create empty labels if file doesn't exist
            self.labels = pd.DataFrame(columns=['is_buggy_commit', 'commit_message', 'code'])
        
        # Filter by projects if specified
        if self.projects:
            project_filter = self.labels.index.str.split('_').str[0].isin(self.projects)
            self.labels = self.labels[project_filter]
    
    def _apply_undersampling(self) -> None:
        """Apply undersampling to balance classes."""
        if len(self.labels) == 0:
            return
        
        # Get positive and negative samples
        positive_samples = self.labels[self.labels['is_buggy_commit'] == 1]
        negative_samples = self.labels[self.labels['is_buggy_commit'] == 0]
        
        # Determine target size (use smaller class size)
        target_size = min(len(positive_samples), len(negative_samples))
        
        # Resample both classes
        if len(positive_samples) > target_size:
            positive_samples = resample(positive_samples, n_samples=target_size, random_state=42)
        
        if len(negative_samples) > target_size:
            negative_samples = resample(negative_samples, n_samples=target_size, random_state=42)
        
        # Combine samples
        self.labels = pd.concat([positive_samples, negative_samples])
        self.labels = self.labels.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def verify_data(self) -> None:
        """Verify data integrity."""
        print(f"Verifying {self.data_type} data...")
        
        valid_commits = []
        for commit_hash in self.labels.index:
            try:
                project_name = commit_hash.split('_')[0]
                commit_id = commit_hash.split('_')[1]
                graph_path = os.path.join(self.data_dir, project_name, commit_id + self.path_suffix)
                
                if os.path.exists(graph_path):
                    valid_commits.append(commit_hash)
                else:
                    print(f"Dropping missing graph: {graph_path}")
                    
            except Exception as e:
                print(f"Error processing commit {commit_hash}: {e}")
        
        # Update labels with valid commits
        self.labels = self.labels.loc[valid_commits]
        print(f"Valid commits: {len(self.labels)}")
    
    def set_config(self, manual_features_columns: List[str], graph_edges: List[str]) -> None:
        """
        Set dataset configuration.
        
        Args:
            manual_features_columns: List of manual feature column names
            graph_edges: List of graph edge types to include
        """
        self.graph_edges = graph_edges
        self.manual_features_columns = manual_features_columns
        
        # Update labels to include only required features
        features = ['is_buggy_commit', 'commit_message', 'code']
        features += self.manual_features_columns
        
        available_features = [f for f in features if f in self.labels.columns]
        self.labels = self.labels[available_features]
    
    def prepare_data(self, data: Tuple) -> Tuple:
        """
        Prepare data for processing.
        
        Args:
            data: Input data tuple
            
        Returns:
            Prepared data tuple
        """
        if self.manual_features_columns is None or self.graph_edges is None:
            print("Warning: manual_features_columns and graph_edges not set")
            return data
        
        graph, commit_message, code, feature_embedding, label, commit_hash = data
        
        # Filter graph by edge types
        if self.graph_edges is not None:
            graph = self.data_processor.filter_graph_by_edges(graph, self.graph_edges)
        
        # Create sample
        sample = (graph, commit_message, code, feature_embedding, label, commit_hash)
        
        # Transform sample
        sample = self.data_processor.transform_sample(sample)
        
        return sample
    
    def get_item_from_commit_hash(self, commit_hash: str) -> Tuple:
        """
        Get data item for a specific commit hash.
        
        Args:
            commit_hash: Commit hash identifier
            
        Returns:
            Data tuple
        """
        # Get basic information
        label = self.labels.at[commit_hash, 'is_buggy_commit']
        commit_message = str(self.labels.at[commit_hash, 'commit_message'])
        code = str(self.labels.at[commit_hash, 'code'])
        
        # Get feature columns
        features_columns = (self.manual_features_columns if self.manual_features_columns 
                          else self.config.get_all_manual_features())
        
        # Check cache
        if commit_hash in self.cached_data:
            graph, feature_embedding = self.cached_data[commit_hash]
        else:
            # Load graph data
            graph = self._load_graph_data(commit_hash)
            
            # Get feature embedding
            feature_embedding = torch.tensor(
                self.labels.loc[commit_hash][features_columns].to_numpy(dtype=np.float32)
            ).to(self.device)
            
            # Prepare data
            sample = (graph, commit_message, code, feature_embedding, label, commit_hash)
            sample = self.prepare_data(sample)
            graph, commit_message, code, feature_embedding, label, commit_hash = sample
            
            # Cache result
            self.cached_data[commit_hash] = (graph, feature_embedding)
        
        return (graph, commit_message, code, feature_embedding, label, commit_hash)
    
    def _load_graph_data(self, commit_hash: str) -> HeteroData:
        """
        Load graph data for a commit.
        
        Args:
            commit_hash: Commit hash identifier
            
        Returns:
            Graph data
        """
        try:
            project_name = commit_hash.split('_')[0]
            commit_id = commit_hash.split('_')[1]
            
            if len(commit_hash.split('_')) == 3:
                commit_id = commit_hash.split('_')[1] + '_' + commit_hash.split('_')[2]
            
            graph_path = os.path.join(self.data_dir, project_name, commit_id + self.path_suffix)
            
            if os.path.exists(graph_path):
                graph = torch.load(graph_path, weights_only=False)
                return graph
            else:
                print(f"Graph file not found: {graph_path}")
                return self.data_processor._create_empty_data()
                
        except Exception as e:
            print(f"Error loading graph for {commit_hash}: {e}")
            return self.data_processor._create_empty_data()
    
    def get_sample_weights(self) -> torch.Tensor:
        """
        Get sample weights for balanced training.
        
        Returns:
            Tensor of sample weights
        """
        if len(self.labels) == 0:
            return torch.tensor([])
        
        pos_count = self.labels['is_buggy_commit'].sum()
        neg_count = len(self.labels) - pos_count
        
        if pos_count == 0 or neg_count == 0:
            return torch.ones(len(self.labels), dtype=torch.float32)
        
        total = pos_count + neg_count
        pos_weight = pos_count / total
        neg_weight = neg_count / total
        
        sample_weights = self.labels['is_buggy_commit'].apply(
            lambda x: pos_weight if x == 1 else neg_weight
        )
        
        return torch.tensor(np.array(sample_weights), dtype=torch.float32)
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.labels)
    
    def __getitem__(self, idx) -> Any:
        """
        Get dataset item.
        
        Args:
            idx: Index or commit hash
            
        Returns:
            Data item
        """
        if isinstance(idx, list):
            return [self.get_item_from_commit_hash(i) for i in idx]
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if isinstance(idx, str):
            return self.get_item_from_commit_hash(idx)
        else:
            commit_hash = self.labels.index[idx]
            return self.get_item_from_commit_hash(commit_hash)
    
    def get_labels(self) -> List[int]:
        """Get all labels."""
        return self.labels['is_buggy_commit'].tolist()
    
    def get_projects(self) -> List[str]:
        """Get list of projects in dataset."""
        try:
            return [commit_hash.split('_')[0] for commit_hash in self.labels.index]
        except Exception as e:
            print(f"Error getting projects: {e}")
            return []
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'data_type': self.data_type,
            'total_samples': len(self.labels),
            'projects': self.get_projects(),
            'num_projects': len(set(self.get_projects())),
            'positive_samples': self.labels['is_buggy_commit'].sum(),
            'negative_samples': len(self.labels) - self.labels['is_buggy_commit'].sum(),
            'manual_features': len(self.manual_features_columns) if self.manual_features_columns else 0,
            'graph_edges': self.graph_edges,
            'data_dir': self.data_dir
        }


# Legacy class name for backward compatibility
class NewLazyLoadDatasetV4(JITSPDDataset):
    """
    Legacy class name for backward compatibility.
    This class inherits from JITSPDDataset and provides the same functionality.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with legacy parameters."""
        super().__init__(*args, **kwargs)

"""
Feature processing for JIT-SPD model.
"""

import os
import torch
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel
from torch_geometric.loader import DataLoader

from .config import TrainingConfig


class TextEmbeddingGenerator:
    """
    Generates text embeddings for commit messages and code changes using CodeBERT.
    
    Code Reference: new_effort/prepare_text_embeddings.py (lines 1-138)
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the text embedding generator.
        
        Args:
            config: Training configuration (uses default if None)
        """
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize CodeBERT
        self._initialize_codebert()
    
    def _initialize_codebert(self) -> None:
        """Initialize CodeBERT model and tokenizer."""
        try:
            print(f"Loading CodeBERT model from: {self.config.codebert_model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.codebert_model_path)
            self.model = AutoModel.from_pretrained(self.config.codebert_model_path, output_attentions=True)
            
            # Add special tokens
            special_tokens_dict = {
                'additional_special_tokens': [self.config.add_token, self.config.del_token]
            }
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.model.resize_token_embeddings(len(self.tokenizer))
            
            # Move to device
            self.model.to(self.device)
            
            print("CodeBERT model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading CodeBERT model: {e}")
            raise
    
    def generate_text_embeddings(self, commit_msgs: List[str], codes: List[str], 
                                full: bool = False) -> Any:
        """
        Generate text embeddings for commit messages and code changes.
        
        Args:
            commit_msgs: List of commit messages
            codes: List of code change dictionaries
            full: Whether to return full embedding information
            
        Returns:
            Text embeddings (format depends on full parameter)
        """
        # Prepare lists for all outputs
        all_cls_embeddings = []
        all_attn_weights = []
        all_tokens = []
        
        # Create lists to store added and removed tokens
        added_tokens_batch = []
        removed_tokens_batch = []
        commit_msgs_tokens_batch = []
        
        # Tokenize commit messages and prepare code tokens
        for commit_msg, code in zip(commit_msgs, codes):
            # Tokenize commit message
            msg_tokens = self.tokenizer.tokenize(commit_msg)
            msg_tokens = msg_tokens[:min(self.config.max_commit_message_tokens, len(msg_tokens))]
            commit_msgs_tokens_batch.append(msg_tokens)
            
            # Parse code changes
            try:
                code_dict = eval(code) if isinstance(code, str) else code
            except:
                code_dict = {'added_code': [], 'removed_code': []}
            
            # Process added and removed codes
            added_codes = [' '.join(line.split()) for line in code_dict.get('added_code', [])]
            removed_codes = [' '.join(line.split()) for line in code_dict.get('removed_code', [])]
            
            codes_added = self.config.add_token.join([line for line in added_codes if line])
            added_tokens_batch.append(self.tokenizer.tokenize(codes_added))
            
            codes_removed = self.config.del_token.join([line for line in removed_codes if line])
            removed_tokens_batch.append(self.tokenizer.tokenize(codes_removed))
        
        # Prepare inputs for batch encoding
        input_ids = []
        attention_masks = []
        
        for msg_tokens, added_tokens, removed_tokens in zip(commit_msgs_tokens_batch, 
                                                           added_tokens_batch, removed_tokens_batch):
            input_tokens = msg_tokens + [self.config.add_token] + added_tokens + [self.config.del_token] + removed_tokens
            input_tokens = input_tokens[:self.config.max_total_tokens - 2]
            
            tokens = [self.tokenizer.cls_token] + input_tokens + [self.tokenizer.sep_token]
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            attention_mask = [1] * len(tokens_ids)
            
            input_ids.append(tokens_ids)
            attention_masks.append(attention_mask)
            all_tokens.append(tokens)
        
        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True)
        attention_masks = torch.nn.utils.rnn.pad_sequence([torch.tensor(mask) for mask in attention_masks], batch_first=True)
        
        # Move to device
        input_ids = input_ids.to(self.device)
        attention_masks = attention_masks.to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)
        
        # Extract embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
        last_layer_attentions = outputs.attentions[-1].cpu()
        
        if full:
            return self._process_full_embeddings(all_tokens, cls_embeddings, last_layer_attentions)
        else:
            return cls_embeddings
    
    def _process_full_embeddings(self, all_tokens: List[List[str]], 
                                cls_embeddings: torch.Tensor, 
                                last_layer_attentions: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Process full embedding information including attention weights.
        
        Args:
            all_tokens: List of token lists for each input
            cls_embeddings: CLS token embeddings
            last_layer_attentions: Attention weights from last layer
            
        Returns:
            List of embedding dictionaries
        """
        batch_attns = []
        
        for i, tokens in enumerate(all_tokens):
            try:
                begin_pos = tokens.index(self.config.add_token)
                end_pos = tokens.index(self.config.del_token) if self.config.del_token in tokens else len(tokens) - 1
                attns = last_layer_attentions[i]
                attns = attns.mean(axis=0)[1][begin_pos:end_pos]
                batch_attns.append(attns)
            except ValueError:
                # Handle case where special tokens are not found
                attns = last_layer_attentions[i].mean(axis=0)[1]
                batch_attns.append(attns)
        
        return [{
            'cls_embeddings': cls_embeddings[i],
            'attn_weights': attn,
            'tokens': token
        } for i, (attn, token) in enumerate(zip(batch_attns, all_tokens))]
    
    def generate_single_embedding(self, commit_msg: str, code: str) -> Dict[str, Any]:
        """
        Generate embeddings for a single commit.
        
        Args:
            commit_msg: Commit message
            code: Code change dictionary
            
        Returns:
            Dictionary containing embedding information
        """
        return self.generate_text_embeddings([commit_msg], [code], full=True)[0]


class FeatureProcessor:
    """
    Processes and manages different types of features for the JIT-SPD model.
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize the feature processor.
        
        Args:
            config: Training configuration (uses default if None)
        """
        self.config = config or TrainingConfig()
        self.text_generator = TextEmbeddingGenerator(self.config)
    
    def prepare_text_embeddings(self, train_dataset, valid_dataset, test_dataset) -> Tuple[Dict, Dict, Dict]:
        """
        Prepare text embeddings for all datasets.
        
        Args:
            train_dataset: Training dataset
            valid_dataset: Validation dataset
            test_dataset: Test dataset
            
        Returns:
            Tuple of (train_embeddings, valid_embeddings, test_embeddings)
        """
        # Train embeddings
        train_text_embeddings = self._load_or_generate_embeddings(
            "train_text_embeddings_v4.pkl", train_dataset, "train"
        )
        
        # Validation embeddings
        valid_text_embeddings = self._load_or_generate_embeddings(
            "valid_text_embeddings_v4.pkl", valid_dataset, "validation"
        )
        
        # Test embeddings
        test_text_embeddings = self._load_or_generate_embeddings(
            "test_text_embeddings_v4.pkl", test_dataset, "test"
        )
        
        return train_text_embeddings, valid_text_embeddings, test_text_embeddings
    
    def _load_or_generate_embeddings(self, cache_file: str, dataset, dataset_type: str) -> Dict:
        """
        Load cached embeddings or generate new ones.
        
        Args:
            cache_file: Cache file name
            dataset: Dataset to process
            dataset_type: Type of dataset for logging
            
        Returns:
            Dictionary of embeddings
        """
        if os.path.exists(cache_file):
            print(f"Loading cached {dataset_type} embeddings from {cache_file}")
            return torch.load(cache_file, weights_only=False)
        
        print(f"Generating {dataset_type} embeddings...")
        embeddings = {}
        
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        for batch in data_loader:
            homogeneous_data, commit_message, code, features_embedding, labels, commits = batch
            
            text_embeddings = self.text_generator.generate_text_embeddings(
                commit_message, code, full=True
            )
            
            for i, commit_id in enumerate(commits):
                embeddings[commit_id] = text_embeddings[i]
        
        # Cache embeddings
        torch.save(embeddings, cache_file)
        print(f"Saved {dataset_type} embeddings to {cache_file}")
        
        return embeddings
    
    def get_feature_statistics(self, embeddings: Dict) -> Dict[str, Any]:
        """
        Get statistics about the generated embeddings.
        
        Args:
            embeddings: Dictionary of embeddings
            
        Returns:
            Dictionary containing statistics
        """
        if not embeddings:
            return {'total_embeddings': 0}
        
        total_embeddings = len(embeddings)
        embedding_dimensions = []
        attention_weights = []
        
        for commit_id, embedding_data in embeddings.items():
            if 'cls_embeddings' in embedding_data:
                embedding_dimensions.append(embedding_data['cls_embeddings'].shape[0])
            
            if 'attn_weights' in embedding_data:
                attention_weights.append(embedding_data['attn_weights'].shape)
        
        return {
            'total_embeddings': total_embeddings,
            'embedding_dimensions': embedding_dimensions,
            'attention_weight_shapes': attention_weights,
            'avg_embedding_dim': sum(embedding_dimensions) / len(embedding_dimensions) if embedding_dimensions else 0
        }
    
    def validate_embeddings(self, embeddings: Dict) -> Dict[str, Any]:
        """
        Validate the structure and content of embeddings.
        
        Args:
            embeddings: Dictionary of embeddings
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'total_embeddings': len(embeddings),
            'valid_embeddings': 0
        }
        
        for commit_id, embedding_data in embeddings.items():
            try:
                # Check required keys
                required_keys = ['cls_embeddings', 'attn_weights', 'tokens']
                missing_keys = [key for key in required_keys if key not in embedding_data]
                
                if missing_keys:
                    validation_results['errors'].append(f"Commit {commit_id}: missing keys {missing_keys}")
                    validation_results['is_valid'] = False
                    continue
                
                # Check embedding dimensions
                cls_emb = embedding_data['cls_embeddings']
                if not isinstance(cls_emb, torch.Tensor) or cls_emb.dim() != 1:
                    validation_results['errors'].append(f"Commit {commit_id}: invalid CLS embedding format")
                    validation_results['is_valid'] = False
                    continue
                
                # Check attention weights
                attn_weights = embedding_data['attn_weights']
                if not isinstance(attn_weights, torch.Tensor) or attn_weights.dim() != 1:
                    validation_results['warnings'].append(f"Commit {commit_id}: unexpected attention weights format")
                
                validation_results['valid_embeddings'] += 1
                
            except Exception as e:
                validation_results['errors'].append(f"Commit {commit_id}: error during validation - {e}")
                validation_results['is_valid'] = False
        
        return validation_results

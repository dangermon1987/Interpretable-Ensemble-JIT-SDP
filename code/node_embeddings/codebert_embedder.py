"""
CodeBERT-based semantic embedding generation for CPG nodes.
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel

from .config import EmbeddingConfig
from .utils import add_state_vector


class CodeBERTEmbedder:
    """
    CodeBERT-based semantic embedding generator for CPG nodes.
    
    Code Reference: new_embedding/GenGraphData.py (lines 76-140)
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the CodeBERT embedder.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or EmbeddingConfig()
        self.tokenizer = None
        self.model = None
        self.device = torch.device(self.config.device)
        
        # Initialize model and tokenizer
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize CodeBERT model and tokenizer."""
        try:
            model_path = self.config.get_codebert_model_path()
            print(f"Loading CodeBERT model from: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModel.from_pretrained(model_path)
            self.model.to(self.device)
            
            print("CodeBERT model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading CodeBERT model: {e}")
            print("Please ensure the model path is correct and the model is accessible.")
            raise
    
    def get_embeddings(self, state: str, node_type: str, node_text: str, 
                      node_code: str) -> torch.Tensor:
        """
        Generate embeddings for a single node.
        
        Args:
            state: Node state ('before', 'unchanged', 'after')
            node_type: Node type
            node_text: Node text representation
            node_code: Node code content
            
        Returns:
            Embedding tensor
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("CodeBERT model not initialized")
        
        # Tokenize input
        state_tokens = self.tokenizer.tokenize(state)
        node_type_tokens = self.tokenizer.tokenize(node_type)
        code_tokens = self.tokenizer.tokenize(node_code)
        text_tokens = self.tokenizer.tokenize(node_text)
        
        # Construct token sequence
        tokens = [self.tokenizer.sep_token] + state_tokens + [self.tokenizer.sep_token] + \
                node_type_tokens + [self.tokenizer.sep_token] + code_tokens + \
                [self.tokenizer.cls_token] + text_tokens + [self.tokenizer.eos_token]
        
        # Truncate to fit model input size
        max_tokens = self.config.codebert_max_tokens - 2
        while len(tokens) > max_tokens:
            # Reduce the size by truncating the longest part: code_tokens or text_tokens
            if len(code_tokens) > len(text_tokens):
                code_tokens.pop()
            else:
                text_tokens.pop()
            
            tokens = [self.tokenizer.cls_token] + state_tokens + [self.tokenizer.sep_token] + \
                    node_type_tokens + [self.tokenizer.sep_token] + code_tokens + \
                    [self.tokenizer.sep_token] + text_tokens + [self.tokenizer.eos_token]
        
        # Convert tokens to IDs
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Generate embeddings
        with torch.no_grad():
            inputs = torch.tensor(tokens_ids)[None, :].to(self.device)
            outputs = self.model(inputs)
        
        # Extract CLS token representation
        result = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return result
    
    def get_batch_embeddings(self, states: List[str], node_types: List[str], 
                            node_texts: List[str], node_codes: List[str]) -> torch.Tensor:
        """
        Generate embeddings for a batch of nodes.
        
        Args:
            states: List of node states
            node_types: List of node types
            node_texts: List of node text representations
            node_codes: List of node code contents
            
        Returns:
            Batch of embedding tensors
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("CodeBERT model not initialized")
        
        batch_tokens_ids = []
        attention_masks = []
        max_length = self.config.codebert_max_tokens - 2
        
        # Process each node in the batch
        for state, node_type, node_text, node_code in zip(states, node_types, node_texts, node_codes):
            # Construct tokens
            tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(state) + [self.tokenizer.sep_token] + \
                     self.tokenizer.tokenize(node_type) + [self.tokenizer.sep_token] + self.tokenizer.tokenize(node_code) + \
                     [self.tokenizer.sep_token] + self.tokenizer.tokenize(node_text) + [self.tokenizer.eos_token]
            
            # Truncate tokens to fit within the model's maximum input length
            tokens = tokens[:max_length] + [self.tokenizer.eos_token] if len(tokens) > max_length else tokens
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            
            # Create attention mask
            attention_mask = [1] * len(token_ids)
            
            # Add to batch
            batch_tokens_ids.append(torch.tensor(token_ids))
            attention_masks.append(torch.tensor(attention_mask))
        
        # Pad sequences
        padded_tokens_ids = pad_sequence(batch_tokens_ids, batch_first=True, 
                                       padding_value=self.tokenizer.pad_token_id)
        padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        
        # Move tensors to device
        padded_tokens_ids = padded_tokens_ids.to(self.device)
        padded_attention_masks = padded_attention_masks.to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(input_ids=padded_tokens_ids, attention_mask=padded_attention_masks)
        
        # Extract CLS token representations
        return outputs.last_hidden_state[:, 0, :]
    
    def process_graph_nodes(self, G: nx.Graph, batch_size: Optional[int] = None) -> Tuple[Dict[Any, Tuple[str, torch.Tensor]], Dict[Any, int]]:
        """
        Process all nodes in a graph to generate embeddings.
        
        Args:
            G: NetworkX graph object
            batch_size: Batch size for processing (uses config default if None)
            
        Returns:
            Tuple of (node embeddings, node indices)
        """
        if batch_size is None:
            batch_size = self.config.codebert_batch_size
        
        # Prepare data for batch processing
        states, node_types, node_texts, node_codes, node_ids = [], [], [], [], []
        
        for node_id, node_data in G.nodes(data=True):
            states.append(node_data.get('state', 'unchanged'))
            node_types.append(node_data.get('labelV', 'default_type'))
            node_codes.append(node_data.get('CODE', ''))
            
            # Create text representation from node attributes
            node_data_clone = node_data.copy()
            # Remove keys that are handled separately
            for key in ['CODE', 'state', 'identifier']:
                if key in node_data_clone:
                    del node_data_clone[key]
            
            node_text = f"/** {node_data_clone} */"  # Simplified text representation
            node_texts.append(node_text)
            node_ids.append(node_id)
        
        # Process nodes in batches
        num_nodes = len(states)
        node_indices = {}  # Will map node_id to its index within its node type
        node_embeddings = {}
        type_specific_counters = {}  # To keep track of indices within each node type
        
        print(f"Processing {num_nodes} nodes in batches of {batch_size}")
        
        for i in range(0, num_nodes, batch_size):
            if (i / batch_size) % 5 == 0:
                print(f"Processing batch {i // batch_size + 1}/{(num_nodes + batch_size - 1) // batch_size}")
            
            # Get batch data
            batch_states = states[i:i + batch_size]
            batch_node_types = node_types[i:i + batch_size]
            batch_node_texts = node_texts[i:i + batch_size]
            batch_node_codes = node_codes[i:i + batch_size]
            
            # Generate embeddings for batch
            embeddings = self.get_batch_embeddings(
                batch_states, batch_node_types, batch_node_texts, batch_node_codes
            )
            
            # Process each node in the batch
            for j, node_id in enumerate(node_ids[i:i + batch_size]):
                embedding = embeddings[j].to('cpu')
                node_type = batch_node_types[j]
                state = batch_states[j]
                
                # Add state vector to embedding
                embedding = add_state_vector(embedding, state, self.config)
                
                # Track type-specific indices
                if node_type not in type_specific_counters:
                    type_specific_counters[node_type] = 0
                
                node_embeddings[node_id] = (node_type, embedding)
                node_indices[node_id] = type_specific_counters[node_type]
                type_specific_counters[node_type] += 1
        
        print(f"Generated CodeBERT embeddings for {len(node_embeddings)} nodes")
        return node_embeddings, node_indices
    
    def get_embedding_dimensions(self) -> int:
        """
        Get the dimensions of the embeddings.
        
        Returns:
            Embedding dimensions
        """
        if self.model is None:
            return 0
        
        # Return hidden size + 1 for state vector
        return self.model.config.hidden_size + 1
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the CodeBERT model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {'status': 'No model available'}
        
        return {
            'status': 'Model loaded',
            'model_type': self.model.__class__.__name__,
            'hidden_size': self.model.config.hidden_size,
            'num_layers': self.model.config.num_hidden_layers,
            'num_attention_heads': self.model.config.num_attention_heads,
            'max_position_embeddings': self.model.config.max_position_embeddings,
            'device': str(self.device)
        }
    
    def validate_inputs(self, states: List[str], node_types: List[str], 
                       node_texts: List[str], node_codes: List[str]) -> Dict[str, Any]:
        """
        Validate input data for embedding generation.
        
        Args:
            states: List of node states
            node_types: List of node types
            node_texts: List of node text representations
            node_codes: List of node code contents
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'input_lengths': {
                'states': len(states),
                'node_types': len(node_types),
                'node_texts': len(node_texts),
                'node_codes': len(node_codes)
            }
        }
        
        # Check if all lists have the same length
        lengths = set([len(states), len(node_types), len(node_texts), len(node_codes)])
        if len(lengths) > 1:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Input lists have different lengths: {lengths}")
        
        # Check for empty inputs
        if len(states) == 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append("Empty input lists")
        
        # Check state values
        valid_states = ['before', 'unchanged', 'after']
        invalid_states = [s for s in states if s not in valid_states]
        if invalid_states:
            validation_results['warnings'].append(f"Invalid states found: {set(invalid_states)}")
        
        # Check for very long inputs
        max_length = self.config.codebert_max_tokens
        long_inputs = []
        for i, (state, node_type, node_code, node_text) in enumerate(zip(states, node_types, node_codes, node_texts)):
            total_length = len(self.tokenizer.tokenize(state)) + len(self.tokenizer.tokenize(node_type)) + \
                          len(self.tokenizer.tokenize(node_code)) + len(self.tokenizer.tokenize(node_text))
            if total_length > max_length:
                long_inputs.append(i)
        
        if long_inputs:
            validation_results['warnings'].append(f"Found {len(long_inputs)} inputs that may be truncated")
        
        return validation_results
    
    def optimize_batch_size(self, available_memory_gb: float = 8.0) -> int:
        """
        Suggest optimal batch size based on available memory.
        
        Args:
            available_memory_gb: Available GPU memory in GB
            
        Returns:
            Suggested batch size
        """
        if self.model is None:
            return self.config.codebert_batch_size
        
        # Estimate memory per sample
        hidden_size = self.model.config.hidden_size
        max_length = self.config.codebert_max_tokens
        
        # Rough memory estimation (in GB)
        memory_per_sample = (hidden_size * max_length * 4) / (1024**3)  # 4 bytes per float32
        
        # Calculate safe batch size
        safe_batch_size = int(available_memory_gb * 0.7 / memory_per_sample)  # Use 70% of available memory
        
        # Ensure batch size is within reasonable bounds
        safe_batch_size = max(1, min(safe_batch_size, 64))
        
        return safe_batch_size
    
    def save_model(self, model_path: str) -> None:
        """
        Save the CodeBERT model and tokenizer.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model to save. Initialize model first.")
        
        import os
        os.makedirs(model_path, exist_ok=True)
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        print(f"CodeBERT model saved to: {model_path}")
    
    def load_model(self, model_path: str) -> None:
        """
        Load a CodeBERT model and tokenizer.
        
        Args:
            model_path: Path to the model
        """
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        print(f"Loading CodeBERT model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        
        print("CodeBERT model loaded successfully!")
    
    def get_attention_weights(self, state: str, node_type: str, node_text: str, 
                            node_code: str) -> Optional[torch.Tensor]:
        """
        Get attention weights for a single node (for interpretability).
        
        Args:
            state: Node state
            node_type: Node type
            node_text: Node text representation
            node_code: Node code content
            
        Returns:
            Attention weights tensor or None if not available
        """
        if self.tokenizer is None or self.model is None:
            return None
        
        # This is a simplified version - full attention weight extraction depends on model architecture
        try:
            # Generate embeddings with output_attentions=True
            tokens = [self.tokenizer.cls_token] + self.tokenizer.tokenize(state) + [self.tokenizer.sep_token] + \
                    self.tokenizer.tokenize(node_type) + [self.tokenizer.sep_token] + self.tokenizer.tokenize(node_code) + \
                    [self.tokenizer.sep_token] + self.tokenizer.tokenize(node_text) + [self.tokenizer.eos_token]
            
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            inputs = torch.tensor(tokens_ids)[None, :].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(inputs, output_attentions=True)
            
            # Return attention weights from the last layer
            if hasattr(outputs, 'attentions') and outputs.attentions:
                return outputs.attentions[-1]  # Last layer attention
            
        except Exception as e:
            print(f"Error extracting attention weights: {e}")
        
        return None

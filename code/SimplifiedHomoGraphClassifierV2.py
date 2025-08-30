import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE  # Assuming GCN for homogeneous graphs
from torch.nn import LayerNorm, BatchNorm1d
from torch_geometric.nn.aggr import AttentionalAggregation

class HomogeneousGraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HomogeneousGraphConvolution, self).__init__()
        # Convolution layer for homogeneous graphs
        self.layernorm = LayerNorm(normalized_shape=out_channels)
        self.batchnorm = BatchNorm1d(out_channels)
        self.conv = GraphSAGE(in_channels, out_channels, num_layers=1, act='gelu')  # You can replace with another GNN layer
        self.gelu = nn.GELU()
    def forward(self, x, edge_index):
        x =  self.conv(x, edge_index)
        x = self.layernorm(x)  # Apply LayerNorm after the convolution
        return self.gelu(x)

class HomogeneousGraphSequentialClassifierV2(nn.Module):
    def __init__(self, in_channels, out_channels, device='cpu', with_manual_features=True, with_text_features=True, with_graph_features=True, manual_size=14):
        super(HomogeneousGraphSequentialClassifierV2, self).__init__()
        self.out_channels = out_channels
        self.device = device
        self.with_manual_features = with_manual_features
        self.with_text_features = with_text_features
        self.with_graph_features = with_graph_features
        self.manual_size = manual_size

        # Homogeneous Graph Convolution Layer
        self.homo_conv = HomogeneousGraphConvolution(in_channels, out_channels)

        # Attention pooling layer to get graph-level embeddings from node features
        self.global_attention = AttentionalAggregation(gate_nn=nn.Linear(out_channels, 1))  # Using gate NN for attention
        self.project_norm = LayerNorm(normalized_shape=in_channels)
        self.batchnorm = BatchNorm1d(in_channels)  # Applying BatchNorm to the input features

        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # Final fully connected layer for classification
        self.msg_fc = nn.Linear(768, out_channels)
        self.features_fc = nn.Linear(self.manual_size, out_channels)
        self.graph_weight = nn.Parameter(torch.tensor(1.0))
        self.num_components = 1 if self.with_graph_features else 0
        self.num_components = self.num_components + (1 if self.with_manual_features else 0)
        self.num_components = self.num_components + (1 if self.with_text_features else 0)
        
        self.mixed_norm = LayerNorm(normalized_shape=self.num_components*out_channels)
        self.fc1 = nn.Linear(self.num_components*out_channels, 1)

    def graph_embedding_gen(self, data):

        x = data['x_dict']
        x = self.project_norm(x)
        edge_index = data['edge_index']
        batch = data['batch']
        batch_size = data['batch_size']
        x = self.homo_conv(x, edge_index)
        graph_embedding, attn_weights = self.global_attention(x, index=batch, dim_size=batch_size)
        return graph_embedding, attn_weights

    def forward(self, data):
        embedding_features = []
        if self.with_graph_features:
            graph_embedding, graph_attn_weights = self.graph_embedding_gen(data)
            graph_embedding = self.dropout(graph_embedding)
            weighted_graph_embedding = self.graph_weight * graph_embedding
            embedding_features = [weighted_graph_embedding]
        else:
            graph_attn_weights = []
            graph_embedding = torch.empty((0,),dtype=torch.long)
        if self.with_text_features:
            commit_msg_data = data['text_embedding']
            commit_msg_embeddings = F.gelu(self.msg_fc(commit_msg_data))
            embedding_features.append(commit_msg_embeddings)
        if self.with_manual_features:
            features_embedding = data['features_embedding']
            features_embedding = F.gelu(self.features_fc(features_embedding))
            embedding_features.append(features_embedding)

        embeddings = torch.cat(embedding_features, dim=1)
        if self.num_components > 1:
            embeddings = self.mixed_norm(embeddings)
        logits = self.fc1(embeddings)

        return logits, graph_embedding, graph_attn_weights

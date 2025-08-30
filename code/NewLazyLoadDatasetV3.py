import itertools
import os

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch_geometric.data import Dataset, HeteroData, Data
from sklearn.preprocessing import RobustScaler
import torch_geometric.transforms as T

generic_callgraph_edge = "CALL"
generic_other_edges = "OTHER"
generic_pdg_edge = "PDG"
generic_ast_edge = "AST"
generic_cfg_edge = "CFG"
generic_self_edge = "SELF"

edge_type_mapping = {
    "AST": "AST",
    "CONDITION": "AST",
    "REF": "AST",
    "CFG": "CFG",
    "CDG": "PDG",
    "REACHING_DEF": "PDG",
    "CALL": "CALL",
    "ARGUMENT": "CALL",
    "RECEIVER": "CALL"
}

def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df

class NewLazyLoadDatasetV3(Dataset):

    def __init__(self, data_dir, projects=None, data_type='train', merge=True, old=True,
                 tokenizer=None, model=None, device='cpu', changes_data=None, features_data=None,
                 manual_features_columns=None, need_attentions=True, graph_edges=None, scaler=None):
        """
        Args:
            data_dir (string): Directory with all the graphs.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.msg_length = 64
        self.manual_features = []
        self.data_dir = data_dir
        self.projects = projects
        self.data_type = data_type
        self.cached_data = {}
        self.device = device
        self.changes_data = changes_data
        self.features_data = features_data
        self.manual_features_columns = manual_features_columns
        self.need_attentions = need_attentions
        self.text_attns = {}
        self.str_graph_edges = ''
        self.graph_edges = graph_edges
        if self.graph_edges is not None:
            self.graph_edges = [et for et in edge_type_mapping.keys() if edge_type_mapping[et] in self.graph_edges]
            self.str_graph_edges = '_'.join(graph_edges)
            print(self.graph_edges)
        if old:
            self.added_path_suffix = '_added_graph_p_1000_hetero_data.pt'
            self.removed_path_suffix = '_removed_graph_p_1000_hetero_data.pt'
        else:
            self.added_path_suffix = '_after_graphml_hetero_data_change_only.pt'
            self.removed_path_suffix = '_before_graphml_hetero_data_change_only.pt'
            self.path_suffix = '_hetero_data_1.pt'
        self.merge = merge
        self.scaler = RobustScaler() if self.data_type == 'train' else scaler
        self.bert_tokenizer = tokenizer
        self.bert_model = model
        if projects is not None:
            all_labels = []
            for project in projects:

                if os.path.exists(data_dir + '/' + data_type + '/' + project + '_' + data_type + '_NewLazyLoadDataset.pkl'):
                    data = pd.read_pickle(
                        data_dir + '/' + data_type + '/' + project + '_' + data_type + '_NewLazyLoadDataset.pkl')
                else:
                    data = pd.read_csv(data_dir + '/' + data_type + '/' + project + '_' + data_type + '_enhanced.csv', low_memory=False)
                    data = data.sort_values(by='author_date')
                    data = data.map(lambda x: np.nan if x is None else x, na_action='ignore')
                    data['code'] = data['commit_hash'].apply(lambda x: self.changes_data.loc[x][2])
                    data['commit_hash'] = project + '_' + data['commit_hash']
                    features = ['commit_hash', 'is_buggy_commit', 'commit_message', 'code']
                    man_features = data[self.manual_features_columns]
                    man_features = man_features.fillna(0)

                    features += self.manual_features_columns
                    data = data[features]

                    man_features = convert_dtype_dataframe(man_features, manual_features_columns)
                    man_features = preprocessing.scale(man_features[manual_features_columns].to_numpy())
                    # if self.scaler is not None:
                        
                    #     if self.data_type == 'train':  
                    #         print('scaling manual features for train', flush=True)
                    #         man_features[manual_features_columns] = self.scaler.fit_transform(man_features[manual_features_columns].to_numpy())
                    #     else:
                    #         print('scaling manual features for test/valid', flush=True)
                    #         man_features[manual_features_columns] = self.scaler.transform(man_features[manual_features_columns].to_numpy())
                    # man_features.set_index(['project', 'commit_hash'], inplace=True)
                    data[self.manual_features_columns] = man_features
                    data['commit_message'] = data['commit_message'].fillna('commit message').astype(str)
                    data['commit_message'] = data['commit_message'].apply(lambda x: str(x))
                    # data['text_embedding'] = data[["commit_message", "code"]].apply(lambda x: self.text_embeddings(x['commit_message'], x['code']), axis=1)
                    data.to_pickle(data_dir + '/' + data_type + '/' + project + '_' + data_type + '_NewLazyLoadDataset.pkl')
                all_labels.append(data)
            self.labels = pd.concat(all_labels)
        # shuffle labels
        if data_type == 'train':
            self.labels = self.labels.sample(frac=1)
        self.labels.reset_index(inplace=True)
        self.labels.set_index('commit_hash', inplace=True)
        self.verify_data()
        self.pos_count = self.labels['is_buggy_commit'].sum()
        self.neg_count = len(self.labels) - self.labels['is_buggy_commit'].sum()
        print(f"Data type: {self.data_type}, pos count: {self.pos_count}, neg count: {self.neg_count}")

    def text_embeddings(self, commit_msg, code):
        # source
        added_tokens = []
        removed_tokens = []
        msg_tokens = self.bert_tokenizer.tokenize(commit_msg)
        msg_tokens = msg_tokens[:min(self.msg_length, len(msg_tokens))]
        # for file_codes in files:
        file_codes = code
        added_codes = [' '.join(line.split()) for line in file_codes['added_code']]
        removed_codes = [' '.join(line.split()) for line in file_codes['removed_code']]

        codes = '[ADD]'.join([line for line in added_codes if len(line)])
        added_tokens.extend(self.bert_tokenizer.tokenize(codes))

        codes = '[DEL]'.join([line for line in removed_codes if len(line)])
        removed_tokens.extend(self.bert_tokenizer.tokenize(codes))

        input_tokens = msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens

        input_tokens = input_tokens[:512 - 2]

        tokens = [self.bert_tokenizer.cls_token] + input_tokens + [self.bert_tokenizer.sep_token]
        tokens_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(tokens_ids)

        # Convert to tensors and add batch dimension (batch size = 1)
        input_ids = torch.tensor(tokens_ids).unsqueeze(0)  # shape: (1, sequence_length)
        attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # shape: (1, sequence_length)

        # Ensure tensors are on the correct device (CPU or GPU)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Perform inference with no gradient tracking (for efficiency)
        with torch.no_grad():
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the hidden state of the [CLS] token (index 0) from the last layer
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
        last_layer_attentions = outputs.attentions[-1]

        return {
            'cls_embedding': cls_embedding.squeeze(0).cpu().numpy(),
            'attn_weights': last_layer_attentions.cpu().numpy(),
            'tokens': tokens
        }

    def verify_data(self):
        print(f"verifying data {self.data_type}")
        for commit_hash in self.labels.index:
            try:
                project_name = commit_hash.split('_')[0]
                commit_id = commit_hash.split('_')[1]
                graph_path = os.path.join(self.data_dir, project_name,
                                                commit_id + self.path_suffix)

                if not os.path.exists(graph_path):
                    print(f"drop missing {graph_path}", flush=True)
                    self.labels.drop(commit_hash, inplace=True)
                # else:
                #     count_non_empty = 0
                #     try:
                #         if os.path.exists(graph_path):
                #             # print(f" added_graph_path {added_graph_path}", flush=True)
                #             graph = torch.load(graph_path, weights_only=False)
                #             if graph.num_nodes > 0:
                #                 count_non_empty += 1
                #     except Exception as e:
                #         print(f"error reading file {graph_path}, drop the commit",
                #               flush=True)
                #         self.labels.drop(commit_hash, inplace=True)
                #         count_non_empty = -1
                #     if (self.data_type == 'train' or self.labels.at[
                #         commit_hash, 'is_buggy_commit'] == 1) and count_non_empty == 0:
                #         print(f"Dropping wrong data: buggy commit {commit_hash} due to no nodes in graphs", flush=True)
                #         self.labels.drop(commit_hash, inplace=True)


            except Exception as e:
                print(f"{commit_hash} {e}", flush=True)

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels['is_buggy_commit'].tolist()

    def get_projects(self):
        try:
            return [commit_hash.split('_')[0] for commit_hash in self.labels.index]
        except Exception as e:
            print(self)

    def get_item_from_commit_hash(self, commit_hash):
        if commit_hash in self.cached_data:
            cached_data =  self.cached_data[commit_hash]
            return cached_data
        # print(f"commit_hash {commit_hash}", flush=True)
        project_name = commit_hash.split('_')[0]
        commit_id = commit_hash.split('_')[1]

        if len(commit_hash.split('_')) == 3:
            commit_id = commit_hash.split('_')[1] + '_' + commit_hash.split('_')[2]
        if self.graph_edges is None:
            self.str_graph_edges = ''
        path = os.path.join(self.data_dir, project_name, commit_id + 'NewLazyLoadDatasetV3' + self.str_graph_edges + '.pt')
        if os.path.exists(path):
            try:
                sample = torch.load(path, weights_only=False)
                return sample
            except Exception as e:
                print(f"error {e} {path}", flush=True)

        graph_path = os.path.join(self.data_dir, project_name, commit_id + self.path_suffix)
        try:
            graph = torch.load(graph_path, weights_only=False)
            if self.graph_edges:
                edge_types = [et for et in graph.edge_types if et[1] in self.graph_edges]
                graph = graph.edge_type_subgraph(edge_types=edge_types)
        except Exception as e:
            # print(f"error reading added graph {added_graph_path}", flush=True)
            graph = HeteroData()
        num_to_node_id_map = {}
        node_id_to_num_map = {}

        # for i in range(0, graph.num_nodes):

        i = 0
        for node_storage in graph.node_stores:
            ids = []
            for node_id in node_storage.id:
                num_to_node_id_map[i] = node_id
                node_id_to_num_map[node_id] = id
                ids.append(i)
                i += 1
            node_storage.id = torch.tensor(ids)
        # if i == 0:
        #     print(graph)
        id_mapping_path = os.path.join(self.data_dir, project_name, commit_id + 'id_mapping.pt')
        torch.save((num_to_node_id_map, node_id_to_num_map), id_mapping_path)
        label = self.labels.at[commit_hash, 'is_buggy_commit']
        # text_embedding = self.labels.at[commit_hash, 'text_embedding']
        # if self.need_attentions:
        #     self.text_attns[commit_hash] = text_embedding['attn_weights']
        # text_embedding = torch.tensor(text_embedding['cls_embedding']).to(self.device)

        commit_message = str(self.labels.at[commit_hash, 'commit_message'])
        code = str(self.labels.at[commit_hash, 'code'])
        feature_embedding = torch.tensor(self.labels.loc[commit_hash][self.manual_features_columns].to_numpy(dtype=np.float32)).to(self.device)
        sample = (graph, commit_message, code, feature_embedding, label, commit_hash)

        sample = transformation(sample)

        torch.save(sample, path)
        self.cached_data[commit_hash] = sample
        return sample

    def get_sample_weights(self):
        pos_count = self.labels['is_buggy_commit'].sum()
        neg_count = len(self.labels) - self.labels['is_buggy_commit'].sum()
        total = pos_count + neg_count
        # give weight to each sample
        pos_weight = pos_count / total
        neg_weight = neg_count / total
        sample_weights = self.labels['is_buggy_commit'].apply(lambda x: neg_weight if x == 1 else neg_weight)
        return torch.tensor(np.array(sample_weights), dtype=torch.float32)


    def __getitem__(self, idx):
        if isinstance(idx, list):
            batch = [self.get_item_from_commit_hash(i) for i in idx]
            return batch
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, str):
            return self.get_item_from_commit_hash(idx)
        else:
            commit_hash = self.labels.index[idx]
            return self.get_item_from_commit_hash(commit_hash)

def to_homogeneous(graph_data):
    if graph_data.num_nodes == 0 or graph_data.num_edges == 0 or len(graph_data.edge_types) == 0:
        data = Data()
        data.x = torch.empty((0, 769 + 128))  # Replace `feature_dim` with the dimensionality of your features
        data.edge_index = torch.empty((2, 0), dtype=torch.long)  # No edges
        data.node_type = torch.empty((0,),
                                     dtype=torch.long)  # Use long if node types are represented as integers
        data.edge_type = torch.empty((0,), dtype=torch.long)
        data.id = torch.empty((0,),dtype=torch.long)
        return data
    homogeneous_data = graph_data.to_homogeneous(
        node_attrs=['x','id'],  # Combine all node attributes automatically if compatible
        edge_attrs=None,
        add_node_type=True,  # Add node_type attribute to keep track of original node types
        add_edge_type=True,  # Add edge_type attribute to keep track of original edge types
        dummy_values=True  # Fill missing attributes with dummy values (NaN, False, -1)
    )
    return homogeneous_data

def transformation(sample):
    graph, commit_message, code, feature_embedding, label, commit = sample
    graph = T.NormalizeFeatures()(graph)
    graph = to_homogeneous(graph)
    return graph, commit_message, code,feature_embedding, commit, label

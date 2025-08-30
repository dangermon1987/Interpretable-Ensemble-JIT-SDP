import base64
import logging
import math
import os
import re
import sys
import pickle
from datetime import datetime
from linecache import cache
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import f1_score, accuracy_score, precision_score, roc_auc_score, matthews_corrcoef, \
    precision_recall_fscore_support, confusion_matrix, auc
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sympy.stats.sampling.sample_numpy import numpy
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

from NewLazyLoadDatasetV3 import NewLazyLoadDatasetV3
from SimplifiedHomoGraphClassifierV2 import HomogeneousGraphSequentialClassifierV2
from EarlyStopping import EarlyStopping

# sys.stdout = open(f'gen_embeddings.log', 'w')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_dtype_dataframe(df, feature_name):
    df['fix'] = df['fix'].apply(lambda x: float(bool(x)))
    df = df.astype({i: 'float32' for i in feature_name})
    return df


# Load the dataset
project_list = ['ant-ivy','commons-bcel','commons-beanutils','commons-codec','commons-collections','commons-compress',
               'commons-configuration','commons-dbcp','commons-digester','commons-io','commons-jcs', 'commons-lang','commons-math',
               'commons-net','commons-scxml','commons-validator','commons-vfs','giraph','gora','opennlp','parquet-mr']

data_dir = '/workspace/s2156631-thesis/new_before_after_data/output/'
model_path = "/workspace/s2156631-thesis/data/codebert-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]}
bert_tokenizer.add_special_tokens(special_tokens_dict)

bert_model = AutoModel.from_pretrained(model_path, output_attentions=True)
bert_model.resize_token_embeddings(len(bert_tokenizer))

bert_model.to(device)

buggy_line_filepath = data_dir + '/changes_complete_buggy_line_level.pkl'
# Load datasets
generic_callgraph_edge = "CALL"
generic_other_edges = "OTHER"
generic_pdg_edge = "PDG"
generic_ast_edge = "AST"
generic_cfg_edge = "CFG"
generic_self_edge = "SELF"

graph_edges = [generic_ast_edge, generic_callgraph_edge, generic_cfg_edge, generic_pdg_edge, generic_other_edges,
               generic_self_edge]

graph_edges = [generic_pdg_edge, generic_cfg_edge, generic_callgraph_edge]
graph_edges = [generic_ast_edge]
graph_edges = None
str_edges = '_'.join(graph_edges) if graph_edges is not None else ''

manual_features_columns = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
                           'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']

# Load datasets
train_dataset_pkl = 'train_dataset_v3_21' + '_' + str_edges + '.pkl'
# train changes
train_changes_file = data_dir + "/train/changes_train.pkl"
train_changes_data = pd.read_pickle(train_changes_file)
train_changes_data = pd.DataFrame(np.array(train_changes_data).T,
                                  columns=['commit_hash', 'label', 'commit_message', 'code'])
train_changes_data.set_index('commit_hash', inplace=True)

if os.path.exists(train_dataset_pkl):
    with open(train_dataset_pkl, 'rb') as handle:
        dataset = pickle.load(handle)
else:
    dataset = NewLazyLoadDatasetV3(data_dir=data_dir, projects=project_list, data_type='train', merge=True,
                                   old=False, device=device, tokenizer=bert_tokenizer, model=bert_model,
                                   changes_data=train_changes_data, manual_features_columns=manual_features_columns,
                                   graph_edges=graph_edges)
    with open(train_dataset_pkl, 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"train_set {dataset.pos_count} vs {dataset.neg_count}", flush=True)

validation_set_pkl = 'valid_dataset_v3_21' + '_' + str_edges + '.pkl'
# valid changes
valid_changes_file = data_dir + "/valid/changes_valid.pkl"
valid_changes_data = pd.read_pickle(valid_changes_file)
valid_changes_data = pd.DataFrame(np.array(valid_changes_data).T,
                                  columns=['commit_hash', 'label', 'commit_message', 'code'])
valid_changes_data.set_index('commit_hash', inplace=True)

if os.path.exists(validation_set_pkl):
    with open(validation_set_pkl, 'rb') as handle:
        validation_set = pickle.load(handle)
else:
    validation_set = NewLazyLoadDatasetV3(data_dir=data_dir, projects=project_list, data_type='valid', merge=True,
                                          device=device,
                                          old=False, tokenizer=bert_tokenizer, model=bert_model,
                                          changes_data=valid_changes_data,
                                          manual_features_columns=manual_features_columns,
                                          graph_edges=graph_edges)
    with open(validation_set_pkl, 'wb') as handle:
        pickle.dump(validation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"valid_set {validation_set.pos_count} vs {validation_set.neg_count}", flush=True)
test_set_pkl = 'test_dataset_v3_21' + '_' + str_edges + '.pkl'

# test changes
test_changes_file = data_dir + "/test/changes_test.pkl"
test_changes_data = pd.read_pickle(test_changes_file)
test_changes_data = pd.DataFrame(np.array(test_changes_data).T,
                                 columns=['commit_hash', 'label', 'commit_message', 'code'])
test_changes_data.set_index('commit_hash', inplace=True)

if os.path.exists(test_set_pkl):
    with open(test_set_pkl, 'rb') as handle:
        test_set = pickle.load(handle)
else:
    test_set = NewLazyLoadDatasetV3(data_dir=data_dir, projects=project_list, data_type='test', merge=True, old=False,
                                    device=device, tokenizer=bert_tokenizer, model=bert_model,
                                    changes_data=test_changes_data, manual_features_columns=manual_features_columns,
                                    graph_edges=graph_edges)
    with open(test_set_pkl, 'wb') as handle:
        pickle.dump(test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"test_set {test_set.pos_count} vs {test_set.neg_count}", flush=True)


def line_evaluation(runs_dir):
    model_path = '{}/{}'.format(runs_dir, 'model.bin')
    meta_path = '{}/{}'.format(runs_dir, 'meta.txt')

    logger = logging.getLogger(__name__)

    meta = torch.load(meta_path, weights_only=False)
    batch_size = 128
    # batch_size = 1
    node_feature_dimension = meta['node_feature_dimension']
    out_channels = meta['out_channels']
    in_channels = meta['in_channels']
    pos_weight = meta['pos_weight']
    lr = meta['lr']
    manual_features_columns = meta['manual_features_columns']
    optimizer_grouped_parameters = meta['optimizer_grouped_parameters']
    sample_weights = meta['sample_weights']
    weight_decay = meta['weight_decay']
    no_decay = meta['no_decay']
    criterion = meta['criterion']
    scheduler = meta['scheduler']
    max_steps = meta['max_steps']
    ALPHA = meta['ALPHA']
    GAMMA = meta['GAMMA']
    project_list = meta['project_list']
    print(project_list)
    print(meta['graph_edges'])
    model = HomogeneousGraphSequentialClassifierV2(in_channels=in_channels, out_channels=out_channels, device=device,
                                                   with_manual_features=False, with_text_features=True,
                                                   with_graph_features=True).to(
        device)

    model_state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(model_state_dict['model_state_dict'])

    class FocalLoss(nn.Module):
        def __init__(self, weight=None, size_average=True):
            super(FocalLoss, self).__init__()

        def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
            # comment out if your model contains a sigmoid or equivalent activation layer
            inputs = F.sigmoid(inputs)

            # flatten label and prediction tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)

            # first compute binary cross-entropy
            BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
            BCE_EXP = torch.exp(-BCE)
            focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

            return focal_loss

    def text_embeddings(commit_msgs, codes, full=False):
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
            msg_tokens = bert_tokenizer.tokenize(commit_msg)
            msg_tokens = msg_tokens[:min(64, len(msg_tokens))]
            commit_msgs_tokens_batch.append(msg_tokens)
            code = eval(code)

            # Process added and removed codes
            added_codes = [' '.join(line.split()) for line in code['added_code']]
            removed_codes = [' '.join(line.split()) for line in code['removed_code']]

            codes_added = '[ADD]'.join([line for line in added_codes if len(line)])
            added_tokens_batch.append(bert_tokenizer.tokenize(codes_added))

            codes_removed = '[DEL]'.join([line for line in removed_codes if len(line)])
            removed_tokens_batch.append(bert_tokenizer.tokenize(codes_removed))

        # Prepare inputs for batch encoding
        input_ids = []
        attention_masks = []

        for msg_tokens, added_tokens, removed_tokens in zip(commit_msgs_tokens_batch, added_tokens_batch,
                                                            removed_tokens_batch):
            input_tokens = msg_tokens + ['[ADD]'] + added_tokens + ['[DEL]'] + removed_tokens
            input_tokens = input_tokens[:512 - 2]

            tokens = [bert_tokenizer.cls_token] + input_tokens + [bert_tokenizer.sep_token]
            tokens_ids = bert_tokenizer.convert_tokens_to_ids(tokens)

            attention_mask = [1] * len(tokens_ids)

            input_ids.append(tokens_ids)
            attention_masks.append(attention_mask)

            # Store the tokens for this input
            all_tokens.append(tokens)

        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True)
        attention_masks = torch.nn.utils.rnn.pad_sequence([torch.tensor(mask) for mask in attention_masks],
                                                          batch_first=True)

        # Ensure tensors are on the correct device (CPU or GPU)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        # Perform inference with no gradient tracking (for efficiency)
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_masks)

        # Extract the hidden state of the [CLS] token (index 0) from the last layer
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # shape: (batch_size, hidden_size)
        last_layer_attentions = outputs.attentions[-1].cpu()

        if full:
            batch_attns = []
            for i in range(0, len(all_tokens)):
                input_tokens = all_tokens[i]
                begin_pos = input_tokens.index('[ADD]')
                end_pos = input_tokens.index('[DEL]') if '[DEL]' in input_tokens else len(input_tokens) - 1
                attns = last_layer_attentions[i]
                attns = attns.mean(axis=0)[0][begin_pos:end_pos]
                batch_attns.append(attns)
            return [{
                'cls_embeddings': cls_embeddings[i],  # shape: (batch_size, hidden_size)
                'attn_weights': attn,  # shape: (batch_size, num_heads, seq_length, seq_length)
                'tokens': token  # List of token lists for each input
            } for i, attn, token in zip(range(0, len(all_tokens)), batch_attns, all_tokens)]
        else:
            return cls_embeddings

    train_text_embeddings = {}
    train_text_embeddings_path = "train_text_embeddings.pkl"
    if not os.path.exists(train_text_embeddings_path):
        train_data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for batch in train_data_loader:
            homogeneous_data, commit_message, code, features_embedding, commits, labels = batch
            text_embedding = text_embeddings(commit_message, code, full=True)
            index = 0
            for commit_id in commits:
                train_text_embeddings[commit_id] = text_embedding[index]
                index += 1
        torch.save(train_text_embeddings, "train_text_embeddings.pkl")
    else:
        train_text_embeddings = torch.load(train_text_embeddings_path, weights_only=False)

    valid_text_embeddings = {}
    valid_text_embeddings_path = "valid_text_embeddings.pkl"
    if not os.path.exists(valid_text_embeddings_path):
        valid_data_loader = DataLoader(validation_set, batch_size=32, shuffle=False)
        for batch in valid_data_loader:
            homogeneous_data, commit_message, code, features_embedding, commits, labels = batch
            text_embedding = text_embeddings(commit_message, code, full=True)
            index = 0
            for commit_id in commits:
                valid_text_embeddings[commit_id] = text_embedding[index]
                index += 1
        torch.save(valid_text_embeddings, "valid_text_embeddings.pkl")
    else:
        valid_text_embeddings = torch.load(valid_text_embeddings_path, weights_only=False)

    test_text_embeddings = {}
    test_text_embeddings_path = "test_text_embeddings.pkl"
    if not os.path.exists(test_text_embeddings_path):
        test_data_loader = DataLoader(test_set, batch_size=32, shuffle=False)
        for batch in test_data_loader:
            homogeneous_data, commit_message, code, features_embedding, commits, labels = batch
            full = True
            text_embedding = text_embeddings(commit_message, code, full=full)
            index = 0
            for commit_id in commits:
                test_text_embeddings[commit_id] = text_embedding[index]
                index += 1
        torch.save(test_text_embeddings, "test_text_embeddings.pkl")
    else:
        test_text_embeddings = torch.load(test_text_embeddings_path, weights_only=False)

    def get_line_level_metrics(line_score, label):
        scaler = MinMaxScaler()
        try:
            line_score = scaler.fit_transform(np.array(line_score).reshape(-1, 1))  # cannot pass line_score as list T-T
        except Exception as e:
            print(e)
        pred = np.round(line_score)

        line_df = pd.DataFrame()
        line_df['scr'] = [float(val.item()) for val in list(line_score)]
        line_df['label'] = label
        line_df['pred'] = pred
        line_df = line_df.sort_values(by='scr', ascending=False)
        line_df['row'] = np.arange(1, len(line_df) + 1)

        real_buggy_lines = line_df[line_df['label'] == 1]

        top_10_acc = 0
        top_5_acc = 0

        if len(real_buggy_lines) < 1:
            IFA = len(line_df)
            top_20_percent_LOC_recall = 0
            effort_at_20_percent_LOC_recall = math.ceil(0.2 * len(line_df))

        else:
            IFA = line_df[line_df['label'] == 1].iloc[0]['row'] - 1
            label_list = list(line_df['label'])
            # score_list = list(line_df['scr'])
            # rows = list(line_df['row'])
            # print(label_list)
            # print(score_list)
            # print(rows)

            all_rows = len(label_list)

            # find top-10 accuracy
            if all_rows < 10:
                top_10_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
            else:
                top_10_acc = np.sum(label_list[:10]) / len(label_list[:10])

            # find top-5 accuracy
            if all_rows < 5:
                top_5_acc = np.sum(label_list[:all_rows]) / len(label_list[:all_rows])
            else:
                top_5_acc = np.sum(label_list[:5]) / len(label_list[:5])

            # find recall
            LOC_20_percent = line_df.head(int(0.2 * len(line_df)))
            buggy_line_num = LOC_20_percent[LOC_20_percent['label'] == 1]
            top_20_percent_LOC_recall = float(len(buggy_line_num)) / float(len(real_buggy_lines))

            # find effort @20% LOC recall

            buggy_20_percent = real_buggy_lines.head(math.ceil(0.2 * len(real_buggy_lines)))
            buggy_20_percent_row_num = buggy_20_percent.iloc[-1]['row']
            effort_at_20_percent_LOC_recall = int(buggy_20_percent_row_num) / float(len(line_df))

        return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc

    def commit_with_codes(filepath, tokenizer):
        data = pd.read_pickle(filepath)
        commit2codes = []
        idx2label = []
        commit_line_label = {}
        for _, item in data.iterrows():
            commit_id, idx, changed_type, label, raw_changed_line, changed_line = item
            if commit_id not in commit_line_label:
                commit_line_label[commit_id] = []
            commit_line_label[commit_id].append([(changed_type, idx, raw_changed_line, label)])
            line_tokens = [token for token in tokenizer.tokenize(changed_line)]
            for token in line_tokens:
                commit2codes.append([commit_id, idx, changed_type, token])
            idx2label.append([commit_id, idx, label])
        commit2codes = pd.DataFrame(commit2codes, columns=['commit_id', 'idx', 'changed_type', 'token'])
        idx2label = pd.DataFrame(idx2label, columns=['commit_id', 'idx', 'label'])
        return commit2codes, idx2label, commit_line_label

    def preprocess_code_line(code, remove_python_common_tokens=False):
        code = code.replace('(', ' ').replace(')', ' ').replace('{', ' ').replace('}', ' ').replace('[', ' ').replace(
            ']',
            ' ').replace(
            '.', ' ').replace(':', ' ').replace(';', ' ').replace(',', ' ').replace(' _ ', '_')

        code = re.sub('``.*``', '<STR>', code)
        code = re.sub("'.*'", '<STR>', code)
        code = re.sub('".*"', '<STR>', code)
        code = re.sub('\d+', '<NUM>', code)

        code = code.split()
        code = ' '.join(code)
        if remove_python_common_tokens:
            new_code = ''
            python_common_tokens = []
            for tok in code.split():
                if tok not in [python_common_tokens]:
                    new_code = new_code + tok + ' '

            return new_code.strip()

        else:
            return code.strip()

    def add_subset_scores_optimized(df1, df2, df1_code='code', df1_score='score', df2_code='code', df2_score='score'):
        # Initialize a Series to hold updated scores
        updated_scores = df1[df1_score].copy()

        # Iterate through each code in df2
        for index2, row2 in df2.iterrows():
            # Create a boolean mask for matching codes
            mask = df1[df1_code].str.contains(''.join(row2[df2_code].split()), regex=False)

            # Update scores where the mask is True
            updated_scores[mask] += row2[df2_score]

        df2_codes = df2[df2_code].tolist()
        df2_codes = [''.join(code.split()) for code in df2_codes]

        # Check if codes in df1 are substrings of any codes in df2
        for index1, row1 in df1.iterrows():
            if any(row1[df1_code] in code for code in df2_codes):
                # Add the score from df2 if a match is found
                updated_scores[index1] += df2[df2[df2_code].str.contains(row1[df1_code], regex=False)][df2_score].sum()

        # Update the score column in df1
        df1[df1_score] = updated_scores
        return df1

    def deal_with_attns(item, text_attns, pred, commit2codes, idx2label, graph_attns=None, line_code_label=None,
                        only_adds=True):
        commit_id = item.commit_id
        input_tokens = item.input_tokens
        commit_label = item.label

        # Remove msg, cls, eos, del
        begin_pos = input_tokens.index('[ADD]')
        end_pos = input_tokens.index('[DEL]') if '[DEL]' in input_tokens else len(input_tokens) - 1

        attn_df = pd.DataFrame()
        attn_df['token'] = [token for token in input_tokens[begin_pos:end_pos]]
        attn_df['score'] = text_attns
        attn_df = attn_df.sort_values(by='score', ascending=False)
        attn_df = attn_df.groupby(['token']).sum().reset_index(drop=False)

        node_attn_df = None  # Initialize node_attn_df for later use

        if graph_attns is not None:
            nodes_attn, nodes_code = graph_attns
            # Create a DataFrame for node attention scores
            node_attn_df = pd.DataFrame({
                'node_code': [' '.join(text.split('|')[3].split()) for text in nodes_code],
                'node_score': [attn[0].item() for attn in nodes_attn]
            })

            node_attn_df = node_attn_df[node_attn_df['node_code'] != '']
            node_attn_df = node_attn_df[node_attn_df['node_code'] != '<EMPTY>']
            node_attn_df = node_attn_df[node_attn_df['node_code'] != '<empty>']

            change_types = []
            codes = []
            labels = []
            idxs = []

            # for item in line_code_label:
            #     change, idx, code, label = item[0]
            #     change_types.append(change)
            #     codes.append(code)
            #     labels.append(label)
            #     idxs.append(idx)

            # line_code_label_df = pd.DataFrame({
            #     'change_type': change_types,
            #     'idx': idxs,
            #     'code': [''.join(code.split()) for code in codes],
            #     'score': [0.0 for _ in codes],
            #     'label': labels
            # })
            # if only_adds:
            #     line_code_label_df = line_code_label_df[line_code_label_df['change_type'] == 'added']

            # line_code_label_df = add_subset_scores_optimized(line_code_label_df, node_attn_df, df2_code='node_code', df2_score='node_score')

            threshold_index = int(len(node_attn_df) * 0.2)  # Top 20%
            node_attn_df['node_score'] = pd.to_numeric(node_attn_df['node_score'], errors='coerce')
            top_nodes = node_attn_df.nlargest(threshold_index, 'node_score')

            # Tokenize the node codes and distribute scores
            flat_tokenized_nodes = []
            flat_node_scores = []

            for i, row in top_nodes.iterrows():
                tokens = bert_tokenizer.tokenize(row['node_code'])
                if tokens:  # Only consider non-empty tokenized codes
                    for token in tokens:
                        flat_tokenized_nodes.append(token)
                        flat_node_scores.append(row['node_score'] / len(tokens))

            # Create a DataFrame for the tokenized node attention scores
            tokenized_node_attn_df = pd.DataFrame({
                'token': [token for token in flat_tokenized_nodes],
                'node_score': flat_node_scores
            })

            # Group by token and sum the node scores
            tokenized_node_attn_df = tokenized_node_attn_df.groupby('token').sum().reset_index()

        # Calculate score for each line in commit

        if only_adds:
            commit2codes = commit2codes[commit2codes['changed_type'] == 'added']  # Only count for added lines

        commit2codes = commit2codes.drop(['commit_id', 'changed_type'], axis=1)

        line_scores = pd.merge(commit2codes, attn_df, how='left', on='token')
        line_scores['score'] = line_scores['score'].fillna(0)
        line_scores = line_scores.groupby(['idx', 'token']).sum().reset_index(drop=False)

        node_scores = pd.merge(commit2codes, tokenized_node_attn_df, how='left', on='token', suffixes=('', '_node'))
        node_scores['node_score'] = node_scores['node_score'].fillna(0)
        node_scores = node_scores.groupby(['idx', 'token']).sum().reset_index(drop=False)

        # node_scores = pd.merge(node_scores, line_scores, how='left', on='token', suffixes=('', '_text'))
        # node_scores['score'] = node_scores['score'].fillna(0)
        # node_scores['node_score'] = node_scores['node_score'] + node_scores['score']
        # node_scores = node_scores.groupby(['idx', 'token']).sum().reset_index(drop=False)

        # line_code_label_df = line_code_label_df.groupby('idx').sum().reset_index(drop=False)
        node_scores = node_scores.groupby('idx').agg({
            'node_score': 'sum'
        }).reset_index(drop=False)
        node_scores = pd.merge(node_scores, idx2label, how='inner', on='idx')

        line_scores = line_scores.groupby('idx').agg({
            'score': 'sum'
        }).reset_index(drop=False)
        line_scores = pd.merge(line_scores, idx2label, how='inner', on='idx')

        # Ensemble scores: you can adjust how you want to combine the scores
        final_scores = pd.merge(line_scores, node_scores, how='left', on='idx', suffixes=('', '_combined'))
        scaler = RobustScaler()

        # Scale the 'score' and 'node_score' columns
        final_scores[['score', 'node_score']] = scaler.fit_transform(final_scores[['score', 'node_score']])

        final_scores['final_score'] = final_scores[['score', 'node_score']].max(axis=1)

        # final_scores = final_scores.groupby('idx').sum().reset_index(drop=False)
        # Grouping by index for final results
        final_scores = final_scores.groupby('idx').agg({
            'score': 'sum',
            'node_score': 'sum',
            'final_score': 'sum'
        }).reset_index()

        final_scores = pd.merge(final_scores, idx2label, how='inner', on='idx')
        # final_scores = final_scores.groupby('idx').sum().reset_index(drop=False)

        # Calculate metrics using both scores
        metrics_line = get_line_level_metrics(line_scores['score'].tolist(), line_scores['label'].tolist())
        metrics_node_1 = get_line_level_metrics(node_scores['node_score'].tolist(), node_scores['label'].tolist())
        # metrics_node = get_line_level_metrics(line_code_label_df['score'].tolist(), line_code_label_df['label'].tolist())
        metrics_final = get_line_level_metrics(final_scores['final_score'].tolist(), final_scores['label'].tolist())

        return metrics_line, metrics_node_1, metrics_final

    def decode_from_base64(base64_string):
        # Convert the Base64 string to bytes
        base64_bytes = base64_string.encode('utf-8')
        # Decode the Base64 bytes to original bytes
        byte_data = base64.b64decode(base64_bytes)
        # Convert the bytes back to a string
        original_string = byte_data.decode('utf-8')
        return original_string

    def evaluate(model, dataset, text_embeddings, line_eval=False, epoch=None, early_stopping=None):
        start_eval_time = datetime.now()
        model.eval()
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0
        all_prods = []
        all_labels = []
        all_embeddings = []
        all_graph_attns = []
        # To store gradients for correlation calculation
        all_weights = {}
        all_grads = {}
        all_commits = []

        with torch.no_grad():
            for batch in data_loader:
                homogeneous_data, commit_message, code, features_embedding, commits, labels = batch
                text_embedding = torch.stack([text_embeddings[commit_id]['cls_embeddings'] for commit_id in commits],
                                             dim=0)
                data = {
                    'x_dict': homogeneous_data.x.to(device),
                    'edge_index': homogeneous_data.edge_index.to(device),
                    'batch': homogeneous_data.batch.to(device),
                    'text_embedding': text_embedding.to(device),
                    'features_embedding': features_embedding.to(device),
                    'batch_size': len(labels)
                }
                logits, embeddings, graph_attn_weights = model(data)
                all_graph_attns.extend(graph_attn_weights)
                labels = labels.unsqueeze(-1)
                loss = criterion(logits, labels.to(device))
                total_loss += loss.item()
                prods = torch.sigmoid(logits).cpu().detach().numpy()
                all_prods.extend(prods)
                all_labels.extend(labels)
                all_embeddings.extend(embeddings.cpu().detach().numpy())
                all_commits.extend(commits)

        end_eval_time = datetime.now()
        val_loss = total_loss / len(data_loader)
        print(f"Evaluation took {end_eval_time - start_eval_time} seconds")
        val_f1, val_acc, val_precision, val_auc, val_mcc = calculate_metrics(all_prods, all_labels)
        print(
            f"{dataset.data_type} Loss: {val_loss}, F1: {val_f1}, Accuracy: {val_acc}, Precision: {val_precision}, AUC: {val_auc}, MCC: {val_mcc}")
        all_prods = np.array(all_prods)
        y_preds = all_prods[:, -1] > 0.5
        result = []
        for commit_id, pred, prob in zip(dataset.labels.index, y_preds, all_prods[:, -1]):
            result.append([commit_id.split('_')[1], prob, pred, dataset.labels.loc[commit_id]['is_buggy_commit']])
        RF_result = pd.DataFrame(result)
        RF_result.to_csv(runs_dir + '/' + dataset.data_type + "_predictions.csv", sep='\t', index=None)
        
        if not line_eval:
            return (all_commits, all_embeddings, all_graph_attns, all_labels), RF_result
        

        cache_buggy_line = os.path.join(os.path.dirname(buggy_line_filepath),
                                        'changes_complete_buggy_line_level_cache.pkl')
        
        if os.path.exists(cache_buggy_line):
            commit2codes, idx2label, commit_line_label = pickle.load(open(cache_buggy_line, 'rb'))
        else:
            commit2codes, idx2label, commit_line_label = commit_with_codes(buggy_line_filepath, bert_tokenizer)
            pickle.dump((commit2codes, idx2label, commit_line_label), open(cache_buggy_line, 'wb'))

        print(len(idx2label))
        df = dataset.labels[['is_buggy_commit']]
        df['prob'] = all_prods
        commit_idx = 0
        line_metrics, node_metrics_1, node_metrics, combined_metrics = [], [], [], []
        for commit_project_id, example in df.iterrows():
            commit_id = commit_project_id.split('_')[1]
            project_name = commit_project_id.split('_')[0]
            prob = example['prob']

            pred = 1.0 if prob > 0.5 else 0.0
            attn = text_embeddings[commit_project_id]['attn_weights']
            label = example['is_buggy_commit']
            input_tokens = text_embeddings[commit_project_id]['tokens']
            graph_attn = all_graph_attns[commit_idx]
            id_mapping_path = os.path.join(data_dir, project_name, commit_id + 'id_mapping.pt')
            id_mapping = torch.load(id_mapping_path, weights_only=False)
            node_texes = []
            for node_attn_idx in range(0, len(graph_attn)):
                try:
                    node_attn = graph_attn[node_attn_idx]
                    node_id = id_mapping[0][node_attn_idx]
                    node_text = decode_from_base64(node_id)
                    node_texes.append(node_text)
                except Exception as e:
                    print(id_mapping)
                    print(id_mapping_path)
                    raise e

            example.commit_id = commit_id
            example.label = label
            example.input_tokens = input_tokens
            line_code_label = commit_line_label[commit_id] if commit_id in commit_line_label else []
            # calculate
            if int(label) == 1 and int(pred) == 1 and '[ADD]' in input_tokens:
                cur_codes = commit2codes[commit2codes['commit_id'] == commit_id]
                cur_labels = idx2label[idx2label['commit_id'] == commit_id]
                line_metric, node_metric_1, combined_metric = deal_with_attns(
                    example, attn,
                    pred, cur_codes,
                    cur_labels, (graph_attn, node_texes), line_code_label, True)
                line_metrics.append(line_metric)
                node_metrics_1.append(node_metric_1)
                # node_metrics.append(node_metric)
                combined_metrics.append(combined_metric)
            commit_idx += 1

        metric_types = [combined_metrics]
        for metrics in metric_types:
            IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []
            for metric in metrics:
                cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, cur_top_5_acc = metric
                IFA.append(cur_IFA)
                top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
                effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
                top_10_acc.append(cur_top_10_acc)
                top_5_acc.append(cur_top_5_acc)

            print(
                'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}, Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
                    round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
                    round(np.mean(top_20_percent_LOC_recall), 4),
                    round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
            )
        return (round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
                round(np.mean(top_20_percent_LOC_recall), 4), round(np.mean(effort_at_20_percent_LOC_recall), 4),
                round(np.mean(IFA), 4)), RF_result

    def calculate_metrics(prods, labels):
        prods = np.array(prods).flatten()
        preds = [1 if p > 0.5 else 0 for p in prods]
        preds = np.array(preds).flatten()
        labels = np.array(labels).flatten()
        print(f"preds {preds}", flush=True)
        print(f"labels {labels}", flush=True)

        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0.0)
        auc = roc_auc_score(labels, prods) if len(set(labels)) > 1 else 0.5
        mcc = matthews_corrcoef(labels, preds)

        return f1, acc, precision, auc, mcc

    def get_recall_at_k_percent_effort(percent_effort, result_df_arg, real_buggy_commits):
        cum_LOC_k_percent = (percent_effort / 100) * result_df_arg.iloc[-1]['cum_LOC']
        buggy_line_k_percent = result_df_arg[result_df_arg['cum_LOC'] <= cum_LOC_k_percent]
        buggy_commit = buggy_line_k_percent[buggy_line_k_percent['label'] == 1]
        recall_k_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

        return recall_k_percent_effort

    def eval_metrics(result_df):
        pred = result_df['defective_commit_pred']
        y_test = result_df['label']

        prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average='binary')  # at threshold = 0.5
        tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
        #     rec = tp/(tp+fn)

        FAR = fp / (fp + tn)  # false alarm rate
        dist_heaven = math.sqrt((pow(1 - rec, 2) + pow(0 - FAR, 2)) / 2.0)  # distance to heaven

        AUC = roc_auc_score(y_test, result_df['defective_commit_prob'])

        result_df['defect_density'] = result_df['defective_commit_prob'] / result_df['LOC']  # predicted defect density
        result_df['actual_defect_density'] = result_df['label'] / result_df['LOC']  # defect density

        result_df = result_df.sort_values(by='defect_density', ascending=False)
        actual_result_df = result_df.sort_values(by='actual_defect_density', ascending=False)
        actual_worst_result_df = result_df.sort_values(by='actual_defect_density', ascending=True)

        result_df['cum_LOC'] = result_df['LOC'].cumsum()
        actual_result_df['cum_LOC'] = actual_result_df['LOC'].cumsum()
        actual_worst_result_df['cum_LOC'] = actual_worst_result_df['LOC'].cumsum()

        real_buggy_commits = result_df[result_df['label'] == 1]

        label_list = list(result_df['label'])

        all_rows = len(label_list)

        # find Recall@20%Effort
        cum_LOC_20_percent = 0.2 * result_df.iloc[-1]['cum_LOC']
        buggy_line_20_percent = result_df[result_df['cum_LOC'] <= cum_LOC_20_percent]
        buggy_commit = buggy_line_20_percent[buggy_line_20_percent['label'] == 1]
        recall_20_percent_effort = len(buggy_commit) / float(len(real_buggy_commits))

        # find Effort@20%Recall
        buggy_20_percent = real_buggy_commits.head(math.ceil(0.2 * len(real_buggy_commits)))
        buggy_20_percent_LOC = buggy_20_percent.iloc[-1]['cum_LOC']
        effort_at_20_percent_LOC_recall = int(buggy_20_percent_LOC) / float(result_df.iloc[-1]['cum_LOC'])

        # find P_opt
        percent_effort_list = []
        predicted_recall_at_percent_effort_list = []
        actual_recall_at_percent_effort_list = []
        actual_worst_recall_at_percent_effort_list = []

        for percent_effort in np.arange(10, 101, 10):
            predicted_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, result_df,
                                                                               real_buggy_commits)
            actual_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort, actual_result_df,
                                                                            real_buggy_commits)
            actual_worst_recall_k_percent_effort = get_recall_at_k_percent_effort(percent_effort,
                                                                                  actual_worst_result_df,
                                                                                  real_buggy_commits)

            percent_effort_list.append(percent_effort / 100)

            predicted_recall_at_percent_effort_list.append(predicted_recall_k_percent_effort)
            actual_recall_at_percent_effort_list.append(actual_recall_k_percent_effort)
            actual_worst_recall_at_percent_effort_list.append(actual_worst_recall_k_percent_effort)

        p_opt = 1 - ((auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                      auc(percent_effort_list, predicted_recall_at_percent_effort_list)) /
                     (auc(percent_effort_list, actual_recall_at_percent_effort_list) -
                      auc(percent_effort_list, actual_worst_recall_at_percent_effort_list)))

        return f1, AUC, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt

    def load_change_metrics_df(data_dir, mode='train'):
        change_metrics = pd.read_pickle(data_dir)
        feature_name = ["ns", "nd", "nf", "entropy", "la", "ld", "lt", "fix", "ndev", "age", "nuc", "exp", "rexp",
                        "sexp"]
        change_metrics = convert_dtype_dataframe(change_metrics, feature_name)

        return change_metrics[['commit_hash'] + feature_name]

    def eval_result(result_path, features_path):
        RF_result = pd.read_csv(result_path, sep='\t')

        RF_result.columns = ['test_commit', 'defective_commit_prob', 'defective_commit_pred', 'label']  # for new result

        test_commit_metrics = load_change_metrics_df(features_path, 'test')[['commit_hash', 'la', 'ld']]
        RF_df = pd.DataFrame()
        RF_df['commit_id'] = RF_result['test_commit']
        RF_df = pd.merge(RF_df, test_commit_metrics, left_on='commit_id', right_on='commit_hash', how='inner')
        RF_df = RF_df.drop('commit_hash', axis=1)
        RF_df['LOC'] = RF_df['la'] + RF_df['ld']
        RF_result = pd.merge(RF_df, RF_result, how='inner', left_on='commit_id', right_on='test_commit')
        f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt = eval_metrics(
            RF_result)
        print(
            'F1: {:.4f}, AUC: {:.4f}, PCI@20%LOC: {:.4f}, Effort@20%Recall: {:.4f}, POpt: {:.4f}'.format(
                f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt))
        return (f1, auc, recall_20_percent_effort, effort_at_20_percent_LOC_recall, p_opt)

    _, train_results = evaluate(model, dataset, train_text_embeddings, line_eval=False)
    _, valid_results = evaluate(model, validation_set, valid_text_embeddings, line_eval=False)
    line_results, test_results = evaluate(model, test_set, test_text_embeddings,line_eval=True)
    all_results = pd.concat([train_results, valid_results, test_results])
    all_results.to_csv(runs_dir + '/' + "all_predictions.csv", sep='\t', index=None, header=['commit_id', 'prod', 'pred', 'label'])

    test_data_file = data_dir + "/test/features_test.pkl"
    commit_results = eval_result(os.path.join(runs_dir, 'test' + "_predictions.csv"), test_data_file)
    return commit_results + line_results


val_results = []
exp_path = '/workspace/s2156631-thesis/results/20241110-180959v4_pdf_cfg_call_graph_14_768_out_no_pos_weight/'
for i in range(0, 10):
    runs_dir = exp_path + str(i)
    results = line_evaluation(runs_dir)
    val_results.append(results)

val_means = np.mean(val_results, axis=0)
val_stds = np.std(val_results, axis=0)

# Number of experiments
n = len(val_results)

# Z-score for 95% confidence interval
z_score = 1.96

# Calculate confidence intervals
val_conf_intervals = z_score * (val_stds / np.sqrt(n))

# Calculate coefficient of variation (CV)
val_cvs = val_stds / val_means

# Display results
metrics = ['f1', 'AUC', 'recall_20_percent_effort', 'effort_at_20_percent_LOC_recall',
           'p_opt', 'top_10_acc_line', 'top_5_acc_line', 'top_20_percent_LOC_recall_line', 'effort_at_20_percent_LOC_recall_line', 'IFA_line']
results_df = pd.DataFrame({
    'Metric': metrics,
    'Mean': val_means,
    'Std': val_stds,
    'CV': val_cvs,
    'CI Lower Bound': val_means - val_conf_intervals,
    'CI Upper Bound': val_means + val_conf_intervals
})

# Save the results to a CSV file
output_file = exp_path + '/results.csv'
results_df.to_csv(output_file, index=False)

# Optionally, print the results to verify
for i, metric in enumerate(metrics):
    print(f"{metric.upper()} - Validation: Mean = {val_means[i]:.4f}, "
          f"Std = {val_stds[i]:.4f}, CV = {val_cvs[i]:.4f}, "
          f"CI = [{val_means[i] - val_conf_intervals[i]:.4f}, {val_means[i] + val_conf_intervals[i]:.4f}]")

print(f"Results saved to {output_file}")
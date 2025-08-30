import os
import random
import re
import sys
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import transformers
from sklearn.metrics import f1_score, accuracy_score, precision_score, roc_auc_score, matthews_corrcoef
from torch.utils.data import WeightedRandomSampler
from torch_geometric import seed_everything
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, get_linear_schedule_with_warmup

from SimplifiedHomoGraphClassifierV2 import HomogeneousGraphSequentialClassifierV2
from EarlyStopping import EarlyStopping
import argparse
import sys

from NewLazyLoadDatasetV4 import NewLazyLoadDatasetV4
from prepare_datasets import prepare_datasets
from prepare_text_embeddings import prepare_text_embeddings

time_exp = datetime.now().strftime("%Y%m%d-%H%M%S")
generic_callgraph_edge = "CALL"
generic_other_edges = "OTHER"
generic_pdg_edge = "PDG"
generic_ast_edge = "AST"
generic_cfg_edge = "CFG"
generic_self_edge = "SELF"

# Load the dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = '/workspace/s2156631-thesis/new_before_after_data/output/'
result_path = '/workspace/s2156631-thesis/results/'
buggy_line_filepath = data_dir + '/changes_complete_buggy_line_level.pkl'

# bert
model_path = "/workspace/s2156631-thesis/data/codebert-base"
bert_tokenizer = AutoTokenizer.from_pretrained(model_path)
special_tokens_dict = {'additional_special_tokens': ["[ADD]", "[DEL]"]}
bert_tokenizer.add_special_tokens(special_tokens_dict)
bert_model = AutoModel.from_pretrained(model_path, output_attentions=True)
bert_model.resize_token_embeddings(len(bert_tokenizer))
bert_model.to(device)

project_list = ['ant-ivy', 'commons-bcel', 'commons-beanutils', 'commons-codec', 'commons-collections',
                'commons-compress',
                'commons-configuration', 'commons-dbcp', 'commons-digester', 'commons-io', 'commons-jcs',
                'commons-lang', 'commons-math',
                'commons-net', 'commons-scxml', 'commons-validator', 'commons-vfs', 'giraph', 'gora', 'opennlp',
                'parquet-mr']

train_dataset, valid_dataset, test_dataset = prepare_datasets(data_dir=data_dir, project_list=project_list, bert_model=bert_model, bert_tokenizer=bert_tokenizer, device=device)


def evaluate(model, dataset, text_embeddings, epoch=None, early_stopping=None, scheduler=None, criterion=None,
             optimizer=None, writer=None, meta=None):
    batch_size = meta['batch_size']
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

    with torch.no_grad():
        for batch in data_loader:
            homogeneous_data, commit_message, code, features_embedding, labels, commits = batch
            # text_embedding = text_embeddings(commit_message, code)
            text_embedding = torch.stack([text_embeddings[commit_id]['cls_embeddings'] for commit_id in commits], dim=0)
            data = {
                'x_dict': homogeneous_data.x.to(device),
                'edge_index': homogeneous_data.edge_index.to(device),
                'batch': homogeneous_data.batch.to(device),
                'text_embedding': text_embedding.to(device),
                'features_embedding': features_embedding.to(device),
                'batch_size': len(labels)
            }
            logits, embeddings, graph_attn_weights = model(data)
            # logits, embeddings, graph_attn_weights = model(data['x_dict'], data['edge_index'], index=data['batch'], batch_size=data['batch_size'])
            all_graph_attns.extend(graph_attn_weights)
            labels = labels.unsqueeze(-1)
            loss = criterion(logits, labels.to(device))
            total_loss += loss.item()
            prods = torch.sigmoid(logits).cpu().detach().numpy()
            all_prods.extend(prods)
            all_labels.extend(labels)
            all_embeddings.extend(embeddings.cpu().detach().numpy())

        end_eval_time = datetime.now()
        val_loss = total_loss / len(data_loader)
        print(f"Evaluation took {end_eval_time - start_eval_time} seconds")
        val_f1, val_acc, val_precision, val_auc, val_mcc = calculate_metrics(all_prods, all_labels)
        if scheduler is not None:
            optimizer.zero_grad()
            scheduler.step(val_f1 + val_auc)
        print(
            f"{dataset.data_type} Loss: {val_loss}, F1: {val_f1}, Accuracy: {val_acc}, Precision: {val_precision}, AUC: {val_auc}, MCC: {val_mcc}")

        if early_stopping is not None:
            early_stopping(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, val_loss=val_loss, val_f1=val_f1, val_mcc=val_mcc, val_auc=val_auc)
        if writer is not None:
            writer.add_scalar(f'Loss/{dataset.data_type}', val_loss, epoch)
            writer.add_scalar(f'F1/{dataset.data_type}', val_f1, epoch)
            writer.add_scalar(f'AUC/{dataset.data_type}', val_auc, epoch)
            writer.add_scalar(f'MCC/{dataset.data_type}', val_mcc, epoch)
            writer.flush()

        return val_f1, val_acc, val_precision, val_auc, val_mcc

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

def experiment_metadata(sample_weights=None, weight_decay=None, no_decay=None, desc=None):
    batch_size = 64
    graph_edges = [generic_ast_edge, generic_callgraph_edge, generic_cfg_edge, generic_pdg_edge, generic_other_edges,
               generic_self_edge]
    manual_features_columns = ['la', 'ld', 'nf', 'ns', 'nd', 'entropy', 'ndev',
                               'lt', 'nuc', 'age', 'exp', 'rexp', 'sexp', 'fix']
    node_feature_dimension = 768 + 128
    out_channels = 128
    in_channels = node_feature_dimension + 1
    pos_weight = torch.tensor(train_dataset.neg_count / train_dataset.pos_count)
    lr = 1e-2
    exp_str = time_exp + (desc if desc is not None else '')
    meta = {
        'project_list': project_list,
        'exp_str': exp_str,
        'result_location': result_location,
        'batch_size': batch_size,
        'node_feature_dimension': node_feature_dimension,
        'out_channels': out_channels,
        'in_channels': in_channels,
        'pos_weight': pos_weight,
        'lr': lr,
        'sample_weights': sample_weights,
        'weight_decay': weight_decay,
        'no_decay': no_decay,
        'manual_features_columns': manual_features_columns,
        'graph_edges': graph_edges,
        'desc': desc,
        'optimizer_grouped_parameters': None,
        'optimizer': None,
        'scheduler': None,
        'criterion': None,
        'model': None,
        'with_text_features': True,
        'with_manual_features': True,
        'with_graph_features': True
    }
    return meta

    
def experiment(exp_idx, mode="train", result_location=None, desc=None):
    # Main training loop
    if mode == "train":
        sample_weights = train_dataset.get_sample_weights().to(device)
        no_decay = ['bias', 'layernorm.weight', 'LayerNorm.weight']
        weight_decay = 1e-4

        meta = experiment_metadata(sample_weights=sample_weights, no_decay=no_decay, weight_decay=weight_decay, desc=desc)
        meta['graph_edges'] = [generic_callgraph_edge, generic_cfg_edge, generic_pdg_edge]
        meta['manual_features_columns'] =  []
        # Initialize model
        # meta['manual_features_columns'] = ['la', 'nf', 'nd', 'entropy', 'current_SM_method_mism_median', 'parent_AST_variabledeclarator', 'current_AST_statementexpression', 'current_SM_class_tnos_stdev', 'current_AST_localvariabledeclaration', 'current_SM_file_loc', 'current_SM_method_loc_max', 'current_SM_method_mims_max', 'current_SM_method_mims_median', 'nuc', 'current_SM_class_nos_median', 'current_SM_method_misei_median', 'current_PMD_rule_type_string and stringbuffer rules', 'current_SM_method_mi_median', 'current_SM_class_loc_stdev', 'current_SM_method_mi_max']
        meta['manual_features_columns'] = ['current_AST_blockstatement', 'delta_AST_memberreference', 'parent_SM_class_nos_median', 'current_SM_method_nii_sum', 'current_SM_method_nos_max', 'current_SM_method_mims_median', 'current_SM_class_nos_median', 'current_SM_class_tlloc_min', 'delta_SM_method_tloc_sum', 'parent_SM_class_nos_stdev', 'delta_AST_referencetype', 'current_AST_localvariabledeclaration', 'current_AST_statementexpression', 'current_SM_method_mims_max', 'parent_AST_memberreference', 'parent_AST_variabledeclarator', 'parent_AST_formalparameter', 'current_PMD_severity_minor', 'current_PMD_arp', 'current_PMD_rule_type_string and stringbuffer rules', 'parent_PMD_vnc', 'current_PMD_gdl', 'current_PMD_adl', 'current_PMD_atret', 'current_SM_class_nlpm_sum', 'current_SM_class_tcloc_max', 'current_PMD_gls', 'current_SM_class_tnos_stdev', 'ld', 'nf', 'ns', 'nd', 'entropy', 'nuc', 'current_SM_method_mism_median', 'current_SM_class_loc_stdev', 'current_SM_method_mi_avg', 'current_SM_method_mi_max', 'current_SM_method_misei_avg', 'current_SM_class_lloc_avg', 'current_SM_method_misei_median', 'current_SM_file_loc', 'current_SM_file_pda', 'current_SM_class_tloc_stdev', 'current_SM_method_loc_median', 'current_SM_method_loc_max', 'current_SM_method_mi_median', 'parent_SM_class_tloc_stdev', 'current_SM_class_lloc_median', 'current_SM_method_misei_min', 'parent_PMD_if', 'la', 'current_PMD_uni', 'parent_PMD_gls', 'current_PMD_acge', 'parent_PMD_gdl', 'current_PMD_fdsbasoc', 'current_SM_class_tcloc_avg', 'delta_SM_method_misei_max', 'current_SM_class_tng_median', 'current_SM_class_lloc_min', 'current_SM_interface_tnpm_median', 'current_SM_interface_ng_sum', 'parent_PMD_fdsbasoc', 'current_SM_interface_npm_median', 'current_PMD_dp', 'current_SM_package_tnm', 'current_AST_typeargument', 'current_AST_methodinvocation', 'current_SM_class_tcloc_median']
        
        model = HomogeneousGraphSequentialClassifierV2(in_channels=meta['in_channels'], out_channels=meta['out_channels'],
                                                       device=device, with_manual_features=meta['with_manual_features'],
                                                       with_text_features=meta['with_text_features'], with_graph_features=meta['with_graph_features'], manual_size=len(meta['manual_features_columns'])).to(device)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=meta['lr'])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=8, min_lr=1e-5,
                                                               factor=0.7)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=meta['pos_weight'].to(device))
        criterion = nn.BCEWithLogitsLoss()
        meta['optimizer_grouped_parameters'] = optimizer_grouped_parameters
        meta['optimizer'] = optimizer
        meta['scheduler'] = scheduler
        meta['criterion'] = criterion
        meta['model'] = model

        result_exp_str = result_path + '/' + meta['exp_str']

        result_location = result_exp_str + "/" + str(exp_idx)
        os.makedirs(result_location, exist_ok=True)
        print(meta, flush=True)
        sys.stdout = open(result_location + '/run.log', 'w')
        print(meta, flush=True)
        early_stopping = EarlyStopping(exp=result_location, meta=meta)

        writer = SummaryWriter(log_dir=result_exp_str + '/tensorboard/' + str(exp_idx),
                               filename_suffix= meta['exp_str'])  # Directory for TensorBoard logs

        train_dataset.set_config(meta['manual_features_columns'], meta['graph_edges'])
        valid_dataset.set_config(meta['manual_features_columns'], meta['graph_edges'])
        test_dataset.set_config(meta['manual_features_columns'], meta['graph_edges'])

        train_text_embeddings, valid_text_embeddings, test_text_embeddings = prepare_text_embeddings(train_dataset, valid_dataset, test_dataset, bert_model=bert_model, bert_tokenizer=bert_tokenizer, device=device)
        print(len(train_text_embeddings))
        print(len(train_dataset))
        print(test_text_embeddings.keys())

        for epoch in range(200):
            model.train()
            print(f"Epoch {epoch} start_time {datetime.now()}", flush=True)
            total_loss = 0

            # Create sampler for balanced sampling
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
            data_loader = DataLoader(train_dataset, batch_size=meta['batch_size'], sampler=sampler)

            # Initialize time tracking
            epoch_start_time = datetime.now()
            epoch_embeddings = []
            epoch_labels = []

            for batch in data_loader:
                homogeneous_data, commit_message, code, features_embedding, labels, commits = batch
                # text_embedding = text_embeddings(commit_message, code)
                text_embedding = torch.stack(
                    [train_text_embeddings[commit_id]['cls_embeddings'] for commit_id in commits], dim=0)
                # print(homogeneous_data, flush=True)
                data = {
                    'x_dict': homogeneous_data.x.to(device),
                    'edge_index': homogeneous_data.edge_index.to(device),
                    'batch': homogeneous_data.batch.to(device),
                    'text_embedding': text_embedding.to(device),
                    'features_embedding': features_embedding.to(device),
                    'batch_size': len(labels)
                }
                logits, graph_embeds, graph_attn_weights = model(data)
                # logits, graph_embeds, graph_attn_weights = model(data['x_dict'], data['edge_index'], index=data['batch'], batch_size=data['batch_size'])
                labels = labels.unsqueeze(-1)
                print(f"logits {torch.reshape(logits, (-1,))}", flush=True)
                print(f"labels {torch.reshape(labels, (-1,))}", flush=True)
                loss = criterion(logits, labels.to(device))
                epoch_embeddings.extend(graph_embeds.cpu().detach().numpy())
                epoch_labels.extend(labels.cpu().detach().numpy())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(data_loader)

            # Log training loss to TensorBoard
            writer.add_scalar('Loss/train', avg_loss, epoch)
            # writer.add_embedding(torch.tensor(np.array(epoch_embeddings)), metadata=epoch_labels, tag=f'embeddings_{epoch}', global_step=epoch)
            writer.add_scalar('LearningRate/train', torch.tensor(scheduler.get_last_lr()[0]), epoch)
            writer.flush()
            print(f"Epoch {epoch}, Loss: {avg_loss}")

            # Validation phase
            evaluate(model, valid_dataset, valid_text_embeddings,epoch, early_stopping, scheduler=scheduler,
                                                                        optimizer=optimizer, criterion=criterion,
                                                                        writer=writer, meta=meta)
            evaluate(model, test_dataset, test_text_embeddings,epoch, criterion=criterion,
                                                                        writer=writer, meta=meta)

            # Calculate elapsed time and estimate remaining time
            elapsed_time = datetime.now() - epoch_start_time
            estimated_remaining = (elapsed_time / (epoch + 1)) * (200 - (epoch + 1))
            print(f"Elapsed time: {elapsed_time}, Estimated remaining time: {estimated_remaining}")
            if early_stopping is not None:
                if early_stopping.early_stop:
                    print(f"early stopping with stage {early_stopping}")
                    break

        # Close TensorBoard writer
        writer.close()
        model.load_state_dict(early_stopping.best_model)
        val_f1, val_acc, val_precision, val_auc, val_mcc = evaluate(model, valid_dataset, valid_text_embeddings,
                                                                    early_stopping.best_epoch, criterion=criterion,
                                                                    meta=meta)
        test_f1, test_acc, test_precision, test_auc, test_mcc = evaluate(model, test_dataset, test_text_embeddings,
                                                                         early_stopping.best_epoch, criterion=criterion,
                                                                         meta=meta)

        return (result_location, (val_f1, val_acc, val_precision, val_auc, val_mcc),
                (test_f1, test_acc, test_precision, test_auc, test_mcc))
    else:
        return evaluation_experiment(exp_idx, result_location)

def evaluation_experiment(experiment_idx, result_location=None):
    runs_dir = result_location + "/" + str(experiment_idx)
    model_path = '{}/{}'.format(runs_dir, 'model.bin')
    meta_path = '{}/{}'.format(runs_dir, 'meta.txt')
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
    project_list = meta['project_list']
    print(project_list)
    model = meta['model']
    model_state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(model_state_dict['model_state_dict'])
    val_f1, val_acc, val_precision, val_auc, val_mcc = evaluate(model, valid_dataset, valid_text_embeddings, 0,
                                                                criterion=criterion, meta=meta)
    test_f1, test_acc, test_precision, test_auc, test_mcc = evaluate(model, test_dataset, test_text_embeddings, 0,
                                                                     criterion=criterion, meta=meta)

    return (result_location, (val_f1, val_acc, val_precision, val_auc, val_mcc),
            (test_f1, test_acc, test_precision, test_auc, test_mcc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='mode', default='train')
    parser.add_argument('--result_location', type=str, help='result location', default='')
    parser.add_argument('--desc', type=str, help='describe the experiments', default='')
    args = parser.parse_args()
    mode = args.mode
    desc = args.desc
    result_location = args.result_location

    experiments = []
    for i in range(0, 10):
        exp_result = experiment(i, mode=mode, result_location=result_location, desc=desc)
        experiments.append(exp_result)

    val_results = np.array([exp[1] for exp in experiments])
    test_results = np.array([exp[2] for exp in experiments])

    # Calculate mean and standard deviation
    val_means = np.mean(val_results, axis=0)
    val_stds = np.std(val_results, axis=0)

    test_means = np.mean(test_results, axis=0)
    test_stds = np.std(test_results, axis=0)

    # Number of experiments
    n = val_results.shape[0]

    # Z-score for 95% confidence interval
    z_score = 1.96

    # Calculate confidence intervals
    val_conf_intervals = z_score * (val_stds / np.sqrt(n))
    test_conf_intervals = z_score * (test_stds / np.sqrt(n))

    # Calculate coefficient of variation (CV)
    val_cvs = val_stds / val_means
    test_cvs = test_stds / test_means

    # Display results
    metrics = ['f1', 'accuracy', 'precision', 'AUC', 'MCC']
    for i, metric in enumerate(metrics):
        print(
            f"{metric.upper()} - Validation: Mean = {val_means[i]:.4f}, Std = {val_stds[i]:.4f}, CV = {val_cvs[i]:.4f}, CI = [{val_means[i] - val_conf_intervals[i]:.4f}, {val_means[i] + val_conf_intervals[i]:.4f}]")
        print(
            f"{metric.upper()} - Test: Mean = {test_means[i]:.4f}, Std = {test_stds[i]:.4f}, CV = {test_cvs[i]:.4f}, CI = [{test_means[i] - test_conf_intervals[i]:.4f}, {test_means[i] + test_conf_intervals[i]:.4f}]")
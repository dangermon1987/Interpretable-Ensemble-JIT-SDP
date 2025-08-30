import os

import torch
from torch_geometric.loader import DataLoader


def text_embeddings(commit_msgs, codes, full=False, bert_tokenizer=None, bert_model=None, device=None):
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
            attns = attns.mean(axis=0)[1][begin_pos:end_pos]
            batch_attns.append(attns)
        return [{
            'cls_embeddings': cls_embeddings[i],  # shape: (batch_size, hidden_size)
            'attn_weights': attn,  # shape: (batch_size, num_heads, seq_length, seq_length)
            'tokens': token  # List of token lists for each input
        } for i, attn, token in zip(range(0, len(all_tokens)), batch_attns, all_tokens)]
    else:
        return cls_embeddings

def prepare_text_embeddings(train_dataset, valid_dataset, test_dataset, bert_model=None, bert_tokenizer=None, device=None):
    train_text_embeddings = {}
    train_text_embeddings_path = "train_text_embeddings_v4.pkl"
    if not os.path.exists(train_text_embeddings_path):
        train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        for batch in train_data_loader:
            homogeneous_data, commit_message, code, features_embedding, labels, commits = batch
            text_embedding = text_embeddings(commit_message, code, full=True, bert_model=bert_model, bert_tokenizer=bert_tokenizer, device=device)
            index = 0
            for commit_id in commits:
                train_text_embeddings[commit_id] = text_embedding[index]
                index += 1
        torch.save(train_text_embeddings, "train_text_embeddings_v4.pkl")
    else:
        train_text_embeddings = torch.load(train_text_embeddings_path, weights_only=False)

    valid_text_embeddings = {}
    valid_text_embeddings_path = "valid_text_embeddings_v4.pkl"
    if not os.path.exists(valid_text_embeddings_path):
        valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
        for batch in valid_data_loader:
            homogeneous_data, commit_message, code, features_embedding, labels, commits = batch
            text_embedding = text_embeddings(commit_message, code, full=True, bert_model=bert_model, bert_tokenizer=bert_tokenizer, device=device)
            index = 0
            for commit_id in commits:
                valid_text_embeddings[commit_id] = text_embedding[index]
                index += 1
        torch.save(valid_text_embeddings, "valid_text_embeddings_v4.pkl")
    else:
        valid_text_embeddings = torch.load(valid_text_embeddings_path, weights_only=False)

    test_text_embeddings = {}
    test_text_embeddings_path = "test_text_embeddings_v4.pkl"
    if not os.path.exists(test_text_embeddings_path):
        test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        for batch in test_data_loader:
            homogeneous_data, commit_message, code, features_embedding, labels, commits = batch
            full = True
            text_embedding = text_embeddings(commit_message, code, full=full, bert_model=bert_model, bert_tokenizer=bert_tokenizer, device=device)
            index = 0
            for commit_id in commits:
                test_text_embeddings[commit_id] = text_embedding[index]
                index += 1
        torch.save(test_text_embeddings, "test_text_embeddings_v4.pkl")
    else:
        test_text_embeddings = torch.load(test_text_embeddings_path, weights_only=False)

    return train_text_embeddings, valid_text_embeddings, test_text_embeddings

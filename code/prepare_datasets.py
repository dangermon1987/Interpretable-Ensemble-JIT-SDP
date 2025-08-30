import os
import pickle

import numpy as np
import pandas as pd

from NewLazyLoadDatasetV4 import NewLazyLoadDatasetV4


def prepare_datasets(data_dir=None, project_list=None, bert_model=None, bert_tokenizer=None, device=None):
    # Load datasets
    train_dataset_pkl = 'train_dataset_v4' +'.pkl'
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
        dataset = NewLazyLoadDatasetV4(data_dir=data_dir, projects=project_list, data_type='train', merge=True,
                                       old=False, device=device, tokenizer=bert_tokenizer, model=bert_model,
                                       changes_data=train_changes_data)
        with open(train_dataset_pkl, 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"train_set {dataset.pos_count} vs {dataset.neg_count}", flush=True)

    validation_set_pkl = 'valid_dataset_v4' + '.pkl'
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
        validation_set = NewLazyLoadDatasetV4(data_dir=data_dir, projects=project_list, data_type='valid', merge=True,
                                              device=device,
                                              old=False, tokenizer=bert_tokenizer, model=bert_model,
                                              changes_data=valid_changes_data, scaler=dataset.scaler)
        with open(validation_set_pkl, 'wb') as handle:
            pickle.dump(validation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"valid_set {validation_set.pos_count} vs {validation_set.neg_count}", flush=True)
    test_set_pkl = 'test_dataset_v4' + '.pkl'

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
        test_set = NewLazyLoadDatasetV4(data_dir=data_dir, projects=project_list, data_type='test', merge=True, old=False,
                                        device=device, tokenizer=bert_tokenizer, model=bert_model,
                                        changes_data=test_changes_data, scaler=dataset.scaler)
        with open(test_set_pkl, 'wb') as handle:
            pickle.dump(test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"test_set {test_set.pos_count} vs {test_set.neg_count}", flush=True)

    return dataset, validation_set, test_set

import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import time
import datetime
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from sklearn.metrics import f1_score as sk_f1_score
from dataclasses import dataclass


def collate_dicts(samples, pad_id=0, return_lens=True):
    '''
        samples: list of dicts
    '''
    batch = {}

    if return_lens:
        sample_lens = [len(sample['input_ids']) for sample in samples]
        batch['original_lens'] = torch.tensor(sample_lens)

    for key in samples[0].keys():
        padding_value = 0
        if key == 'input_ids':
            padding_value = pad_id

        data = [sample[key] for sample in samples]
        batch[key] = pad_sequence(data, batch_first=True, padding_value=padding_value)

    return batch


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unpack(batches, original_lens_batches):
    unpacked = []
    for batch, original_lens in zip(batches, original_lens_batches):
        for row, length in zip(batch.tolist(), original_lens.tolist()):
            unpacked.append(row[:length])
    return unpacked


def compute_metrics(labels, preds, original_lens, int2label):
    '''
        labels, result, original_lens - each is a list of batched tensors of shape (batch_len, seq_len)
    '''
    # we are only intrested in metrics for non-pad tokens
    # so we must filter them out first

    label_ids = unpack(labels, original_lens) # list of (list of true labels for doc)
    preds_ids = unpack(preds, original_lens)

    labels = [list(map(lambda x: int2label[x], doc)) for doc in label_ids]
    preds = [list(map(lambda x: int2label[x], doc)) for doc in preds_ids]

    f = lambda token: token if 'ADR' in token else 'O'
    binary_labels = [list(map(f, doc)) for doc in labels]
    binary_preds = [list(map(f, doc)) for doc in labels]

    modes = ['default', 'strict']
    types = ['multiclass', 'binary']

    res = {}

    res['accuracy'] = accuracy_score(labels, preds) # for the sake of visibility, compute only 1 accuracy
    
    for type_ in types:
        (y_true, y_pred) = (labels, preds) if (type_ != 'binary') else (binary_labels, binary_preds)
        for mode in modes:
            f1 = f1_score(y_true, y_pred, mode=mode, average='macro')
            res['_'.join(['f1', type_, mode])] = f1

        y_true = [item for sublist in y_true for item in sublist]  # flatten
        y_pred = [item for sublist in y_pred for item in sublist]
        res['_'.join(['f1', 'token', type_])] = sk_f1_score(y_true, y_pred, average='macro')

    return res
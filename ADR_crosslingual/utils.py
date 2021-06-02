import os
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
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


def make_brat_pair(documents_iter, fold_dir):
    entities_list = []
    texts_list = []
    offset = 0
    e_id = 1
    for document in documents_iter:
        for entity in document.entities.values():
            s = "T"+str(e_id)+'\t'
            s += entity.type + " "
            s += str(offset + entity.start) + " "
            s += str(offset + entity.end) + '\t'
            s += entity.text
            entities_list.append(s)
            e_id += 1
        
        offset += len(document.text) + 1
        texts_list.append(document.text)
        

    doc_id = documents_iter[0].doc_id

    ann_path = os.path.join(fold_dir, "doc"+str(doc_id)+".ann")
    with open(ann_path, 'w+') as ann_file:
        ann_file.write("\n".join(entities_list) + "\n")

    txt_path = os.path.join(fold_dir, "doc"+str(doc_id)+".txt")
    with open(txt_path, 'w+') as txt_file:
        txt_file.write("\n".join(texts_list) + "\n")


def map_labels(dataset, mapper, label2int=None):
    for idx, doc_labels in enumerate(dataset.labels):
        dataset.labels[idx] = list(map(lambda s: mapper.get(s, 'UNK'), doc_labels))
    dataset.set_label_info(label2int)


def get_cur_labeled_loaders(cur_labeled_set, batch_size, rudrec_labeled_set, collate_teacher, collate_student):
    N = len(cur_labeled_set)
    batch = []
    for idx in range(N, min(N+batch_size, len(rudrec_labeled_set))):
        batch.append(rudrec_labeled_set[idx])
    cur_labeled_set.extend(batch)

    student_loader = DataLoader(cur_labeled_set, batch_size=batch_size, collate_fn=collate_student, pin_memory=True)
    teacher_loader = DataLoader(batch, batch_size=batch_size, collate_fn=collate_teacher, pin_memory=True)

    return teacher_loader, student_loader


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
    binary_preds = [list(map(f, doc)) for doc in preds]

    modes = ['strict']
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
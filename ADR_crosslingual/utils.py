import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import time
import datetime
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score
from dataclasses import dataclass


def collate_dicts(samples, pad_id=0):
    '''
        samples: list of dicts
    '''
    batch = {}

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
        for row, len in zip(batch.tolist(), original_lens):
            unpacked.append(row[:len])
    return unpacked


def compute_metrics(labels, preds, original_lens, int2label):
    '''
        labels, result, original_lens - each is a list of batched tensors of shape (batch_len, seq_len)
    '''
    # we are only intrested in metrics for non-pad tokens
    # so we must filter them out first

    labels = [list(map(lambda x: int2label[x], doc)) for doc in unpack(labels, original_lens)]
    preds = [list(map(lambda x: int2label[x], doc)) for doc in unpack(preds, original_lens)]

    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
    }


@dataclass
class ExperimentConfig:
    learning_rate: float = 2e-5
    base_model_type: str = 'bert'
    pretrained_name: str = 'bert-base-multilingual-cased'
    n_few_shot_samples: int = 0
    seed: int = None
    
    def get_tb_info():
        return "_".join([
                        "arch="+base_model_type,
                        "lr="+str(learning_rate),
                        "n_few_shot="+n_few_shot,
                ])

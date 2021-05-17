import torch
from torch.nn.utils.rnn import pad_sequence

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
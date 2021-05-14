import os
from glob import glob
import re
import sys
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from operator import itemgetter, attrgetter
from NLPDatasetIO.dataset import Dataset



class CADECDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_dir, split=None):
        '''
          corpus_dir: path, must contain corresponding .txt and .ann files
          split: 'train', 'dev' or 'test'
        '''
        if split is None:  # infer from path
            split = corpus_dir.split('/')[-1]
        self.split = split

        self.documents = Dataset(location=corpus_dir, format='brat', split=self.split).documents


    def __len__(self):
        return len(self.documents)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        document = self.documents[idx]


        'input_ids'      #
        'token_type_ids' #all ones
        'attention_mask' #zero for padding tokens
        '''
            item = {key: torch.tensor(val) for key, val in self.encoded_texts[idx].items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        '''


class CadecDataset(Dataset):





    def __init__(self, corpus_dir):
        raise NotImplementedError


    def __getitem__(self, idx):
        if torch.is_tensor(index):
            index = index.tolist(index)

        raise NotImplementedError


    def __len__(self):

        raise NotImplementedError






class PsytarDataset(Dataset):
    def __init__(self, corpus_dir):
        raise NotImplementedError


    def __getitem__(self, idx):
        if torch.is_tensor(index):
            index = index.tolist(index)

        raise NotImplementedError


    def __len__(self):

        raise NotImplementedError

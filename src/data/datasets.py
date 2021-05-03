import os
from glob import glob
import re
import sys
import torch
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class CadecDataset(Dataset):
    def __init__(self, corpus_dir):
        raise NotImplementedError


    def __getitem__(self, idx):
        if torch.is_tensor(index):
            index = index.tolist(index)

        raise NotImplementedError


    def __len__(self):

        raise NotImplementedError

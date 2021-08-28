import numpy as np
import os
import sys
from glob import glob
import torch
from torch.utils.data import DataLoader
import gdown
import yaml

import NLPDatasetIO
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

from ADR_crosslingual.data.datasets import BratDataset, JsonDataset
from ADR_crosslingual.data.splitter import BratSplitter
from ADR_crosslingual.data.convert import webanno2brat
from ADR_crosslingual.data.make_datasets import make_datasets
import time
from copy import deepcopy
from ADR_crosslingual.utils import (
    format_time, set_seed, unpack, compute_metrics,
    make_brat_pair, map_labels, get_cur_labeled_loaders,
)

from ADR_crosslingual.configs import TrainConfig, SamplerConfig, ExperimentConfig
from transformers import (
    BertTokenizer, BertConfig
)

def stats_printer(exp_config):

    cadec_tuple, psytar_tuple, rudrec_tuple, joined_tuple, big_unlabeled_set, rudrec_big_labeled_set = make_datasets(exp_config)

    rudrec_big_labeled_set.label2int = big_unlabeled_set.label2int
    rudrec_big_labeled_set.int2label = big_unlabeled_set.int2label
    rudrec_big_labeled_set.num_labels = big_unlabeled_set.num_labels


    (cadec_train_set, cadec_test_set) = cadec_tuple 
    (psytar_train_set, psytar_test_set) = psytar_tuple 
    (rudrec_labeled_set, rudrec_test_set, rudrec_unlabeled_set) = rudrec_tuple 
    (joined_train_set, joined_test_set) = joined_tuple

    print("cadec train:")
    cadec_train_set.print_stats()

    print("cadec test:")
    cadec_test_set.print_stats()

    print("psytar train:")
    psytar_train_set.print_stats()

    print("psytar test:")
    psytar_test_set.print_stats()

    print("rudrec labeled:")
    rudrec_labeled_set.print_stats()

    print("rudrec test:")
    rudrec_test_set.print_stats()

    print("rudrec unlabeled:")
    rudrec_unlabeled_set.print_stats()

    print("joined train set:")
    joined_train_set.print_stats()

    print("joined test set:")
    joined_test_set.print_stats()
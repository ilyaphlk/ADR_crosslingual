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
from ADR_crosslingual.utils import collate_dicts as collate_dicts_
from torch.utils.tensorboard import SummaryWriter
from ADR_crosslingual.models.single_model import BertTokenClassifier, XLMTokenClassifier
from ADR_crosslingual.trainers.trainer import train_model
from ADR_crosslingual.trainers.trainer import eval_model
from ADR_crosslingual.configs import TrainConfig, SamplerConfig, ExperimentConfig
from transformers import (
    BertTokenizer, BertConfig, BertPreTrainedModel, BertModel,
    XLMTokenizer, XLMConfig, XLMPreTrainedModel, XLMModel,
    AdamW,
)

from ADR_crosslingual.trainers.samplers import (
    BaseUncertaintySampler,
    MarginOfConfidenceSampler,
    BALDSampler,
    VarianceSampler,
    EntropySampler,
    RandomSampler
)


def read_yaml_config(path_to_yaml):
    with open(path_to_yaml) as cfg_yaml:
        cfg = yaml.load(cfg_yaml, Loader=yaml.FullLoader)
        t_cfg = cfg['teacher_config']
        st_cfg = cfg['student_config']
        spl_cfg = cfg['sampler_config']
        exp_cfg = cfg['exp_config']

    teacher_config = TrainConfig(
        model_type={
            'tokenizer': eval(t_cfg['model_type']['tokenizer']),
            'config': eval(t_cfg['model_type']['config']),
            'model': eval(t_cfg['model_type']['model']),
            'subword_prefix': t_cfg['model_type'].get('subword_prefix', None),
            'subword_suffix': t_cfg['model_type'].get('subword_suffix', None),
        },
        model_checkpoint=t_cfg['model_checkpoint'],
        optimizer_class=eval(t_cfg['optimizer_class']),
        optimizer_kwargs={'lr':float(t_cfg['optimizer_kwargs']['lr']),
                          'eps':float(t_cfg['optimizer_kwargs']['eps']),
                          'weight_decay':float(t_cfg['optimizer_kwargs'].get('weight_decay', 0))},
        train_batch_sz=t_cfg['train_batch_sz'],
        test_batch_sz=t_cfg['test_batch_sz'],
        epochs=t_cfg['epochs'],
        L2_coef=float(t_cfg['L2_coef']),
    )

    student_config = TrainConfig(
        model_type={
            'tokenizer': eval(st_cfg['model_type']['tokenizer']),
            'config': eval(st_cfg['model_type']['config']),
            'model': eval(st_cfg['model_type']['model']),
            'subword_prefix': st_cfg['model_type'].get('subword_prefix', None),
            'subword_suffix': st_cfg['model_type'].get('subword_suffix', None)
        },
        model_checkpoint=st_cfg['model_checkpoint'],
        optimizer_class=eval(st_cfg['optimizer_class']),
        optimizer_kwargs={'lr':float(st_cfg['optimizer_kwargs']['lr']),
                          'eps':float(st_cfg['optimizer_kwargs']['eps']),
                          'weight_decay':float(st_cfg['optimizer_kwargs'].get('weight_decay', 0))},
        train_batch_sz=st_cfg['train_batch_sz'],
        test_batch_sz=st_cfg['test_batch_sz'],
        epochs=st_cfg['epochs']
    )

    sampler_config = SamplerConfig(
        sampler_class=eval(spl_cfg['sampler_class']),#BALDSampler,#RandomSampler,
        sampler_kwargs={'strategy':spl_cfg['sampler_kwargs']['strategy'],
                        'n_samples_out':spl_cfg['sampler_kwargs'].get('n_samples_out', student_config.train_batch_sz),
                        'scoring_batch_sz':spl_cfg['sampler_kwargs'].get('scoring_batch_sz', 1),
                        'averaging_share':spl_cfg['sampler_kwargs'].get('averaging_share', None),
                        'return_vars':spl_cfg['sampler_kwargs'].get('return_vars', False),
                        },
        n_samples_in= spl_cfg['n_samples_in'],
        
    )

    if 'n_forward_passes' in spl_cfg['sampler_kwargs']:
        sampler_config.sampler_kwargs['n_forward_passes'] = spl_cfg['sampler_kwargs']['n_forward_passes']

    exp_config = ExperimentConfig(
        teacher_config=teacher_config,
        student_config=student_config,
        sampler_config=sampler_config,
        n_few_shot=exp_cfg['n_few_shot'],
        experiment_name=exp_cfg['experiment_name'],
        seed=exp_cfg['seed'],
        teacher_set=exp_cfg['teacher_set'],
        student_set=exp_cfg.get('student_set', 'small'),
        classification_type=exp_cfg.get('classification_type', 'multiclass'),
        init_with_teacher=exp_cfg.get('init_with_teacher', False),
        big_set_sample_cnt=exp_cfg.get('big_set_sample_cnt', 0),
        to_sentences=exp_cfg.get('to_sentences', False)
    )

    
    common_tokenize=exp_cfg.get('common_tokenize', None)
    if common_tokenize is not None:
        exp_config.common_tokenize = eval(common_tokenize)

    if exp_config.init_with_teacher:
        assert teacher_config.model_type['model'] == student_config.model_type['model']

    return exp_config



def stats_printer(path_to_yaml):

    exp_config = read_yaml_config(path_to_yaml)

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
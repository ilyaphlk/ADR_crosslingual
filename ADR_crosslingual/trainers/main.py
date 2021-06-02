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

from ADR_crosslingual.data.datasets import BratDataset
from ADR_crosslingual.data.splitter import BratSplitter
from ADR_crosslingual.data.convert import webanno2brat
import time
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
                          'eps':float(t_cfg['optimizer_kwargs']['eps'])},
        train_batch_sz=t_cfg['train_batch_sz'],
        test_batch_sz=t_cfg['test_batch_sz'],
        epochs=t_cfg['epochs']
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
                          'eps':float(st_cfg['optimizer_kwargs']['eps'])},
        train_batch_sz=st_cfg['train_batch_sz'],
        test_batch_sz=st_cfg['test_batch_sz'],
        epochs=st_cfg['epochs']
    )

    sampler_config = SamplerConfig(
        sampler_class=eval(spl_cfg['sampler_class']),#BALDSampler,#RandomSampler,
        sampler_kwargs={'strategy':spl_cfg['sampler_kwargs']['strategy'],
                        'n_samples_out':spl_cfg['sampler_kwargs'].get('n_samples_out', student_config.train_batch_sz),},
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
        classification_type=exp_cfg.get('classification_type', 'multiclass')
    )

    return exp_config



def make_cadec(folds_dir, exp_config):
    teacher_config = exp_config.teacher_config
    cadec_dir = './cadec'
    cadec_splitter = BratSplitter(cadec_dir+"/text", cadec_dir+"/original", folds_dir, 0.89, 0.09, dev_share=0.0,
                             name_postfix='_cadec', random_state=exp_config.seed, shuffle=True)

    cadec_splitter.split()

    # TODO check common_tokenize
    teacher_tokenizer = teacher_config.model_type['tokenizer'].from_pretrained(teacher_config.model_checkpoint)
    bratlike_dict = {'format':'brat',
                     'tokenize':teacher_tokenizer.tokenize} # todo get tokenize either from common_tokenize or from teacher
    if exp_config.common_tokenize is not None:
        bratlike_dict['tokenize'] = exp_config.common_tokenize

    bratlike_dict['subword_prefix'] = teacher_config.model_type.get('subword_prefix', None)
    bratlike_dict['subword_suffix'] = teacher_config.model_type.get('subword_suffix', None)


    cadec_train_set = BratDataset(folds_dir+"/train"+cadec_splitter.name_postfix, "train", teacher_tokenizer,
                         kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True)

    cadec_test_set = BratDataset(folds_dir+"/test"+cadec_splitter.name_postfix, "test", teacher_tokenizer,
                            label2int=cadec_train_set.label2int,
                            kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True)

    for idx, elem in enumerate(cadec_train_set):
        s = elem['input_ids'].shape
        for key, t in elem.items():
            assert t.shape == s, f"idx = {idx}, mismatch in tensor shapes"


    return cadec_train_set, cadec_test_set


def make_psytar(folds_dir, exp_config, batched=True):
    teacher_config = exp_config.teacher_config
    teacher_tokenizer = teacher_config.model_type['tokenizer'].from_pretrained(teacher_config.model_checkpoint)
    psytar_folds_dir = '/content/ADR_crosslingual/psytarpreprocessor/data/all/'
    conlllike_dict = {'format':'conll', 'sep':'\t'}
    conll_train_docs = NLPDatasetIO.dataset.Dataset(location=psytar_folds_dir+'train.txt',
                                   split='train', **conlllike_dict).documents
    conll_test_docs = NLPDatasetIO.dataset.Dataset(location=psytar_folds_dir+'test.txt',
                                       split='test', **conlllike_dict).documents

    for doc in conll_train_docs:
        make_brat_pair([doc], os.path.join(folds_dir, 'psytar_train'))

    for doc in conll_test_docs:
        make_brat_pair([doc], os.path.join(folds_dir, 'psytar_test'))

    conll_batching_sz = 4

    for idx in range(0, len(conll_train_docs), conll_batching_sz):
        conll_batch = conll_train_docs[idx:idx+conll_batching_sz]
        make_brat_pair(conll_batch, os.path.join(folds_dir, 'psytar_train_batched'))

    for idx in range(0, len(conll_test_docs), conll_batching_sz):
        conll_batch = conll_test_docs[idx:idx+conll_batching_sz]
        make_brat_pair(conll_batch, os.path.join(folds_dir, 'psytar_test_batched'))

    train_fold_dir, test_fold_dir = (os.path.join(folds_dir, 'psytar_train'),
                                     os.path.join(folds_dir, 'psytar_test'))
    if batched:
        train_fold_dir += '_batched'
        test_fold_dir += '_batched'

    teacher_tokenizer = teacher_config.model_type['tokenizer'].from_pretrained(teacher_config.model_checkpoint)
    bratlike_dict = {'format':'brat',
                     'tokenize':teacher_tokenizer.tokenize} # todo get tokenize either from common_tokenize or from teacher
    if exp_config.common_tokenize is not None:
        bratlike_dict['tokenize'] = exp_config.common_tokenize
    bratlike_dict['subword_prefix'] = teacher_config.model_type.get('subword_prefix', None)
    bratlike_dict['subword_suffix'] = teacher_config.model_type.get('subword_suffix', None)
    
    psytar_train_set = BratDataset(train_fold_dir, "train", teacher_tokenizer,
                         kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True)

    psytar_test_set = BratDataset(test_fold_dir, "test", teacher_tokenizer,
                             label2int=psytar_train_set.label2int,
                             kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True)

    for idx, elem in enumerate(psytar_test_set):
        s = elem['input_ids'].shape
        for key, t in elem.items():
            assert t.shape == s, f"idx = {idx}, mismatch in tensor shapes"

    return psytar_train_set, psytar_test_set


def make_rudrec(folds_dir, exp_config):
    student_config = exp_config.student_config
    rudrec_brat_dir = './rudrec_brat'
    rudrec_size = webanno2brat('./rudrec_labeled', rudrec_brat_dir)

    rudrec_folds_dir = folds_dir

    train_share = exp_config.n_few_shot / rudrec_size

    # we will be using dev fold as our (artificially) unlabeled data - hence its larger share
    rudrec_splitter = BratSplitter(rudrec_brat_dir+"/text", rudrec_brat_dir+"/annotation", rudrec_folds_dir,
                                   train_share, 0.1, 0.8,
                                    name_postfix='_rudrec', random_state=exp_config.seed, shuffle=True)

    rudrec_splitter.split()
    # TODO check common_tokenize
    student_tokenizer = student_config.model_type['tokenizer'].from_pretrained(student_config.model_checkpoint)
    bratlike_dict = {'format':'brat',
                     'tokenize':student_tokenizer.tokenize}
    if exp_config.common_tokenize is not None:
        bratlike_dict['tokenize'] = exp_config.common_tokenize
    bratlike_dict['subword_prefix'] = student_config.model_type.get('subword_prefix', None)
    bratlike_dict['subword_suffix'] = student_config.model_type.get('subword_suffix', None)

    rudrec_labeled_set = BratDataset(rudrec_folds_dir+"/train"+rudrec_splitter.name_postfix, "train", student_tokenizer,
                        kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True)

    rudrec_test_set = BratDataset(rudrec_folds_dir+"/test"+rudrec_splitter.name_postfix, "test", student_tokenizer,
                        label2int=rudrec_labeled_set.label2int,
                        kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True)

    rudrec_unlabeled_set = BratDataset(rudrec_folds_dir+"/dev"+rudrec_splitter.name_postfix, "dev", student_tokenizer,
                       labeled=False,
                       kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True)

    for idx, elem in enumerate(rudrec_test_set):
        s = elem['input_ids'].shape
        for key, t in elem.items():
            assert t.shape == s, f"idx = {idx}, mismatch in tensor shapes"

    return rudrec_labeled_set, rudrec_test_set, rudrec_unlabeled_set


def make_mappers(exp_config):
    if exp_config.classification_type == 'binary':
        raise NotImplementedError
    else:
        rudrec_to_cadec = {
            'B-ADR':'B-ADR', 'I-ADR':'I-ADR',
            'B-Finding':'B-Finding', 'I-Finding':'I-Finding',
            'B-DI':'B-Disease','I-DI':'I-Disease',
            'B-Drugname':'B-Drug','I-Drugname':'I-Drug',
            'B-Drugform':'O','I-Drugform':'O',
            'B-Drugclass':'O','I-Drugclass':'O',
            'O':'O',
        }

        psytar_to_cadec = {
            'B-ADR':'B-ADR','I-ADR':'I-ADR',
            'B-WD':'B-ADR','I-WD':'I-ADR',
            'B-EF':'O','I-EF':'O',
            'B-INF':'O','I-INF':'O',
            'B-SSI':'B-Symptom','I-SSI':'I-Symptom',
            'B-DI':'B-Disease','I-DI':'I-Disease',
            'O':'O',
        }

    return rudrec_to_cadec, psytar_to_cadec



def unify_data(rudrec_to_cadec, psytar_to_cadec, cadec_train_set,
               rudrec_labeled_set, rudrec_test_set, rudrec_unlabeled_set,
               psytar_train_set, psytar_test_set):

    map_labels(rudrec_labeled_set, rudrec_to_cadec, cadec_train_set.label2int)
    map_labels(rudrec_test_set, rudrec_to_cadec, cadec_train_set.label2int)

    map_labels(psytar_train_set, psytar_to_cadec, cadec_train_set.label2int)
    map_labels(psytar_test_set, psytar_to_cadec, cadec_train_set.label2int)

    # dirty hack, remove in the future
    rudrec_unlabeled_set.label2int = rudrec_labeled_set.label2int
    rudrec_unlabeled_set.int2label = rudrec_labeled_set.int2label
    rudrec_unlabeled_set.num_labels = rudrec_labeled_set.num_labels



def make_joined(exp_config,
                cadec_train_set, cadec_test_set,
                psytar_train_set, psytar_test_set):

    teacher_config = exp_config.teacher_config
    teacher_tokenizer = teacher_config.model_type['tokenizer'].from_pretrained(teacher_config.model_checkpoint)

    joined_train_set = BratDataset(None, "train", teacher_tokenizer,
                                   kwargsDataset={}, random_state=exp_config.seed, shuffle=True,
                                   datasets_iter=[cadec_train_set, psytar_train_set])

    joined_test_set = BratDataset(None, "test", teacher_tokenizer,
                             label2int=joined_train_set.label2int,
                             kwargsDataset={}, random_state=exp_config.seed, shuffle=True,
                             datasets_iter=[cadec_test_set, psytar_test_set])

    for idx, elem in enumerate(joined_train_set):
        s = elem['input_ids'].shape
        for key, t in elem.items():
            assert t.shape == s, f"idx = {idx}, mismatch in tensor shapes"

    return joined_train_set, joined_test_set


def make_teacher(exp_config, device, teacher_sets, checkpoint_path=None):
    last_successful_epoch = -1
    teacher_config = exp_config.teacher_config
    teacher_tokenizer = teacher_config.model_type['tokenizer'].from_pretrained(teacher_config.model_checkpoint)

    collate_teacher = lambda x: collate_dicts_(x, pad_id=teacher_tokenizer.pad_token_id)

    teacher_train_set, teacher_test_set = teacher_sets[exp_config.teacher_set]

    teacher_train_dataloader = DataLoader(teacher_train_set,
                                          batch_size=teacher_config.train_batch_sz,
                                          collate_fn=collate_teacher,)
                                          #pin_memory=True)

    teacher_test_dataloader = DataLoader(teacher_test_set,
                                         batch_size=teacher_config.test_batch_sz,
                                         collate_fn=collate_teacher,)
                                         #pin_memory=True)

    set_seed(exp_config.seed)

    teacher_model_cfg = teacher_config.model_type['config'](
        return_dict = True,
        num_labels = len(teacher_train_set.label2int)
    )
    if teacher_config.model_type['model'] is XLMTokenClassifier:
        teacher_model_cfg.emb_dim = 1024

    teacher_model = teacher_config.model_type['model'](teacher_model_cfg)
    if teacher_config.model_checkpoint is not None:
        teacher_model.bert = teacher_model.bert.from_pretrained(teacher_config.model_checkpoint)
    teacher_model.to(device)

    teacher_optimizer = teacher_config.optimizer_class(
        teacher_model.parameters(),
        **teacher_config.optimizer_kwargs
    )


    if checkpoint_path is not None:
        print("loading teacher from checkpoint...")
        chk = torch.load(checkpoint_path)
        last_successful_epoch = chk['epoch']
        teacher_model.load_state_dict(chk['model_state_dict'])
        teacher_optimizer.load_state_dict(chk['optimizer_state_dict'])


    return (teacher_model, teacher_optimizer, last_successful_epoch,
        teacher_train_dataloader, teacher_test_dataloader, collate_teacher)


def train_teacher(exp_config, device,
                teacher_model, teacher_optimizer, last_successful_epoch,
                teacher_train_dataloader, teacher_test_dataloader,
                writer, teacher_save_path):
    teacher_config = exp_config.teacher_config

    total_t0 = time.time()

    for epoch_i in range(last_successful_epoch + 1, teacher_config.epochs):

        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, teacher_config.epochs))
        
        train_model(teacher_model, teacher_train_dataloader, epoch_i, device, teacher_optimizer,
              logging_interval=10, tensorboard_writer=writer, tb_postfix=' (teacher, train, source language)',
              compute_metrics=compute_metrics)
        
        eval_model(teacher_model, teacher_test_dataloader, epoch_i, device,
             logging_interval=10, tensorboard_writer=writer, tb_postfix=' (teacher, test, source language)',
             compute_metrics=compute_metrics)

        teacher_checkpoint_dict = {
            'epoch': epoch_i,
            'model_state_dict': teacher_model.state_dict(),
            'optimizer_state_dict': teacher_optimizer.state_dict(),
        }
        torch.save(teacher_checkpoint_dict, teacher_save_path)
        del teacher_checkpoint_dict

    print("")
    print("Training teacher complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))


def make_student(exp_config, device, student_sets, teacher_train_set, checkpoint_path=None):
    student_config = exp_config.student_config
    sampler_config = exp_config.sampler_config
    student_tokenizer = student_config.model_type['tokenizer'].from_pretrained(student_config.model_checkpoint)

    collate_student = lambda x: collate_dicts_(x, pad_id=student_tokenizer.pad_token_id)

    unlabeled_set, test_set = student_sets[exp_config.student_set]

    set_seed(exp_config.seed)

    student_unlabeled_dataloader = DataLoader(
        unlabeled_set,
        batch_size=sampler_config.n_samples_in,  # feed batches to sampler, sampler reduces them to n_samples_out <= train_batch_sz
        collate_fn=collate_student,
        shuffle = True,  # reshuffle unlabeled samples every epoch
        pin_memory=True
    )

    student_test_dataloader = DataLoader(
        test_set,
        batch_size=student_config.test_batch_sz,
        collate_fn=collate_student,
        pin_memory=True
    )

    last_successful_epoch = -1
    cur_labeled_set = []

    student_model_cfg = student_config.model_type['config'](
        return_dict = True,
        num_labels = len(teacher_train_set.label2int)
    )
    if student_config.model_type['model'] is XLMTokenClassifier:
        student_model_cfg.emb_dim = 1024

    student_model = student_config.model_type['model'](student_model_cfg)
    if student_config.model_checkpoint is not None:
        student_model.bert = student_model.bert.from_pretrained(student_config.model_checkpoint)

    student_model.to(device)

    student_optimizer = student_config.optimizer_class(
        student_model.parameters(),
        **student_config.optimizer_kwargs
    )

    if checkpoint_path is not None:
        print("loading student from checkpoint...")
        chk = torch.load(checkpoint_path)
        last_successful_epoch = chk['epoch']
        student_model.load_state_dict(chk['model_state_dict'])
        student_optimizer.load_state_dict(chk['optimizer_state_dict'])
        cur_labeled_set = chk['cur_labeled_set']

    sampler = sampler_config.sampler_class(**sampler_config.sampler_kwargs)

    return (student_model, student_optimizer, last_successful_epoch,
        student_unlabeled_dataloader, student_test_dataloader, sampler, collate_student, cur_labeled_set)


def train_student(exp_config, device, last_successful_epoch,
                  teacher_args, student_args,
                  sampler, writer, rudrec_labeled_set, cur_labeled_set, student_save_path, teacher_save_path):

    teacher_config = exp_config.teacher_config
    student_config = exp_config.student_config

    total_t0 = time.time()

    #cur_labeled_set = []

    rudrec_labeled_b_sz = min(
        teacher_config.train_batch_sz,
        student_config.train_batch_sz
    )

    (teacher_model, teacher_optimizer, collate_teacher) = teacher_args
    (student_model, student_optimizer, student_unlabeled_dataloader, student_test_dataloader, collate_student) = student_args

    for epoch_i in range(last_successful_epoch + 1, student_config.epochs):

        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, student_config.epochs))

        if exp_config.n_few_shot > 0:
            teacher_labeled_loader, student_labeled_loader = get_cur_labeled_loaders(cur_labeled_set, rudrec_labeled_b_sz,
                rudrec_labeled_set, collate_teacher, collate_student)

        if exp_config.n_few_shot > 0:
            # first make a step with teacher for labeled samples
            if len(teacher_labeled_loader) > 0:
                train_model(teacher_model, teacher_labeled_loader, epoch_i, device, teacher_optimizer,
                    logging_interval=10, tensorboard_writer=writer, tb_postfix=' (teacher, few-shot, rudrec)',
                    compute_metrics=compute_metrics,
                    int2label=rudrec_labeled_set.int2label)
        
        # tune a student with teacher
        train_model(student_model, student_unlabeled_dataloader, epoch_i, device, student_optimizer,
              teacher_model=teacher_model, sampler=sampler,
              logging_interval=10, tensorboard_writer=writer, tb_postfix=' (student, train, rudrec)',
              compute_metrics=compute_metrics)
        
        if exp_config.n_few_shot > 0:
            # tune a student with labeled samples
            train_model(student_model, student_labeled_loader, epoch_i, device, student_optimizer,
                logging_interval=10, tensorboard_writer=writer, tb_postfix=' (student, few-shot, rudrec)',
                compute_metrics=compute_metrics,
                int2label=rudrec_labeled_set.int2label)
        
        eval_model(student_model, student_test_dataloader, epoch_i, device,
             logging_interval=10, tensorboard_writer=writer, tb_postfix=' (student, test, rudrec)',
             compute_metrics=compute_metrics)


        student_checkpoint_dict = {
            'epoch': epoch_i,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': student_optimizer.state_dict(),
            'cur_labeled_set': cur_labeled_set
        }
        torch.save(student_checkpoint_dict, student_save_path)

        del student_checkpoint_dict

        teacher_checkpoint_dict = {
            'epoch': epoch_i,
            'model_state_dict': teacher_model.state_dict(),
            'optimizer_state_dict': teacher_optimizer.state_dict(),
        }
        torch.save(teacher_checkpoint_dict, teacher_save_path)

        del teacher_checkpoint_dict

    print("")
    print("Training student complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))



def main(path_to_yaml, runs_path,
    teacher_load_path=None, teacher_save_path=None,
    student_load_path=None, student_save_path=None,
    do_train_teacher=True, do_train_student=True):
    
    # todo os setpath?

    exp_config = read_yaml_config(path_to_yaml)

    print(exp_config)

    teacher_config = exp_config.teacher_config
    student_config = exp_config.student_config

    teacher_tokenizer = teacher_config.model_type['tokenizer'].from_pretrained(teacher_config.model_checkpoint)
    student_tokenizer = student_config.model_type['tokenizer'].from_pretrained(student_config.model_checkpoint)

    folds_dir = './folds'  

    cadec_train_set, cadec_test_set = make_cadec(folds_dir, exp_config)

    psytar_train_set, psytar_test_set = make_psytar(folds_dir, exp_config)

    rudrec_labeled_set, rudrec_test_set, rudrec_unlabeled_set = make_rudrec(folds_dir, exp_config)

    ############################
    ### make sure the labels are consistent
    ############################

    rudrec_to_cadec, psytar_to_cadec = make_mappers(exp_config)

    unify_data(rudrec_to_cadec, psytar_to_cadec, cadec_train_set,
               rudrec_labeled_set, rudrec_test_set, rudrec_unlabeled_set,
               psytar_train_set, psytar_test_set)

    ############################
    # make a joined set
    ############################

    joined_train_set, joined_test_set = make_joined(exp_config,
                cadec_train_set, cadec_test_set,
                psytar_train_set, psytar_test_set)
    ############################
    # configure teacher
    ############################

    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        device_name = torch.cuda.get_device_name(0)
        print('We will use the GPU:', device_name)
        if 'K80' in device_name:
            print('wild K80 appears. lowering the batch size...')
            student_config.train_batch_sz = 2
            exp_config.student_config.train_batch_sz = 2

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")    


    teacher_sets = {
        'cadec': (cadec_train_set, cadec_test_set),
        'psytar': (psytar_train_set, psytar_test_set),
        'joined': (joined_train_set, joined_test_set),
    }


    (teacher_model, teacher_optimizer, last_successful_epoch,
    teacher_train_dataloader, teacher_test_dataloader, collate_teacher) = make_teacher(exp_config, device,
                                                                                        teacher_sets, teacher_load_path)

    ############################
    ############ train a teacher
    ############################    

    writer = SummaryWriter(log_dir=os.path.join(runs_path, exp_config.experiment_name))
    writer.add_text('experiment_info', str(exp_config))

    if do_train_teacher:
        train_teacher(exp_config, device,
                    teacher_model, teacher_optimizer, last_successful_epoch,
                    teacher_train_dataloader, teacher_test_dataloader,
                    writer, teacher_save_path)

    '''
    print("\n\nmemory stats:")
    print("total:", torch.cuda.get_device_properties(0).total_memory)
    print("reserved:",torch.cuda.memory_reserved(0))
    print("allocated:",torch.cuda.memory_allocated(0))
    '''


    del teacher_train_dataloader
    del teacher_test_dataloader
    #torch.cuda.empty_cache()

    print(torch.cuda.memory_summary(0))

    '''
    print("\n\nemptied cache. stats:")
    print("total:", torch.cuda.get_device_properties(0).total_memory)
    print("reserved:",torch.cuda.memory_reserved(0))
    print("allocated:",torch.cuda.memory_allocated(0))
    '''


    ############################
    # make a student
    ############################

    student_sets = {
        'small': (rudrec_unlabeled_set, rudrec_test_set),
    }

    teacher_train_set = teacher_sets[exp_config.teacher_set][0]

    (student_model, student_optimizer, last_successful_epoch,
    student_unlabeled_dataloader, student_test_dataloader,
    sampler, collate_student, cur_labeled_set) = make_student(exp_config, device, student_sets, teacher_train_set, student_load_path)

    ############################
    # train student
    ############################

    teacher_args = (teacher_model, teacher_optimizer, collate_teacher)
    student_args = (student_model, student_optimizer, student_unlabeled_dataloader, student_test_dataloader, collate_student)

    if do_train_student:
        train_student(exp_config, device, last_successful_epoch,
                      teacher_args, student_args,
                      sampler, writer, rudrec_labeled_set, cur_labeled_set, student_save_path, teacher_save_path)
import os
from glob import glob
import sys
import torch
import numpy as np
from NLPDatasetIO.dataset import Dataset
from transformers import BertTokenizer, XLMTokenizer
import json
from nltk.tokenize import sent_tokenize


class BratDataset(torch.utils.data.Dataset):
    def __init__(self, fold_path, fold_type, tokenizer, labeled=True, label2int=None, kwargsDataset={'format':'brat'},
                 to_sentences=False, random_state=None, shuffle=False, datasets_iter=None, is_binary=False):
        '''
          fold_path: path to fold folder, must contain corresponding .txt and .ann files
          fold_type: 'train', 'dev' or 'test'
          tokenizer: tokenizer to use with dataset
          kwargsDataset: dict with options for NLPDatasetIO.Dataset
          to_sentences: whether to split each document into sentences
        '''
        assert fold_type == 'train' or fold_type == 'test' or fold_type == 'dev'
        if fold_type != 'train' and labeled:
            assert label2int is not None

        self.fold_type = fold_type
        self.fold_path = fold_path

        if datasets_iter is None:
            self.documents = Dataset(location=fold_path, split=fold_type, **kwargsDataset).documents
        else:
            self.documents = []
            for dataset in datasets_iter:
                self.documents.extend(dataset.documents)

        if to_sentences and datasets_iter is None:
            sentences = []
            for doc in self.documents:
                sentences.extend(doc.sentences)
            self.documents = sentences

        self.tokenizer = tokenizer
        self.labeled = labeled
        self.is_binary = is_binary

        if self.labeled:
            if datasets_iter is None:
                self.labels = [doc.token_labels for doc in self.documents]
            else:
                self.labels = []
                for dataset in datasets_iter:
                    self.labels.extend(dataset.labels)

            if self.is_binary: # make it ADR vs Other
                for idx, doc_labels in enumerate(self.labels):
                    self.labels[idx] = list(map(lambda label: label if 'ADR' in label else 'O', doc_labels))

            self.set_label_info(label2int)


        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        self.shuffle = shuffle
        if shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(self.documents)
            if self.labeled:
                np.random.set_state(rng_state)
                np.random.shuffle(self.labels)
            


    def set_label_info(self, label2int):
        self.label_set = set(['O'])
        for token_labels in self.labels:
            self.label_set = self.label_set | set(token_labels)

        if label2int is None:  # learn labels
            self.label2int = {'O': 0}
            for idx, label in enumerate(sorted(self.label_set - set(['O'])), 1):
                self.label2int[label] = idx
        else:  # set labels from other fold
            self.label2int = label2int

        self.int2label = {val: key for key, val in self.label2int.items()}

        self.num_labels = len(self.int2label)


    def __len__(self):
        return len(self.documents)


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        document = self.documents[idx]
        
        #encoded_text = self.tokenizer.encode_plus(document.text, max_length=512)
        #can't use that robustly because of how NLPDatasetIO works

        # do it manually, I guess
        preceding_token_id, trailing_token_id = None, None
        if isinstance(self.tokenizer, BertTokenizer):
            preceding_token_id, trailing_token_id = (self.tokenizer.cls_token_id,
                                                     self.tokenizer.sep_token_id)
        if isinstance(self.tokenizer, XLMTokenizer):
            preceding_token_id, trailing_token_id = (self.tokenizer.bos_token_id,
                                                     self.tokenizer.sep_token_id)

        text_tokens = [token.token for token in document._tokens][:510]
        encoded_text = {}
        encoded_text['input_ids'] = ([preceding_token_id] + 
                                     self.tokenizer.convert_tokens_to_ids(text_tokens) +
                                     [trailing_token_id])
        encoded_text['token_type_ids'] = torch.zeros(len(encoded_text['input_ids'])).long()
        encoded_text['attention_mask'] = torch.ones(len(encoded_text['input_ids'])).long()

        item = {key: torch.tensor(val) for key, val in encoded_text.items()}

        if self.labeled:
            encoded_labels = list(map(lambda elem: self.label2int.get(elem, self.label2int['O']),
                              self.labels[idx][:len(encoded_text['input_ids'])-2]))
            labels = [self.label2int['O']] + encoded_labels + [self.label2int['O']]
            item['labels'] = torch.tensor(labels)

        return item



class JsonDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_json, tokenizer, labeled=False, sample_count=None,
        random_state=None, shuffle=False, datasets_iter=None, tokenize=None,
        to_sentences=False):
        '''
          
        '''

        self.documents = []
        self.tokenizer = tokenizer
        self.tokenize = tokenize if tokenize is not None else tokenizer.tokenize
        self.to_sentences = to_sentences
        with open(path_to_json) as json_file:
            data = json.load(json_file)
            for p in data:
                text = p.get('comment', None)
                if text is not None and text != '':
                    if self.to_sentences:
                        sentences = sent_tokenize(text)
                        self.documents.extend(sentences)
                    else:
                        self.documents.append(text)

        self.labeled = labeled
        self.labels = []
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        self.shuffle = shuffle
        if shuffle:
            rng_state = np.random.get_state()
            np.random.shuffle(self.documents)
            if self.labeled:
                np.random.set_state(rng_state)
                np.random.shuffle(self.labels)

        if sample_count is not None and sample_count < len(self.documents):
            self.documents = self.documents[:sample_count]
            

    def __len__(self):
        return len(self.documents)


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        document = self.documents[idx]
        
        #encoded_text = self.tokenizer.encode_plus(document.text, max_length=512)
        #can't use that robustly because of how NLPDatasetIO works

        # do it manually, I guess
        preceding_token_id, trailing_token_id = None, None
        if isinstance(self.tokenizer, BertTokenizer):
            preceding_token_id, trailing_token_id = (self.tokenizer.cls_token_id,
                                                     self.tokenizer.sep_token_id)
        if isinstance(self.tokenizer, XLMTokenizer):
            preceding_token_id, trailing_token_id = (self.tokenizer.bos_token_id,
                                                     self.tokenizer.sep_token_id)

        text_tokens = [token for token in self.tokenize(document)][:510]
        encoded_text = {}
        encoded_text['input_ids'] = ([preceding_token_id] + 
                                     self.tokenizer.convert_tokens_to_ids(text_tokens) +
                                     [trailing_token_id])
        encoded_text['token_type_ids'] = torch.zeros(len(encoded_text['input_ids'])).long()
        encoded_text['attention_mask'] = torch.ones(len(encoded_text['input_ids'])).long()

        item = {key: torch.tensor(val) for key, val in encoded_text.items()}

        if self.labeled:
            encoded_labels = list(map(lambda elem: self.label2int.get(elem, self.label2int['O']),
                              self.labels[idx][:len(encoded_text['input_ids'])-2]))
            labels = [self.label2int['O']] + encoded_labels + [self.label2int['O']]
            item['labels'] = torch.tensor(labels)

        return item

import os
from glob import glob
import sys
import torch
from NLPDatasetIO.dataset import Dataset


class CADECDataset(torch.utils.data.Dataset):
    def __init__(self, fold_path, fold_type, tokenizer, kwargsDataset={}, to_sentences=False):
        '''
          fold_path: path to fold folder, must contain corresponding .txt and .ann files
          fold_type: 'train', 'dev' or 'test'
          tokenizer: tokenizer to with dataset
          kwargsDataset: dict with options for NLPDatasetIO.Dataset
          to_sentences: whether to split each document into sentences
        '''
        self.fold_type = fold_type
        self.fold_path = fold_path

        self.documents = Dataset(location=fold_path, format='brat', split=fold_type,
                                 tokenize=tokenizer.tokenize, **kwargsDataset).documents

        if to_sentences:
            sentences = []
            for doc in self.documents:
                sentences.extend(doc.sentences)
            self.documents = sentences

        self.labels = [doc.token_labels for doc in self.documents]

        label_set = set(['UNK'])
        for token_labels in self.labels:
            label_set = label_set | set(token_labels)

        self.label2int = {'UNK': 0}
        for idx, label in enumerate(label_set, 1):
            self.label2int[label] = idx

        self.tokenizer = tokenizer


    def __len__(self):

        return len(self.documents)


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        document = self.documents[idx]
        encoded_text = self.tokenizer.encode_plus(document.text)
        labels = list(map(lambda elem: self.label2int.get(elem, self.label2int['UNK']),
                          self.labels[idx]))

        item = {key: torch.tensor(val) for key, val in encoded_text.items()}
        item['labels'] = torch.tensor(labels)

        return item


class PsytarDataset(torch.utils.data.Dataset):
    def __init__(self, corpus_dir):
        raise NotImplementedError


    def __getitem__(self, idx):
        if torch.is_tensor(index):
            index = index.tolist(index)

        raise NotImplementedError


    def __len__(self):

        raise NotImplementedError

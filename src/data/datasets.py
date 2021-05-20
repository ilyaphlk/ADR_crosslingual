import os
from glob import glob
import sys
import torch
from NLPDatasetIO.dataset import Dataset


class CadecDataset(torch.utils.data.Dataset):
    def __init__(self, fold_path, fold_type, tokenizer, label2int=None, kwargsDataset={},
                 to_sentences=False, random_state=None, shuffle=False):
        '''
          fold_path: path to fold folder, must contain corresponding .txt and .ann files
          fold_type: 'train', 'dev' or 'test'
          tokenizer: tokenizer to with dataset
          kwargsDataset: dict with options for NLPDatasetIO.Dataset
          to_sentences: whether to split each document into sentences
        '''
        assert fold_type == 'train' or fold_type == 'test' or fold_type == 'dev'
        if fold_type != 'train':
            assert label2int is not None

        self.fold_type = fold_type
        self.fold_path = fold_path

        self.documents = Dataset(location=fold_path, format='brat', split=fold_type,
                                 tokenize=tokenizer.tokenize, **kwargsDataset).documents

        if to_sentences:
            sentences = []
            for doc in self.documents:
                sentences.extend(doc.sentences)
            self.documents = sentences

        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.documents)

        self.labels = [doc.token_labels for doc in self.documents]

        self.label_set = set(['UNK'])
        for token_labels in self.labels:
            self.label_set = self.label_set | set(token_labels)

        if self.fold_type == 'train':  # learn labels
            self.label2int = {'UNK': 0}
            for idx, label in enumerate(sorted(self.label_set - set(['UNK'])), 1):
                self.label2int[label] = idx
        else:  # set labels from train
            self.label2int = label2int

        self.int2label = {val: key for key, val in self.label2int.items()}

        self.tokenizer = tokenizer

        self.num_labels = len(self.int2label)


    def __len__(self):

        return len(self.documents)


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        document = self.documents[idx]
        encoded_text = self.tokenizer.encode_plus(document.text, max_length=512)
        encoded_labels = list(map(lambda elem: self.label2int.get(elem, self.label2int['UNK']),
                          self.labels[idx][:len(encoded_text['input_ids'])-2]))

        labels = [self.label2int['UNK']] + encoded_labels + [self.label2int['UNK']]

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

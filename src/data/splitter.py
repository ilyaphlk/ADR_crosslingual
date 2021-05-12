import os
import sys
from glob import glob
import numpy as np
from shutil import copy2
from dataclasses import dataclass


@dataclass
class Fold:
    path: str = None
    share: float = None
    size: int = None


class CadecSplitter:
    def __init__(self, corpus_dir, train_share, test_share, dev_share=0., name_prefix="", override=False, shuffle=True):
        '''
            corpus_dir - path to folder containing CADEC corpus. Must contain "original" and "text" folders
            <fold>_share - expected sampling share of a fold
            name_prefix - prefix for naming resulting folds directories
            override - whether or not to rewrite the existing split
        '''
        self.corpus_dir = corpus_dir

        self.train_fold = Fold(share=train_share)
        self.test_fold = Fold(share=test_share)
        self.dev_fold = Fold(share=dev_share)

        self.name_prefix = name_prefix
        self.override = override
        self.split_successful = False


    def split():

        # todo: check split for existence
        self.train_fold.path = os.path.join(corpus_dir, name_prefix+"_train")
        self.test_fold.path = os.path.join(corpus_dir, name_prefix+"_test")
        self.dev_fold.path = os.path.join(corpus_dir, name_prefix+"_dev")

        if (any([os.path.exists(self.train_fold.path),
                 os.path.exists(self.test_fold.path),
                 os.path.exists(self.dev_fold.path)])):
            # TODO: implement override logic
            pass

        txt_files_pattern = os.path.join(corpus_dir, 'text', '*.txt')
        ann_files_pattern = os.path.join(corpus_dir, 'original','*.ann')

        self.txt_files = sorted([txt_file for txt_file in glob(txt_files_pattern)])
        self.ann_files = sorted([ann_file for ann_file in glob(ann_files_pattern)])
        
        if not self.is_consistent_set():
            # TODO: find consistent pairs and use them
            raise Exception("found mismatched pairs of txt/ann files.")

        self.idx = np.arange(txt_files.shape[0])
        if self.shuffle:
            np.random.shuffle(self.idx)

        train_size = make_fold(self.train_fold, 0)
        test_size = make_fold(self.test_fold, train_size)
        make_fold(self.dev_fold, train_size + test_size)

        self.split_successful = True
        print("Split successful.")

        return train_len, dev_len, test_len


    def is_consistent_set(self):
        for txt, ann in zip(self.txt_files, self.ann_files):
            if txt.split('/')[-1][:-4] != ann.split('/')[-1][:-4]:
                return False
        return True


    def make_fold(self, fold, start_idx):
        # todo make option for deleting folds
        fold_len = int(fold.share * self.idx.shape[0])
        fold_idx = self.idx[start_idx:start_idx+fold_len]

        os.mkdir(fold.path)

        for txt_file, ann_file in zip(self.txt_files[fold_idx], self.ann_files[fold_idx]):
            copy2(txt_file, fold.path)
            copy2(ann_file, fold.path)

        self.fold.size = fold_len

        return fold_len


if __name__ == "__main__":

    try:
        corpus_dir = sys.argv[1]
    except:
        raise Exception("dir with CADEC corpus not specified. Aborting.")

    splitter = CadecSplitter(corpus_dir, 0.01, 0.005, 0.005)
    splitter.split()

    print(splitter.train_fold)
    print(splitter.test_fold)
    print(splitter.dev_fold)

    


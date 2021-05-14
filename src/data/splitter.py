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
    def __init__(self, corpus_dir, train_share, test_share,
                 dev_share=0.,
                 out_dir=None,
                 random_state=None,
                 name_postfix="",
                 override=False,
                 shuffle=True):
        '''
            corpus_dir - path to folder containing CADEC corpus. Must contain "original" and "text" folders
            <fold>_share - expected sampling share of a fold
            name_postfix - postfix for naming resulting folds directories
            override - whether or not to rewrite the existing split
        '''
        self.corpus_dir = corpus_dir

        self.train_fold = Fold(share=train_share)
        self.test_fold = Fold(share=test_share)
        self.dev_fold = Fold(share=dev_share)

        if out_dir is None:
            out_dir = corpus_dir
        self.out_dir = out_dir
        
        self.random_state = random_state
        self.name_postfix = name_postfix
        self.override = override
        self.split_successful = False
        self.shuffle = shuffle


    def split(self):

        # todo: check split for existence
        self.train_fold.path = os.path.join(self.out_dir, "train"+self.name_postfix)
        self.test_fold.path = os.path.join(self.out_dir, "test"+self.name_postfix)
        self.dev_fold.path = os.path.join(self.out_dir, "dev"+self.name_postfix)

        if (any([os.path.exists(self.train_fold.path),
                 os.path.exists(self.test_fold.path),
                 os.path.exists(self.dev_fold.path)])):

            # TODO: implement override logic
            raise Exception("Split with specified name postfix already exists. Aborting.")
            

        txt_files_pattern = os.path.join(self.corpus_dir, 'text', '*.txt')
        ann_files_pattern = os.path.join(self.corpus_dir, 'original','*.ann')

        self.txt_files = sorted([txt_file for txt_file in glob(txt_files_pattern)])
        self.ann_files = sorted([ann_file for ann_file in glob(ann_files_pattern)])
        
        if not self.is_consistent_set():
            # TODO: find consistent pairs and use them
            raise Exception("found mismatched pairs of txt/ann files.")

        self.txt_files = np.array(self.txt_files)
        self.ann_files = np.array(self.ann_files)

        self.idx = np.arange(self.txt_files.shape[0])

        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self.shuffle:
            np.random.shuffle(self.idx)

        train_size = self.make_fold(self.train_fold, 0)
        test_size = self.make_fold(self.test_fold, train_size)
        self.make_fold(self.dev_fold, train_size + test_size)

        self.split_successful = True
        print("Split successful.")


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

        fold.size = fold_len

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

    


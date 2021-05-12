import os
from glob import glob
import numpy as np
from shutil import copy2


class CadecSplitter:
    def __init__(self, corpus_dir, train_share, test_share, dev_share=0., name_prefix="", override=False, shuffle=True):
        '''
            corpus_dir - path to folder containing CADEC corpus. Must contain "original" and "text" folders
            <fold>_share - expected sampling share of a fold
            name_prefix - prefix for naming resulting folds directories
            override - whether or not to rewrite the existing split
        '''
        self.corpus_dir = corpus_dir
        self.train_share = train_share
        self.test_share = test_share
        self.dev_share = dev_share
        self.name_prefix = name_prefix
        self.override = override
        self.split_successful = False


    def split():

        # todo: check split for existence
        self.train_path = os.path.join(corpus_dir, name_prefix+"_train")
        self.test_path = os.path.join(corpus_dir, name_prefix+"_test")
        self.dev_path = os.path.join(corpus_dir, name_prefix+"_dev")

        if (any(os.path.exists(train_path),
                os.path.exists(test_path),
                os.path.exists(dev_path))):
            # TODO implement override logic
            pass

        txt_files_pattern = os.path.join(corpus_dir, 'text', '*.txt')
        ann_files_pattern = os.path.join(corpus_dir, 'original','*.ann')

        self.txt_files = sorted([txt_file for txt_file in glob(txt_files_pattern)])
        self.ann_files = sorted([ann_file for ann_file in glob(ann_files_pattern)])


        # TODO: find consistent pairs and use them
        if not self.is_consistent_set():
            raise Exception("found mismatched pairs of txt/ann files.")


        self.idx = np.arange(txt_files.shape[0])
        if self.shuffle:
            np.random.shuffle(self.idx)


        train_len = make_fold(self.train_share, 0, self.train_path)
        test_len = make_fold(self.test_share, train_len, self.test_path)
        dev_len = make_fold(self.dev_share, train_len + test_len, self.dev_path)

        return train_len, dev_len, test_len


    def is_consistent_set(self):
        for txt, ann in zip(self.txt_files, self.ann_files):
            if txt.split('/')[-1][:-4] != ann.split('/')[-1][:-4]:
                return False
        return True



    def make_fold(self, fold_share, start_idx, fold_path):
        # todo make option for deleting folds
        fold_len = int(fold_share * self.idx.shape[0])
        fold_idx = self.idx[start_idx:start_idx+fold_len]

        os.mkdir(fold_path)

        for txt_file, ann_file in zip(self.txt_files[fold_idx], self.ann_files[fold_idx]):
            copy2(txt_file, fold_path)
            copy2(ann_file, fold_path)

        return fold_len


if __name__ == "__main__":

    fold_lens = splitter('/home/ipakhalko/Documents/Lectures/TermWork/data/CADEC/cadec', 0.01, 0.005, 0.005, debug=True)

    print(fold_lens)

    


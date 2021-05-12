import os
from glob import glob
import numpy as np
from shutil import copy2


def is_consistent_set(txt_files, ann_files):
    for txt, ann in zip(sorted(txt_files), sorted(ann_files)):
        if txt.split('/')[-1][:-4] != ann.split('/')[-1][:-4]:
            return False
    return True


def make_fold(txt_files, ann_files, fold_share, idx, left, fold_path, debug):
    # todo make option for deleting folds
    fold_len = int(fold_share * idx.shape[0])
    fold_idx = idx[left:left+fold_len]
    if debug:
        fold_path += '_debug'

    try: 
        os.mkdir(fold_path)
    except OSError as error: 
        print("Skipping mkdir;", error)

    for txt_file, ann_file in zip(txt_files[fold_idx], ann_files[fold_idx]):
        copy2(txt_file, fold_path)
        copy2(ann_file, fold_path)



    return fold_len


def splitter(corpus_dir, train_share, dev_share, test_share=0., debug=False):
    txt_files_pattern = os.path.join(corpus_dir, 'text', '*.txt')
    ann_files_pattern = os.path.join(corpus_dir, 'original','*.ann')

    txt_files = [txt_file for txt_file in glob(txt_files_pattern)]
    ann_files = [ann_file for ann_file in glob(ann_files_pattern)]

    # TODO: check consistency - we need only matched pairs of files
    if not is_consistent_set(txt_files, ann_files):
        raise Exception("found mismatched pairs of txt/ann files.")


    # Now everything is consistent
    txt_files = np.sort(txt_files)
    ann_files = np.sort(ann_files)

    idx = np.arange(txt_files.shape[0])
    np.random.shuffle(idx)

    train_len = make_fold(txt_files, ann_files, train_share,
        idx, 0, os.path.join(corpus_dir, "train"), debug)

    dev_len = make_fold(txt_files, ann_files, dev_share,
        idx, train_len, os.path.join(corpus_dir, "dev"), debug)

    test_len = make_fold(txt_files, ann_files, test_share,
        idx, train_len+dev_len, os.path.join(corpus_dir, "test"), debug)

    return train_len, dev_len, test_len


if __name__ == "__main__":

    fold_lens = splitter('/home/ipakhalko/Documents/Lectures/TermWork/data/CADEC/cadec', 0.01, 0.005, 0.005, debug=True)

    print(fold_lens)

    


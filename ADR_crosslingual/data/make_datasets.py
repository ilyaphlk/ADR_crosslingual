# all the imports
from ADR_crosslingual.data.datasets import BratDataset, JsonDataset
from ADR_crosslingual.data.splitter import BratSplitter
from ADR_crosslingual.data.convert import webanno2brat


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

    is_binary = (exp_config.classification_type == 'binary')
    cadec_train_set = BratDataset(folds_dir+"/train"+cadec_splitter.name_postfix, "train", teacher_tokenizer,
                         kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True, is_binary=is_binary)

    cadec_test_set = BratDataset(folds_dir+"/test"+cadec_splitter.name_postfix, "test", teacher_tokenizer,
                            label2int=cadec_train_set.label2int,
                            kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True, is_binary=is_binary)

    '''
    for idx, elem in enumerate(cadec_train_set):
        s = elem['input_ids'].shape
        for key, t in elem.items():
            assert t.shape == s, f"idx = {idx}, mismatch in tensor shapes"
    '''


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
    

    is_binary = (exp_config.classification_type == 'binary')
    psytar_train_set = BratDataset(train_fold_dir, "train", teacher_tokenizer,
                         kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True, is_binary=is_binary)

    psytar_test_set = BratDataset(test_fold_dir, "test", teacher_tokenizer,
                             label2int=psytar_train_set.label2int,
                             kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True, is_binary=is_binary)
    '''
    for idx, elem in enumerate(psytar_test_set):
        s = elem['input_ids'].shape
        for key, t in elem.items():
            assert t.shape == s, f"idx = {idx}, mismatch in tensor shapes"
    '''

    return psytar_train_set, psytar_test_set


def make_rudrec(folds_dir, exp_config):
    student_config = exp_config.student_config
    rudrec_brat_dir = './rudrec_brat'
    rudrec_size = webanno2brat('./rudrec_labeled', rudrec_brat_dir)

    rudrec_folds_dir = folds_dir

    train_share = exp_config.n_few_shot / rudrec_size
    test_share = 0.1
    unlabeled_share = 1 - train_share - test_share

    # to fix the test set in place, we flip the dev and test folds
    rudrec_splitter = BratSplitter(rudrec_brat_dir+"/text", rudrec_brat_dir+"/annotation", rudrec_folds_dir,
                                   train_share, unlabeled_share, test_share,
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

    is_binary = (exp_config.classification_type == 'binary')

    rudrec_labeled_set = BratDataset(rudrec_folds_dir+"/train"+rudrec_splitter.name_postfix, "train", student_tokenizer,
                        kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True, is_binary=is_binary)

    rudrec_test_set = BratDataset(rudrec_folds_dir+"/dev"+rudrec_splitter.name_postfix, "test", student_tokenizer,
                        label2int=rudrec_labeled_set.label2int,
                        kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True, is_binary=is_binary)

    rudrec_unlabeled_set = BratDataset(rudrec_folds_dir+"/test"+rudrec_splitter.name_postfix, "dev", student_tokenizer,
                       labeled=False,
                       kwargsDataset=bratlike_dict, random_state=exp_config.seed, shuffle=True, is_binary=is_binary)

    '''
    for idx, elem in enumerate(rudrec_test_set):
        s = elem['input_ids'].shape
        for key, t in elem.items():
            assert t.shape == s, f"idx = {idx}, mismatch in tensor shapes"
    '''

    return rudrec_labeled_set, rudrec_test_set, rudrec_unlabeled_set


def make_mappers(exp_config):
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
               psytar_train_set, psytar_test_set,
               big_unlabeled_set):

    map_labels(rudrec_labeled_set, rudrec_to_cadec, cadec_train_set.label2int)
    map_labels(rudrec_test_set, rudrec_to_cadec, cadec_train_set.label2int)

    map_labels(psytar_train_set, psytar_to_cadec, cadec_train_set.label2int)
    map_labels(psytar_test_set, psytar_to_cadec, cadec_train_set.label2int)

    # dirty hack, remove in the future
    rudrec_unlabeled_set.label2int = rudrec_labeled_set.label2int
    rudrec_unlabeled_set.int2label = rudrec_labeled_set.int2label
    rudrec_unlabeled_set.num_labels = rudrec_labeled_set.num_labels

    big_unlabeled_set.label2int = rudrec_labeled_set.label2int
    big_unlabeled_set.int2label = rudrec_labeled_set.int2label
    big_unlabeled_set.num_labels = rudrec_labeled_set.num_labels    



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



def make_datasets(exp_config):
    teacher_config = exp_config.teacher_config
    student_config = exp_config.student_config

    teacher_tokenizer = teacher_config.model_type['tokenizer'].from_pretrained(teacher_config.model_checkpoint)
    student_tokenizer = student_config.model_type['tokenizer'].from_pretrained(student_config.model_checkpoint)

    folds_dir = './folds'  

    cadec_train_set, cadec_test_set = make_cadec(folds_dir, exp_config)
    print("set of labels:", cadec_train_set.label_set)

    psytar_train_set, psytar_test_set = make_psytar(folds_dir, exp_config)

    rudrec_labeled_set, rudrec_test_set, rudrec_unlabeled_set = make_rudrec(folds_dir, exp_config)

    json_dir = './consumers_drugs_reviews.json'

    big_unlabeled_set = JsonDataset(json_dir, teacher_tokenizer,
        labeled=False, sample_count=exp_config.big_set_sample_cnt,
        random_state=exp_config.seed, shuffle=True, tokenize=exp_config.common_tokenize)

    ############################
    ### make sure the labels are consistent
    ############################

    rudrec_to_cadec, psytar_to_cadec = make_mappers(exp_config)

    unify_data(rudrec_to_cadec, psytar_to_cadec,
               cadec_train_set,
               rudrec_labeled_set, rudrec_test_set, rudrec_unlabeled_set,
               psytar_train_set, psytar_test_set,
               big_unlabeled_set)

    ############################
    # make a joined set
    ############################

    joined_train_set, joined_test_set = make_joined(exp_config,
                cadec_train_set, cadec_test_set,
                psytar_train_set, psytar_test_set)


    cadec_tuple = (cadec_train_set, cadec_test_set)
    psytar_tuple = (psytar_train_set, psytar_test_set)
    rudrec_tuple = (rudrec_labeled_set, rudrec_test_set, rudrec_unlabeled_set)
    joined_tuple = (joined_train_set, joined_test_set)

    return cadec_tuple, psytar_tuple, rudrec_tuple, joined_tuple, big_unlabeled_set


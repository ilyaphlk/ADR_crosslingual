from dataclasses import dataclass
from typing import Any

from transformers import (
    BertTokenizer, BertConfig, BertModel,
    AdamW,
)

from ADR_crosslingual.models.single_model import BertTokenClassifier, XLMTokenClassifier


class TrainConfig:
    def __init__(self,
        model_type={
            'tokenizer':BertTokenizer,
            'config':BertConfig,
            'model':BertTokenClassifier,
            'subword_prefix': '##'
        },
        optimizer_class=AdamW,
        optimizer_kwargs={'lr':2e-5, 'eps':1e-8},
        model_checkpoint='bert-base-multilingual-cased',
        train_batch_sz=1,
        test_batch_sz=1,
        epochs = 1,
    ):
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.model_type = model_type
        self.model_checkpoint = model_checkpoint
        self.train_batch_sz = train_batch_sz
        self.test_batch_sz = test_batch_sz
        self.epochs = epochs


    def __str__(self):
        return "\n".join(["model_type = "+str(self.model_type['model']),
                          "model_checkpoint = "+str(self.model_checkpoint),
                          "optimizer = "+str(self.optimizer_class),
                          "optimizer_kwargs = "+str(self.optimizer_kwargs),
                          "train_batch_sz = "+str(self.train_batch_sz),
                          "test_batch_sz = "+str(self.test_batch_sz),])


class SamplerConfig:
    def __init__(self,
        sampler_class=None,
        sampler_kwargs={
            'strategy':None,
            'n_samples_out':None
        },
        n_samples_in=None
    ):
        self.sampler_class = sampler_class
        self.sampler_kwargs = sampler_kwargs
        self.n_samples_in = n_samples_in

    def __str__(self):
        return "\n".join(["sampler = ", str(self.sampler_class),
                          "sampler_args = ", str(self.sampler_kwargs),
                          "n_samples_in = ", str(self.n_samples_in)])


@dataclass
class ExperimentConfig:
    teacher_config: TrainConfig
    student_config: TrainConfig
    sampler_config: SamplerConfig
    experiment_name: str = "sample_exp_name"
    seed: int = 42
    n_few_shot: int = 0
    common_tokenize: Any = None
    teacher_set: str = "cadec"
    student_set: str = "small"
    classification_type: str = "multiclass"


    def __str__(self):
        return "\n".join(["experiment_name = "+str(self.experiment_name)+"; ",
                         "n_few_shot = "+str(self.n_few_shot)+"; ",
                         "teacher_set = "+str(self.teacher_set)+"; \n",
                         "teacher:\n"+str(self.teacher_config)+"\n",
                         "student:\n"+str(self.student_config)+"\n",
                         "sampler:\n"+str(self.sampler_config)])

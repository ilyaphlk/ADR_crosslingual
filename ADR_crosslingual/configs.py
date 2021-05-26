from dataclasses import dataclass

from transformers import (
    BertTokenizer, BertConfig, BertPretrainedModel, BertModel
    AdamW,

)


class TrainConfig:
    def __init__(self,
        optimizer_class=AdamW,
        optimizer_kwargs={'lr':2e-5},
        model_type={
            'tokenizer':BertTokenizer,
            'config':BertConfig,
            'pretrained':BertPretrainedModel
            'model':BertModel,
        },
        model_checkpoint='bert-base-multilingual-cased',
        train_batch_sz=1,
        test_batch_sz=1,
    ):

    self.optimizer_class = optimizer_class
    self.optimizer_kwargs = optimizer_kwargs
    self.model_type = model_type
    self.model_checkpoint = model_checkpoint
    self.train_batch_sz = train_batch_sz
    self.test_batch_sz = test_batch_sz

    def __str__(self):
        return "; ".join([str(self.model_type['model']),
                          str(self.optimizer_class),
                          str(self.optimizer_kwargs),
                          "train_batch_sz="+str(self.train_batch_sz),
                          "test_batch_sz="+str(self.test_batch_sz)])


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
        return "; ".join(["sampler=", str(sampler_class),
                          "sampler_args=", str(sampler_kwargs),
                          "n_samples_in=", str(n_samples_in)])


@dataclass
class ExperimentConfig:
    teacher_config: TrainConfig,
    student_config: TrainConfig,
    sampler_config: SamplerConfig,
    experiment_name: "sample_exp_name"


    def __str__():
        return "\n".join(self.experiment_name,
                         "teacher="+str(self.teacher_config),
                         "student="+str(self.student_config),
                         "sampler="+str(self.sampler_config))

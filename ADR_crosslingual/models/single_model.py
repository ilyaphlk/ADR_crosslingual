import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import (
    BertPreTrainedModel, BertModel,
    XLMPreTrainedModel, XLMModel,
)

from transformers.models.bert.modeling_bert import TokenClassifierOutput


def custom_mse(output, target, variances=None):
    if variances is None:
        return torch.mean((output - target)**2)
    return torch.mean(torch.mean((output - target)**2, dim=-1) * variances)


class ModelOutput(TokenClassifierOutput):
    def __init__(self, loss, logits=None, loss_float=None, hidden_states=None, attentions=None):
        super().__init__(loss, logits, hidden_states, attentions)
        self.loss_float = loss_float


class BertTokenClassifier(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        teacher_logits=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        samples_variances=None,
        original_lens=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        loss_float = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss_float = float(loss)

        elif teacher_logits is not None:
            #loss_fct = MSELoss(reduction="mean")
            loss_fct = custom_mse
            probs = torch.nn.functional.softmax(logits, dim=-1)
            src_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
            if attention_mask is not None:
                try:
                    active_loss = attention_mask.view(-1) == 1
                except:
                    active_loss = attention_mask.reshape(-1) == 1
                #inactive_subword = labels.view(-1) == loss_ignore_index
                #active_loss[inactive_subword] = 0

                #print("probs before view:",probs.size())

                probs = probs.view(-1, self.num_labels)[active_loss]

                #print("probs after view:",probs.size())

                src_probs = src_probs.view(-1, self.num_labels)[active_loss]

                if samples_variances is not None:
                    #print("vars before view:", samples_variances.size())
                    samples_variances = samples_variances.reshape(-1)[active_loss]
                    #print("vars after view:", samples_variances.size())

            loss = loss_fct(probs, src_probs, samples_variances)
            with torch.no_grad():
                loss_float = float(MSELoss(reduction="mean")(probs, src_probs))


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ModelOutput(
            loss=loss,
            logits=logits,
            loss_float=loss_float,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def __str__(self):
        return "BertTokenClassifier"

    def __repr__(self):
        return "BertTokenClassifier"




class XLMTokenClassifier(XLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = XLMModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        langs=None,
        token_type_ids=None,
        position_ids=None,
        lengths=None,
        cache=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        teacher_logits=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        original_lens=None
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            langs=langs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            lengths=lengths,
            cache=cache,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        elif teacher_logits is not None:
            loss_fct = MSELoss(reduction="mean")
            probs = torch.nn.functional.softmax(logits, dim=-1)
            src_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                #inactive_subword = labels.view(-1) == loss_ignore_index
                #active_loss[inactive_subword] = 0
                active_probs = probs.view(-1, self.num_labels)[active_loss]
                active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]

                loss = loss_fct(active_probs, active_src_probs)
            else:
                loss = loss_fct(probs, src_probs)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def __str__(self):
        return "XLMTokenClassifier"

    def __repr__(self):
        return "XLMTokenClassifier"
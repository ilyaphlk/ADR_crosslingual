import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import TokenClassifierOutput 


class BertStudent(BertPreTrainedModel):
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
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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

        loss_KD_fct = MSELoss(reduction="mean")
        probs = torch.nn.functional.softmax(logits, dim=-1)
        src_probs = torch.nn.functional.softmax(labels, dim=-1)
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            #inactive_subword = labels.view(-1) == loss_ignore_index
            #active_loss[inactive_subword] = 0
            active_probs = probs.view(-1, self.num_labels)[active_loss]
            active_src_probs = src_probs.view(-1, self.num_labels)[active_loss]

            loss_KD = loss_KD_fct(active_probs, active_src_probs)
        else:
            loss_KD = loss_KD_fct(probs, src_probs)


        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss_KD is not None else output

        return TokenClassifierOutput(
            loss=loss_KD,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
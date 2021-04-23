from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification, RobertaModel, RobertaConfig, RobertaPreTrainedModel
import torch
from torch import nn
import logging
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
import math

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "RobertaConfig"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "roberta-base",
    "roberta-large",
    "roberta-large-mnli",
    "distilroberta-base",
    "roberta-base-openai-detector",
    "roberta-large-openai-detector",
    # See all RoBERTa models at https://huggingface.co/models?filter=roberta
]

class TokenClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, features, **kwargs):
        pass

class ClassificationHead(nn.Module):
    def __init__(self, config, num_labels, hidden_size=-1):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size if hidden_size < 0 else hidden_size, num_labels)

    def forward(self, features, *kwargs):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x

class RepresentationProjectionLayer(nn.Module):
    def __init__(self, config, layer=-1, tokens=False, tagger=False, relations=False, num_attention_heads=-1):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if relations:
            self.dense = nn.Identity() #nn.Linear(num_attention_heads, num_attention_heads)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            
        self.layer_to_use = layer
        self.tokens = tokens
        self.tagger = tagger
        self.relations = relations
        self.hidden_size = config.hidden_size
        
        if num_attention_heads <= 0 and relations:
            raise Exception("Inconsistent configuration: num_attention_heads must be > 0 for relations")

        if relations:
            self.num_attention_heads = num_attention_heads
            self.attention_head_size = 64 #int(config.hidden_size / config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)

        if tokens and (tagger or relations):
            raise Exception('Inconsistent configuration: tokens cannot be true in tagger or relation mode')

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, features, event_tokens, **kwargs):
        seq_length = features[0].shape[1]
        if self.tokens:
            # grab the average over the tokens of the thing we want to classify
            # probably involved passing in some sub-sequence of interest so we know what tokens to grab,
            # then we average across those tokens.
            token_lens = event_tokens.sum(1)
            expanded_tokens = event_tokens.unsqueeze(2).expand(features[0].shape[0], seq_length, self.hidden_size)
            filtered_features = features[self.layer_to_use] * expanded_tokens
            x = filtered_features.sum(1) / token_lens.unsqueeze(1).expand(features[0].shape[0], self.hidden_size)
        elif self.tagger:
            x = features[self.layer_to_use]
        elif self.relations:
            # something like multi-headed attention but without the weighted sum at the end, so i get (num_heads) features for each of N x N grid, which feads into NxN softmax (with the same parameters)
            hidden_states = features[self.layer_to_use]
            key_layer = self.transpose_for_scores(self.key(hidden_states))   # Batch X num_heads X seq len X head_size
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # Batch X num_heads X seq_len X seq_len
            # Now we have num_heads features for each N X N relations.
            x = attention_scores / math.sqrt(self.attention_head_size)
            # move the 12 dimension to the end for easier classification
            x = x.permute(0, 2, 3, 1)

        else:
            # take <s> token (equiv. to [CLS])
            x = features[self.layer_to_use][:, 0, :]

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        return x


class CnlpRobertaForClassification(RobertaPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, num_labels_list=[], layer=-1, freeze=False, tokens=False, tagger=False, relations=False, num_attention_heads=12, class_weights=None, final_task_weight=1.0):
        super().__init__(config)
        self.num_labels = num_labels_list

        self.roberta = RobertaModel(config)
        
        if freeze:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        self.feature_extractors = nn.ModuleList()
        self.classifiers = nn.ModuleList()

        for task_ind,task_num_labels in enumerate(num_labels_list):
            self.feature_extractors.append(RepresentationProjectionLayer(config, layer=layer, tokens=tokens, tagger=tagger[task_ind], relations=relations[task_ind], num_attention_heads=num_attention_heads))
            if relations[task_ind]:
                self.classifiers.append(ClassificationHead(config, task_num_labels, hidden_size=num_attention_heads))
            else:
                self.classifiers.append(ClassificationHead(config, task_num_labels))

        # Are we operating as a sequence classifier (1 label per input sequence) or a tagger (1 label per input token in the sequence)
        self.tagger = tagger
        self.relations = relations

        if class_weights is None:
            self.class_weights = [None] * len(self.classifiers)
        else:
            self.class_weights = class_weights

        self.final_task_weight = final_task_weight
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
        event_tokens=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )
        
        batch_size,seq_len = input_ids.shape

        logits = []

        loss = None
        task_label_ind = 0

        for task_ind,task_num_labels in enumerate(self.num_labels):
            features = self.feature_extractors[task_ind](outputs.hidden_states, event_tokens)
            task_logits = self.classifiers[task_ind](features)
            logits.append(task_logits)
            
            if labels is not None:
                if task_num_labels == 1:
                    #  We are doing regression
                    loss_fct = MSELoss()
                    task_loss = loss_fct(task_logits.view(-1), labels.view(-1))
                else:
                    if not self.class_weights[task_ind] is None:
                        class_weights = torch.FloatTensor(self.class_weights[task_ind]).to(self.device)
                    else:
                        class_weights = None
                    loss_fct = CrossEntropyLoss(weight=class_weights)
                    
                    if self.relations[task_ind]:
                        task_labels = labels[:,0,task_label_ind:task_label_ind+seq_len, :]
                        task_label_ind += seq_len
                        task_loss = loss_fct(task_logits.permute(0, 3, 1, 2), task_labels.type(torch.LongTensor).to(labels.device))
                    else:
                        task_labels = labels[:,0, task_label_ind,:]
                        task_label_ind += 1
                        task_loss = loss_fct(task_logits.view(-1, task_num_labels), task_labels.reshape([batch_size*seq_len,]).type(torch.LongTensor).to(labels.device))
            
                if loss is None:
                    loss = task_loss
                else:
                    task_weight = 1.0 if task_ind+1 < len(self.num_labels) else self.final_task_weight
                    loss += (task_weight * task_loss)

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

        if self.training:
            return SequenceClassifierOutput(
                loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(loss=loss, logits=logits)

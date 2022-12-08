"""
Module containing the CNLP transformer model.
"""
# from transformers.models.auto import  AutoModel, AutoConfig
import copy
import inspect
from typing import Optional, List, Any, Dict

from transformers import AutoModel, AutoConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

import torch
from torch import nn
import logging
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn.functional import softmax, relu
import math

logger = logging.getLogger(__name__)


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
    def __init__(self, config, layer=10, tokens=False, tagger=False, relations=False, num_attention_heads=-1, head_size=64):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if relations:
            self.dense = nn.Identity()
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
            self.attention_head_size = head_size
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
            x = features[self.layer_to_use][..., 0, :]

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        return x


class CnlpConfig(PretrainedConfig):
    """
    The config class for :class:`CnlpModelForClassification`.

    :param encoder_name: the encoder name to use with :meth:`transformers.AutoConfig.from_pretrained`
    :param typing.Optional[str] finetuning_task: the tasks for which this model is fine-tuned
    :param typing.List[int] num_labels_list: the number of labels for each task
    :param int layer: the index of the encoder layer to extract features from
    :param bool tokens: if true, sentence-level classification is done based on averaged token embeddings for token(s) surrounded by <e> </e> special tokens
    :param int num_rel_attention_heads: the number of features/attention heads to use in the NxN relation classifier
    :param int rel_attention_head_dims: the number of parameters in each attention head in the NxN relation classifier
    :param typing.List[bool] tagger: for each task, whether the task is a sequence tagging task
    :param typing.List[bool] relations: for each task, whether the task is a relation extraction task
    :param bool use_prior_tasks: whether to use the outputs from the previous tasks as additional inputs for subsequent tasks
    :param \**kwargs: arguments for :class:`transformers.PretrainedConfig`
    """
    model_type='cnlpt'

    def __init__(
        self,
        encoder_name='roberta-base',
        finetuning_task=None,
        num_labels_list=[],
        layer=-1,
        tokens=False,
        num_rel_attention_heads=12,
        rel_attention_head_dims=64,
        tagger = [False],
        relations = [False],
        use_prior_tasks=False,
        **kwargs
     ):
        super().__init__(**kwargs)
        # self.name_or_path='cnlpt'
        self.finetuning_task = finetuning_task
        self.num_labels_list = num_labels_list
        self.layer = layer
        self.tokens = tokens
        self.num_rel_attention_heads = num_rel_attention_heads
        self.rel_attention_head_dims = rel_attention_head_dims
        self.tagger = tagger
        self.relations = relations
        self.use_prior_tasks = use_prior_tasks
        self.encoder_name = encoder_name
        self.encoder_config = AutoConfig.from_pretrained(encoder_name).to_dict()
        if encoder_name.startswith('distilbert'):
            self.hidden_dropout_prob = self.encoder_config['dropout']
            self.hidden_size = self.encoder_config['dim']
        else:
            try:
                self.hidden_dropout_prob = self.encoder_config['hidden_dropout_prob']
                self.hidden_size = self.encoder_config['hidden_size']
            except KeyError as ke:
                raise ValueError(f'Encoder config does not have an attribute'
                                 f' "{ke.args[0]}"; this is likely because the API of'
                                 f' the chosen encoder differs from the BERT/RoBERTa'
                                 f' API and the DistilBERT API. Encoders with different'
                                 f' APIs are not yet supported (#35).')


class CnlpModelForClassification(PreTrainedModel):
    """
    The CNLP transformer model.

    :param typing.Optional[typing.List[float]] class_weights: if provided,
        the weights to use for each task when computing the loss
    :param float final_task_weight: the weight to use for the final task
        when computing the loss; default 1.0.
    :param float argument_regularization: if provided, the argument
        regularization to use when computing the loss
    :param bool freeze: whether to freeze the weights of the encoder
    :param bool bias_fit: whether to fine-tune only the bias of the encoder
    """
    base_model_prefix = 'cnlpt'
    config_class = CnlpConfig

    def __init__(self,
                 config: config_class,
                 *,
                 class_weights: Optional[List[float]] = None,
                 final_task_weight: float = 1.0,
                 argument_regularization: float = -1,
                 freeze=False,
                 bias_fit=False,
                 ):

        super().__init__(config)

        encoder_config = AutoConfig.from_pretrained(config.encoder_name)
        encoder_config.vocab_size = config.vocab_size
        config.encoder_config = encoder_config.to_dict()
        encoder_model = AutoModel.from_config(encoder_config)
        self.encoder = encoder_model.from_pretrained(config.encoder_name)
        self.encoder.resize_token_embeddings(encoder_config.vocab_size)

        if config.layer > len(encoder_model.encoder.layer):
            raise ValueError('The layer specified (%d) is too big for the specified encoder which has %d layers' % (
                config.layer,
                len(encoder_model.encoder.layer)
            ))
        self.num_labels = config.num_labels_list
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if bias_fit:
            for name, param in self.encoder.named_parameters():
                if not 'bias' in name:
                    param.requires_grad = False

        self.feature_extractors = nn.ModuleList()
        self.logit_projectors = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        total_prev_task_labels = 0
        for task_ind,task_num_labels in enumerate(self.num_labels):
            self.feature_extractors.append(RepresentationProjectionLayer(config, layer=config.layer, tokens=config.tokens, tagger=config.tagger[task_ind], relations=config.relations[task_ind], num_attention_heads=config.num_rel_attention_heads, head_size=config.rel_attention_head_dims))
            if config.relations[task_ind]:
                hidden_size = config.num_rel_attention_heads
                if config.use_prior_tasks:
                    hidden_size += total_prev_task_labels

                self.classifiers.append(ClassificationHead(config, task_num_labels, hidden_size=hidden_size))
            else:
                self.classifiers.append(ClassificationHead(config, task_num_labels))
            total_prev_task_labels += task_num_labels

        # Are we operating as a sequence classifier (1 label per input sequence) or a tagger (1 label per input token in the sequence)
        self.tagger = config.tagger
        self.relations = config.relations

        if class_weights is None:
            self.class_weights = [None] * len(self.classifiers)
        else:
            self.class_weights = class_weights

        self.final_task_weight = final_task_weight
        self.use_prior_tasks = config.use_prior_tasks
        self.argument_regularization = argument_regularization
        self.reg_temperature = 1.0

        # self.init_weights()

    def predict_relations_with_previous_logits(self, features, logits):
        seq_len = features.shape[1]
        for prior_task_logits in logits:
            if len(features.shape) == 4:
                # relations - batch x len x len x dim
                if len(prior_task_logits.shape) == 3:
                    # prior task is sequence tagging:
                    # we have batch x len x num_classes.
                    # we want to concatenate the num_classes to the variables at each element of the sequence,
                    # but then need to broadcast it down all the rows of the matrix.
                    aug = prior_task_logits.unsqueeze(2) # add another dimension to repeat along
                    aug = aug.repeat(1, 1, seq_len, 1) # repeat along the new empty dimension so we have our seq logits repeated seq_len x seq_len
                    features = torch.cat( (features, aug), 3) # concatenate the  relation matrix with the sequence matrix
                else:
                    logging.warn("It is not implemented to add a task of shape %s to a relation matrix" % ( str(prior_task_logits.shape)))
            elif len(features.shape) == 3:
                # sequence
                logging.warn("It is not implemented to add previous task of any type to a sequence task")

        return features

    def compute_loss(self,
                     task_logits: torch.FloatTensor,
                     labels: torch.LongTensor,
                     task_ind,
                     task_num_labels,
                     batch_size,
                     seq_len,
                     state: dict,
                     ) -> None:
        """
        Computes the loss for a single batch and a single task.

        Args:
            task_logits:
            labels:
            task_ind:
            task_num_labels:
            batch_size:
            seq_len:
            state:
        :meta private:
        """
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
                task_labels = labels[:, :, state['task_label_ind']:state['task_label_ind'] + seq_len]
                state['task_label_ind'] += seq_len
                task_loss = loss_fct(task_logits.permute(0, 3, 1, 2),
                                     task_labels.type(torch.LongTensor).to(labels.device))
            elif self.tagger[task_ind]:
                # in cases where we are only given a single task the HF code will have one fewer dimension in the labels, so just add a dummy dimension to make our indexing work:
                if labels.ndim == 2:
                    task_labels = labels
                elif labels.ndim == 3:
                    # labels = labels.unsqueeze(1)
                    task_labels = labels[:, :, state['task_label_ind']]
                else:
                    task_labels = labels[:, 0, state['task_label_ind'], :]

                state['task_label_ind'] += 1
                task_loss = loss_fct(task_logits.view(-1, task_num_labels),
                                     task_labels.reshape([batch_size * seq_len, ]).type(torch.LongTensor).to(
                                         labels.device))
            else:
                if labels.ndim == 1:
                    task_labels = labels
                elif labels.ndim == 2:
                    task_labels = labels[:, task_ind]
                elif labels.ndim == 3:
                    task_labels = labels[:, 0, task_ind]
                else:
                    raise NotImplementedError(
                        'Have not implemented the case where a classification task '
                        'is part of an MTL setup with relations and sequence tagging')

                state['task_label_ind'] += 1
                task_loss = loss_fct(task_logits, task_labels.type(torch.LongTensor).to(labels.device))

        if state['loss'] is None:
            state['loss'] = task_loss
        else:
            task_weight = 1.0 if task_ind + 1 < len(self.num_labels) else self.final_task_weight
            state['loss'] += (task_weight * task_loss)

    def apply_arg_reg(self,
                      logits: List[torch.FloatTensor],
                      attention_mask: torch.LongTensor,
                      state: dict) -> None:
        """
        Applies argument regularization to the logits for a single batch on all tasks.

        Args:
            logits (`List[torch.FloatTensor]`): the computed logits of the batch.
            attention_mask (`torch.LongTensor`): the attention mask for the batch.
            state: the state dict containing the loss for the batch.
        :meta private:
        """
        # standard e2e relation task -- two entity extractors and relation extractor.
        prob_no_rel = softmax(logits[2], dim=3)[:, :, :, 0]

        ## product gets us something like a joint probability over all relation categories.
        # the downside is, we're penalizing the event "some relation being more likely than none"
        # in the joint sense, but we never actually use that event anywhere, i.e., if no
        # relation meets the threshold we will never create a relation.
        # so maybe doing something like relu + sum makes more sense.
        #
        # prob_a1_norel = prob_no_rel.prod(dim=1)
        # prob_a2_norel = prob_no_rel.prod(dim=2)
        # prob_some_rel = relu ( 1 - (prob_a1_norel * prob_a2_norel) - 0.5)
        # These values will be greater than 0 at position i if there is any relation that
        # has i as arg1 or i as arg2.
        prob_a1_rel = relu(0.5 - prob_no_rel).sum(dim=1)
        prob_a2_rel = relu(0.5 - prob_no_rel).sum(dim=2)
        prob_some_rel = prob_a1_rel + prob_a2_rel

        # prob_no_e1_type = relu( softmax(logits[0], dim=2)[:,:,0] - 0.5)
        # prob_no_e2_type = relu( softmax(logits[1], dim=2)[:,:,0] - 0.5)
        probs_e1 = softmax(logits[0], dim=2)
        probs_e2 = softmax(logits[1], dim=2)

        # threshold: the penalty is possible if more than this number of probabilities are greater than the
        # "no entity" threshold. we subtract 2 because we are removing the None category, and C-1 is the default
        # case where there is no relation, only if more than that is an issue.
        t1_threshold = self.num_labels[0] - 2
        t2_threshold = self.num_labels[1] - 2

        # we take p(none) - p(other relations) inside the sign.
        # then take the sign, if there are any -1s, the sum will be less than tx_threshold and the inner part will be < 0,
        # and relu will be 0. If there are no -1, the sum will be > tx_threshold and the innter part will be 1, relu also 1.
        prob_no_e1_type = relu(
            torch.sign(probs_e1[:, :, 0].unsqueeze(2) - probs_e1[:, :, 1:]).sum(dim=2) - t1_threshold)
        prob_no_e2_type = relu(
            torch.sign(probs_e2[:, :, 0].unsqueeze(2) - probs_e2[:, :, 1:]).sum(dim=2) - t2_threshold)

        prob_no_ent = prob_no_e1_type * prob_no_e2_type

        prob_rel_no_ent = prob_some_rel * prob_no_ent * attention_mask

        state["loss"] += self.argument_regularization * prob_rel_no_ent.sum()

    def generalize_encoder_forward_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        new_kwargs = dict()
        params = inspect.signature(self.encoder.forward).parameters
        for name, value in kwargs.items():
            if name not in params and value is not None:
                # Warn if a contentful parameter is not valid
                logger.warning(f"Parameter {name} not present for encoder class {self.encoder.__class__.__name__}.")
            elif name in params:
                # Pass all, and only, parameters that are valid,
                # regardless of whether they are None
                new_kwargs[name] = value
            # else, value is None and not in params, so we ignore it
        return new_kwargs

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
        Forward method.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_len)`, *optional*):
                A batch of chunked documents as tokenizer indices.
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_len)`, *optional*):
                Attention masks for the batch.
            token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_len)`, *optional*):
                Token type IDs for the batch.
            position_ids: (`torch.LongTensor` of shape `(batch_size, sequence_len)`, *optional*):
                Position IDs for the batch.
            head_mask (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Token encoder head mask.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_len, hidden_size)`, *optional*):
                A batch of chunked documents as token embeddings.
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss.
                Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
                If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
                If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            output_attentions (`bool`, *optional*): Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states: not used.
            event_tokens: a mask defining which tokens in the input are to be averaged for input to classifier head; only used when self.tokens==True.

        Returns: (`transformers.SequenceClassifierOutput`) the output of the model
        """

        kwargs = self.generalize_encoder_forward_kwargs(
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )

        outputs = self.encoder(
            input_ids,
            **kwargs
        )
        
        batch_size,seq_len = input_ids.shape

        logits = []

        state = dict(
            loss=None,
            task_label_ind=0
        )

        for task_ind,task_num_labels in enumerate(self.num_labels):
            features = self.feature_extractors[task_ind](outputs.hidden_states, event_tokens)
            if self.use_prior_tasks:
                # note: this specific way of incorporating previous logits doesn't help in my experiments with thyme/clinical tempeval
                if self.relations[task_ind]:
                    features = self.predict_relations_with_previous_logits(features, logits)
            task_logits = self.classifiers[task_ind](features)
            logits.append(task_logits)

            if labels is not None:
                self.compute_loss(
                    task_logits,
                    labels,
                    task_ind,
                    task_num_labels,
                    batch_size,
                    seq_len,
                    state
                )

        if len(self.num_labels) == 3 and self.relations[-1] and self.argument_regularization > 0:
            self.apply_arg_reg(logits, attention_mask, state)

        if self.training:
            return SequenceClassifierOutput(
                loss=state['loss'], logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(loss=state['loss'], logits=logits, attentions=outputs.attentions)

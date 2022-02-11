import logging
import time
import copy
import tempfile
import pickle
import shutil
import os
import random

import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.models.distilbert import DistilBertPreTrainedModel, DistilBertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import f1_score, roc_auc_score
# import wandb

# from transformer import EncoderLayer
# from utils import set_seed
from src.cnlpt.CnlpModelForClassification import CnlpModelForClassification

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)

        output = self.w_2(F.relu(self.w_1(x)))
        output = self.dropout(output)
        output += residual

        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class HierarchicalTransformerConfig(object):
    def __init__(self,
                 n_layers,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout


class HierarchicalModel(CnlpModelForClassification):
    base_model_prefix = 'hier'

    def __init__(self,
                 config,
                 transformer_head_config,
                 *,
                 class_weights=None,
                 final_task_weight=1.0,
                 argument_regularization=-1,
                 freeze=False,
                 ):
        super(HierarchicalModel, self).__init__(config,
                                                class_weights=class_weights,
                                                final_task_weight=final_task_weight,
                                                argument_regularization=argument_regularization,
                                                freeze=freeze,
                                                )
        # Transformer layer
        transformer_layer = EncoderLayer(d_model=transformer_head_config.d_model,
                                         d_inner=transformer_head_config.d_inner,
                                         n_head=transformer_head_config.n_head,
                                         d_k=transformer_head_config.d_k,
                                         d_v=transformer_head_config.d_v,
                                         dropout=transformer_head_config.dropout)
        self.transformer = nn.ModuleList(
            [copy.deepcopy(transformer_layer) for _ in range(transformer_head_config.n_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor =None,
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
        batch_size, num_chunks, chunk_len = input_ids.shape

        flat_shape = (batch_size * num_chunks, chunk_len)

        outputs = self.encoder(
            input_ids.reshape(flat_shape),
            attention_mask=attention_mask.reshape(flat_shape) if attention_mask is not None else None,
            token_type_ids=token_type_ids.reshape(flat_shape) if token_type_ids is not None else None,
            position_ids=position_ids.reshape(flat_shape) if position_ids is not None else None,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )

        logits = []

        state = dict(
            loss=None,
            task_label_ind=0
        )

        for task_ind, task_num_labels in enumerate(self.num_labels):
            if self.use_prior_tasks:
                raise NotImplementedError('use_prior_tasks is not defined for hierarchical model')
            if self.config.tokens:
                raise NotImplementedError('tokens projection is not defined for hierarchical model')
            if self.config.tagger[task_ind]:
                raise NotImplementedError('tagger projection is not defined for hierarchical model')
            if self.config.relations[task_ind]:
                raise NotImplementedError('relations projection is not defined for hierarchical model')

            # outputs.last_hidden_state.shape: (B * n_chunks, chunk_len, hidden_size)

            # (B * n_chunk, hidden_size)
            chunks_reps = self.feature_extractors[task_ind](outputs.hidden_states, event_tokens)

            # (B, n_chunk, hidden_size)
            chunks_reps = chunks_reps.reshape(batch_size, num_chunks, chunks_reps.shape[-1])

            # Use pre-trained model's position embedding
            position_ids = torch.arange(num_chunks, dtype=torch.long,
                                        device=chunks_reps.device)  # (n_chunk)
            position_ids = position_ids.unsqueeze(0).expand_as(chunks_reps[:, :, 0])  # (B, n_chunk)
            position_embeddings = self.encoder.embeddings.position_embeddings(position_ids)
            chunks_reps += position_embeddings

            # document encoding (B, n_chunk, hidden_size)
            for layer_module in self.transformer:
                chunks_reps, _ = layer_module(chunks_reps)

            # extract first Documents as rep. (B, hidden_size)
            doc_rep = chunks_reps[:, 0, :]

            # predict (B, 5)
            task_logits = self.classifiers[task_ind](doc_rep)
            logits.append(task_logits)

            if labels is not None:
                self.compute_loss(
                    task_logits,
                    labels,
                    task_ind,
                    task_num_labels,
                    batch_size,
                    -1,  # only used for relation adn tagger
                    state
                )

        if len(self.num_labels) == 3 and self.relations[-1] and self.argument_regularization > 0:
            # standard e2e relation task -- two entity extractors and relation extractor.
            prob_no_rel = F.softmax(logits[2], dim=3)[:, :, :, 0]

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
            prob_a1_rel = F.relu(0.5 - prob_no_rel).sum(dim=1)
            prob_a2_rel = F.relu(0.5 - prob_no_rel).sum(dim=2)
            prob_some_rel = prob_a1_rel + prob_a2_rel

            # prob_no_e1_type = relu( softmax(logits[0], dim=2)[:,:,0] - 0.5)
            # prob_no_e2_type = relu( softmax(logits[1], dim=2)[:,:,0] - 0.5)
            probs_e1 = F.softmax(logits[0], dim=2)
            probs_e2 = F.softmax(logits[1], dim=2)

            # threshold: the penalty is possible if more than this number of probabilities are greater than the
            # "no entity" threshold. we subtract 2 because we are removing the None category, and C-1 is the default
            # case where there is no relation, only if more than that is an issue.
            t1_threshold = self.num_labels[0] - 2
            t2_threshold = self.num_labels[1] - 2

            # we take p(none) - p(other relations) inside the sign.
            # then take the sign, if there are any -1s, the sum will be less than tx_threshold and the inner part will be < 0,
            # and relu will be 0. If there are no -1, the sum will be > tx_threshold and the innter part will be 1, relu also 1.
            prob_no_e1_type = F.relu(
                torch.sign(probs_e1[:, :, 0].unsqueeze(2) - probs_e1[:, :, 1:]).sum(dim=2) - t1_threshold)
            prob_no_e2_type = F.relu(
                torch.sign(probs_e2[:, :, 0].unsqueeze(2) - probs_e2[:, :, 1:]).sum(dim=2) - t2_threshold)

            prob_no_ent = prob_no_e1_type * prob_no_e2_type

            prob_rel_no_ent = prob_some_rel * prob_no_ent * attention_mask

            state['loss'] += self.argument_regularization * prob_rel_no_ent.sum()

        if self.training:
            return SequenceClassifierOutput(
                loss=state['loss'], logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(loss=state['loss'], logits=logits)

"""
Module containing the Hierarchical Transformer module, adapted from Xin Su.
"""
import logging
import copy
import random
from typing import Optional, List, cast

import numpy as np
from torch import nn
import torch.nn.functional as F
import torch
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers import AutoModel, AutoConfig

from .CnlpModelForClassification import CnlpConfig, ClassificationHead, generalize_encoder_forward_kwargs, freeze_encoder_weights

logger = logging.getLogger(__name__)


def set_seed(seed, n_gpu):
    """
    Set the random seeds for ``random``, numpy, and pytorch to a specific value.

    Args:
        seed: the seed to use
        n_gpu: the number of GPUs being used
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module

    Original author: Yu-Hsiang Huang (https://github.com/jadore801120/attention-is-all-you-need-pytorch)

    Args:
        n_head: the number of attention heads
        d_model: the dimensionality of the input and output of the encoder
        d_k: the size of the query and key vectors
        d_v: the size of the value vector
    """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

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
    """
    A two-feed-forward-layer module

    Original author: Yu-Hsiang Huang (https://github.com/jadore801120/attention-is-all-you-need-pytorch)

    Args:
        d_in: the dimensionality of the input and output of the encoder
        d_hid: the inner hidden size of the positionwise FFN in the encoder
        dropout: the amount of dropout to use in training (default 0.1)
    """

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
    """
    Scaled Dot-Product Attention

    Original author: Yu-Hsiang Huang (https://github.com/jadore801120/attention-is-all-you-need-pytorch)

    Args:
        temperature: the temperature for scaled dot product attention
        attn_dropout: the amount of dropout to use in training
          for scaled dot product attention (default 0.1, not
          tuned in the rest of the code)
    """

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
    """
    Compose with two layers

    Original author: Yu-Hsiang Huang (https://github.com/jadore801120/attention-is-all-you-need-pytorch)

    Args:
        d_model: the dimensionality of the input and output of the encoder
        d_inner: the inner hidden size of the positionwise FFN in the encoder
        n_head: the number of attention heads
        d_k: the size of the query and key vectors
        d_v: the size of the value vector
        dropout: the amount of dropout to use in training in both the
          attention and FFN steps (default 0.1)
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class HierarchicalModel(PreTrainedModel):
    """
    Hierarchical Transformer model (https://arxiv.org/abs/2105.06752)

    Adapted from Xin Su's implementation (https://github.com/xinsu626/DocTransformer)

    Args:
        config:
        transformer_head_config:
        class_weights:
        final_task_weight:
        freeze:
    """
    base_model_prefix = "hier"
    config_class = CnlpConfig

    def __init__(
        self,
        config: config_class,
        *,
        freeze: float = -1.0,
        class_weights: Optional[List[float]] = None,
    ):
        # Initialize common components
        super(HierarchicalModel, self).__init__(
            config,
        )

        self.config = cast(CnlpConfig, self.config)  # for PyCharm

        assert self.config.hier_head_config is not None, "Hierarchical model is being instantiated with no hierarchical head config"

        encoder_config = AutoConfig.from_pretrained(self.config.encoder_name)
        encoder_config.vocab_size = self.config.vocab_size
        self.config.encoder_config = encoder_config.to_dict()
        encoder_model = AutoModel.from_config(encoder_config)
        self.encoder = encoder_model.from_pretrained(self.config.encoder_name)
        self.encoder.resize_token_embeddings(encoder_config.vocab_size)

        if self.config.layer > self.config.hier_head_config["n_layers"]:
            raise ValueError('The layer specified (%d) is too big for the specified chunk transformer which has %d layers' % (
                self.config.layer,
                self.config.hier_head_config["n_layers"]
            ))
        self.layer = self.config.layer

        if freeze > 0:
            freeze_encoder_weights(self.encoder, freeze)

        self.num_labels = self.config.num_labels_list

        # Document-level transformer layer
        transformer_layer = EncoderLayer(
            d_model=self.config.hidden_size,
            d_inner=self.config.hier_head_config["d_inner"],
            n_head=self.config.hier_head_config["n_head"],
            d_k=self.config.hier_head_config["d_k"],
            d_v=self.config.hier_head_config["d_v"],
            dropout=self.config.hier_head_config["dropout"],
        )
        self.transformer = nn.ModuleList(
            [
                copy.deepcopy(transformer_layer)
                for _ in range(self.config.hier_head_config["n_layers"])
            ]
        )

        self.classifiers = nn.ModuleList()
        for task_num_labels in self.num_labels:
            self.classifiers.append(ClassificationHead(self.config, task_num_labels))

        if class_weights is None:
            self.class_weights = [None] * len(self.classifiers)
        else:
            self.class_weights = class_weights

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
        output_hidden_states=False,
        event_tokens=None,
    ):
        """
        Forward method.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, num_chunks, chunk_len)`, *optional*):
                A batch of chunked documents as tokenizer indices.
            attention_mask (`torch.LongTensor` of shape `(batch_size, num_chunks, chunk_len)`, *optional*):
                Attention masks for the batch.
            token_type_ids (`torch.LongTensor` of shape `(batch_size, num_chunks, chunk_len)`, *optional*):
                Token type IDs for the batch.
            position_ids: (`torch.LongTensor` of shape `(batch_size, num_chunks, chunk_len)`, *optional*):
                Position IDs for the batch.
            head_mask (`torch.LongTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Token encoder head mask.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_chunks, chunk_len, hidden_size)`, *optional*):
                A batch of chunked documents as token embeddings.
            labels (`torch.LongTensor` of shape `(batch_size, num_tasks)`, *optional*):
                Labels for computing the sequence classification/regression loss.
                Indices should be in `[0, ..., self.num_labels[task_ind] - 1]`.
                If `self.num_labels[task_ind] == 1` a regression loss is computed (Mean-Square loss),
                If `self.num_labels[task_ind] > 1` a classification loss is computed (Cross-Entropy).
            output_attentions (`bool`, *optional*): Whether or not to return the attentions tensors of all attention layers.
            output_hidden_states: If True, return a matrix of shape (batch_size, num_chunks, hidden size) representing the contextualized embeddings of each chunk. The 0-th element of each chunk is the classifier representation for that instance.
            event_tokens: not currently used (only relevant for token classification)

        Returns:

        """
        if input_ids is not None:
            batch_size, num_chunks, chunk_len = input_ids.shape
            flat_shape = (batch_size * num_chunks, chunk_len)
        else:  # inputs_embeds is not None
            batch_size, num_chunks, chunk_len, embed_dim = inputs_embeds.shape
            flat_shape = (batch_size * num_chunks, chunk_len, embed_dim)

        kwargs = generalize_encoder_forward_kwargs(
            self.encoder,
            attention_mask=attention_mask.reshape(flat_shape[:3])
            if attention_mask is not None
            else None,
            token_type_ids=token_type_ids.reshape(flat_shape[:3])
            if token_type_ids is not None
            else None,
            position_ids=position_ids.reshape(flat_shape[:3])
            if position_ids is not None
            else None,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds.reshape(flat_shape)
            if inputs_embeds is not None
            else None,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        outputs = self.encoder(
            input_ids.reshape(flat_shape[:3])
            if input_ids is not None
            else None,
            **kwargs
        )

        logits = []
        hidden_states = None

        # outputs.last_hidden_state.shape: (B * n_chunks, chunk_len, hidden_size)
        # (B * n_chunk, hidden_size)
        chunks_reps = outputs.last_hidden_state[...,0,:].reshape(batch_size, num_chunks, outputs.last_hidden_state.shape[-1])

        # Use pre-trained model's position embedding
        position_ids = torch.arange(
            num_chunks, dtype=torch.long, device=chunks_reps.device
        )  # (n_chunk)
        position_ids = position_ids.unsqueeze(0).expand_as(
            chunks_reps[:, :, 0]
        )  # (B, n_chunk)
        position_embeddings = self.encoder.embeddings.position_embeddings(
            position_ids
        )
        chunks_reps = chunks_reps + position_embeddings

        # document encoding (B, n_chunk, hidden_size)
        for layer_ind, layer_module in enumerate(self.transformer):
            chunks_reps, _ = layer_module(chunks_reps)

            ## this case is mainly for when we are doing subsequent fine-tuning using a pre-trained
            ## hierarchical model and we want to check whether an earlier layer might provide better
            ## classification performance (e.g., if we think the last layer(s) are overfit to the pre-training
            ## objective) Just short circuit rather than doing the whole computation.
            if layer_ind+1 >= self.layer:
                break

        if output_hidden_states:
            hidden_states = chunks_reps

        # extract first Documents as rep. (B, hidden_size)
        doc_rep = chunks_reps[:, 0, :]

        total_loss = 0
        for task_ind, task_num_labels in enumerate(self.num_labels):
            if not self.class_weights[task_ind] is None:
                class_weights = torch.FloatTensor(self.class_weights[task_ind]).to(self.device)
            else:
                class_weights = None
            loss_fct = CrossEntropyLoss(weight=class_weights)

            # predict (B, 5)
            task_logits = self.classifiers[task_ind](doc_rep)
            logits.append(task_logits)

            if labels is not None:
                task_labels = labels[:, task_ind]
                task_loss = loss_fct(task_logits, task_labels.type(torch.LongTensor).to(labels.device))
                total_loss += task_loss
            

        if self.training:
            return SequenceClassifierOutput(
                loss=total_loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        else:
            return SequenceClassifierOutput(loss=total_loss, logits=logits, hidden_states=hidden_states)

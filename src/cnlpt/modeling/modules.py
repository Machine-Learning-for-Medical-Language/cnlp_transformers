from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn

from ..data.task_info import CLASSIFICATION, RELATIONS, TAGGING, TaskType


class ClassificationHead(nn.Module):
    """Generic classification head that can be used for any task."""

    def __init__(self, hidden_dropout_prob: float, hidden_size: int, num_labels: int):
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, *kwargs):
        x = self.dropout(features)
        x = self.out_proj(x)
        return x


class RepresentationProjectionLayer(nn.Module):
    """The class that maps from some output from a text encoder into a feature representation that can be classified.
    Project the representation to a new space depending on the task type, based on arguments passed in to the constructor."""

    def __init__(
        self,
        hidden_dropout_prob: float,
        hidden_size: int,
        layer: int = 10,
        tokens: bool = False,
        task_type: TaskType = CLASSIFICATION,
        num_attention_heads: int = -1,
        head_size: int = 64,
    ):
        """
        Args:
            config: The config file for the encoder
            layer: Which layer to pull the encoder representation from
            tokens: Whether to classify an entity based on the token reprsentation rather than the CLS representation
            tagger: Whether the current task is a token tagging task
            relations: Whether the current task is relation exttraction
            num_attention_heads: For relations, how many "features" to use
            head_size: For relations, how big each head should be
        """
        super().__init__()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        if task_type == RELATIONS:
            self.dense = nn.Identity()
        else:
            self.dense = nn.Linear(hidden_size, hidden_size)

        self.layer_to_use = layer
        self.tokens = tokens
        self.task_type = task_type
        self.hidden_size = hidden_size

        if num_attention_heads <= 0 and self.task_type == RELATIONS:
            raise Exception(
                "Inconsistent configuration: num_attention_heads must be > 0 for relations"
            )

        if self.task_type == RELATIONS:
            self.num_attention_heads = num_attention_heads
            self.attention_head_size = head_size
            self.all_head_size = self.num_attention_heads * self.attention_head_size

            self.query = nn.Linear(hidden_size, self.all_head_size)
            self.key = nn.Linear(hidden_size, self.all_head_size)

        if tokens and self.task_type in (TAGGING, RELATIONS):
            raise Exception(
                "Inconsistent configuration: tokens cannot be true in tagger or relation mode"
            )

    def transpose_for_scores(self, x):
        # x: (batch x seq x all_head)

        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        # (batch x seq x n_heads x head_size)
        x = x.view(*new_x_shape)

        # (batch x n_heads x seq x head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, features, event_tokens: torch.Tensor, **kwargs):
        # features: (layers x batch x seq x hidden)
        # event_tokens: (batch x seq)

        seq_length = features[0].shape[1]
        if self.tokens:
            # grab the average over the tokens of the thing we want to classify
            # probably involved passing in some sub-sequence of interest so we know what tokens to grab,
            # then we average across those tokens.

            # (batch)
            event_toks_per_seq = event_tokens.sum(1)

            # (batch x seq x hidden)
            expanded_tokens = event_tokens.unsqueeze(2).expand(
                features[0].shape[0], seq_length, self.hidden_size
            )

            # (batch x seq x hidden)
            filtered_features = features[self.layer_to_use] * expanded_tokens

            # (batch x hidden)
            x = filtered_features.sum(1) / event_toks_per_seq.unsqueeze(1).expand(
                features[0].shape[0], self.hidden_size
            )
        elif self.task_type == TAGGING:
            # (batch x seq x hidden)
            x = features[self.layer_to_use]
        elif self.task_type == RELATIONS:
            # something like multi-headed attention but without the weighted sum at the end, so i get (num_heads) features for each of N x N grid, which feads into NxN softmax (with the same parameters)
            # (batch x seq x hidden)
            hidden_states = features[self.layer_to_use]

            # (batch x n_heads x seq x head_size)
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            # (batch x n_heads x seq x head_size)
            query_layer = self.transpose_for_scores(self.query(hidden_states))

            # (batch x n_heads x seq x seq)
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            # Now we have num_heads features for each N X N relations.
            x = attention_scores / math.sqrt(self.attention_head_size)
            # move the 12 dimension to the end for easier classification

            # (batch x seq x seq x n_heads)
            x = x.permute(0, 2, 3, 1)

        else:
            # take <s> token (equiv. to [CLS])
            # (batch x hidden)
            x = features[self.layer_to_use][..., 0, :]

        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)

        # for classification (including event tokens mode): (batch x hidden)
        # for tagging: (batch x seq x hidden)
        # for relations: (batch x seq x seq x n_heads)
        return x


### MODULES FOR HIERARCHICAL MODEL ###


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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn: torch.Tensor = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


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

    def __init__(
        self,
        n_head: int,
        d_model: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
    ):
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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Union[torch.Tensor, None] = None,
    ):
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

        output: torch.Tensor
        attn: torch.Tensor
        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

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
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
        self, enc_input: torch.Tensor, slf_attn_mask: Union[torch.Tensor, None] = None
    ):
        enc_output: torch.Tensor
        enc_slf_attn: torch.Tensor
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask
        )
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

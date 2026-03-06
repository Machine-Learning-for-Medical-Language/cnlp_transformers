"""
Module containing the Hierarchical Transformer module, adapted from Xin Su.
"""

import logging
from dataclasses import dataclass
from typing import Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from ...data.task_info import TaskInfo
from ..config.hierarchical_config import HierarchicalModelConfig
from ..modules import ClassificationHead, EncoderLayer
from ..utils import freeze_encoder_weights, generalize_encoder_forward_kwargs

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalSequenceClassifierOutput(SequenceClassifierOutput):
    chunk_attentions: Union[tuple[torch.FloatTensor], None] = None


class HierarchicalModel(PreTrainedModel):
    """
    Hierarchical Transformer model (https://arxiv.org/abs/2105.06752)

    Adapted from Xin Su's implementation (https://github.com/xinsu626/DocTransformer)
    """

    base_model_prefix = "cnlpt.hier"
    config_class = HierarchicalModelConfig

    def __init__(
        self,
        config: HierarchicalModelConfig,
        *,
        freeze: float = -1.0,
        class_weights: Union[dict[str, torch.FloatTensor], None] = None,
        **kwargs,
    ):
        super().__init__(config)
        self.config: HierarchicalModelConfig

        self.encoder = self.config.load_encoder_model(True)

        if freeze > 0:
            freeze_encoder_weights(self.encoder, freeze)

        # Document-level transformer layer
        self.transformer: list[EncoderLayer] = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=self.config.encoder_output_dim,
                    d_inner=self.config.d_inner,
                    n_head=self.config.n_head,
                    d_k=self.config.d_k,
                    d_v=self.config.d_v,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.n_layers)
            ]
        )

        self.configure_for_tasks(self.config.tasks, class_weights)

    def configure_for_tasks(
        self,
        tasks: list[TaskInfo],
        class_weights: Union[dict[str, torch.FloatTensor], None],
    ):
        self.tasks = self.config.tasks = tasks
        self.classifiers = nn.ModuleDict()

        for task in self.tasks:
            self.classifiers[task.name] = ClassificationHead(
                hidden_dropout_prob=self.config.dropout,
                hidden_size=self.config.encoder_output_dim,
                num_labels=len(task.labels),
            )

        self.class_weights = class_weights

    def forward(
        self,
        input_ids: Union[torch.LongTensor, None] = None,
        attention_mask: Union[torch.LongTensor, None] = None,
        token_type_ids: Union[torch.LongTensor, None] = None,
        position_ids: Union[torch.LongTensor, None] = None,
        head_mask: Union[torch.LongTensor, None] = None,
        inputs_embeds: Union[torch.FloatTensor, None] = None,
        labels: Union[torch.LongTensor, None] = None,
        output_attentions: Union[bool, None] = None,
        **kwargs,
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

        encoder_kwargs = generalize_encoder_forward_kwargs(
            self.encoder,
            attention_mask=(
                attention_mask.reshape(flat_shape[:3])
                if attention_mask is not None
                else None
            ),
            token_type_ids=(
                token_type_ids.reshape(flat_shape[:3])
                if token_type_ids is not None
                else None
            ),
            position_ids=(
                position_ids.reshape(flat_shape[:3])
                if position_ids is not None
                else None
            ),
            head_mask=head_mask,
            inputs_embeds=(
                inputs_embeds.reshape(flat_shape) if inputs_embeds is not None else None
            ),
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        outputs: BaseModelOutput = self.encoder(
            input_ids.reshape(flat_shape[:3]) if input_ids is not None else None,
            **encoder_kwargs,
        )

        logits = []

        # outputs.last_hidden_state.shape: (B * n_chunks, chunk_len, hidden_size)
        # (B * n_chunk, hidden_size)
        chunks_reps = outputs.last_hidden_state[..., 0, :].reshape(
            batch_size, num_chunks, outputs.last_hidden_state.shape[-1]
        )

        # Use pre-trained model's position embedding
        position_ids = torch.arange(
            num_chunks, dtype=torch.long, device=chunks_reps.device
        )  # (n_chunk)
        position_ids = position_ids.unsqueeze(0).expand_as(
            chunks_reps[:, :, 0]
        )  # (B, n_chunk)
        position_embeddings: torch.Tensor = self.encoder.embeddings.position_embeddings(
            position_ids
        )
        chunks_reps = chunks_reps + position_embeddings
        chunks_attns: Union[list[torch.Tensor], None] = None

        # document encoding (B, n_chunk, hidden_size)
        for layer_ind, layer_module in enumerate(self.transformer):
            chunks_reps: torch.Tensor
            chunks_attn: torch.Tensor
            chunks_reps, chunks_attn = layer_module(chunks_reps)
            if output_attentions:
                if chunks_attns is None:
                    chunks_attns = []
                chunks_attns.append(chunks_attn)

            ## this case is mainly for when we are doing subsequent fine-tuning using a pre-trained
            ## hierarchical model and we want to check whether an earlier layer might provide better
            ## classification performance (e.g., if we think the last layer(s) are overfit to the pre-training
            ## objective) Just short circuit rather than doing the whole computation.
            if layer_ind + 1 >= self.config.layer:
                break

        hidden_states = chunks_reps

        # extract first Documents as rep. (B, hidden_size)
        doc_rep = chunks_reps[:, 0, :]

        total_loss = None
        for task in self.tasks:
            loss_fct = CrossEntropyLoss(
                weight=self.class_weights[task.name]
                if self.class_weights is not None
                else None
            )

            # predict (B, 5)
            task_logits = self.classifiers[task.name](doc_rep)
            logits.append(task_logits)

            if labels is not None:
                task_labels = labels[:, task.index]
                task_loss = loss_fct(
                    task_logits, task_labels.type(torch.LongTensor).to(labels.device)
                )
                if total_loss is None:
                    total_loss = task_loss
                else:
                    total_loss += task_loss

        output = HierarchicalSequenceClassifierOutput(
            loss=total_loss,
            logits=logits,
        )
        if self.config.output_hidden_states:
            output.hidden_states = (*outputs.hidden_states, *hidden_states)
        if self.config.output_attentions:
            output.attentions = outputs.attentions
            output.chunk_attentions = chunks_attns
        return output

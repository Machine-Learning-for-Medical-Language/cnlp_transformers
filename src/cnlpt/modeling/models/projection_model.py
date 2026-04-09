"""
Module containing the CNLP transformer model.
"""

from __future__ import annotations

import logging
from typing import Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from ...data.task_info import RELATIONS, TAGGING
from ..config.projection_config import ProjectionModelConfig
from ..modules import ClassificationHead, RepresentationProjectionLayer
from ..utils import freeze_encoder_weights, generalize_encoder_forward_kwargs

logger = logging.getLogger(__name__)


class ProjectionModel(PreTrainedModel):
    base_model_prefix = "cnlpt.proj"
    config_class = ProjectionModelConfig

    def __init__(
        self,
        config: ProjectionModelConfig,
        *,
        class_weights: Union[dict[str, torch.FloatTensor], None] = None,
        final_task_weight: float = 1.0,
        freeze: float = -1.0,
        bias_fit: bool = False,
        **kwargs,
    ):
        """Create a new CNLP transformer model instance from a config object.

        Args:
            config: The CnlpConfig object that configures this model
            class_weights: If provided, the weights to use for each task when computing the loss. Defaults to None.
            final_task_weight: The weight to use for the final task when computing the loss. Defaults to 1.0.
            freeze: What proportion of encoder weights to freeze (-1 for none). Defaults to -1.0.
            bias_fit: Whether to fine-tune only the bias of the encoder. Defaults to False.
        """
        super().__init__(config)
        self.config: ProjectionModelConfig

        # part of the motivation for only resizing embeddings for non character-level models
        # is that at the time of writing,  CANINE and Flair are the only game in town.
        # CANINE's hashable embeddings for unicode codepoints allows for
        # additional parameterization, which rn doesn't seem so relevant
        self.encoder = self.config.load_encoder_model(not self.config.character_level)

        self.tasks = self.config.tasks

        if self.encoder.config.model_type == "modernbert":
            self.num_layers = len(self.encoder.base_model.layers)
        else:
            self.num_layers = len(self.encoder.encoder.layer)
        if self.config.encoder_layer > self.num_layers:
            raise ValueError(
                f"The layer specified ({self.config.encoder_layer}) is too big for the specified encoder which has {self.num_layers} layers"
            )

        if freeze > 0:
            freeze_encoder_weights(self.encoder, freeze)

        if bias_fit:
            for name, param in self.encoder.named_parameters():
                if "bias" not in name:
                    param.requires_grad = False

        self.feature_extractors = nn.ModuleDict()
        self.classifiers = nn.ModuleDict()

        total_prev_task_labels = 0
        for task in self.tasks:
            self.feature_extractors[task.name] = RepresentationProjectionLayer(
                hidden_dropout_prob=self.config.encoder_dropout,
                hidden_size=self.config.encoder_output_dim,
                layer=self.config.encoder_layer,
                tokens=self.config.tokens,
                task_type=task.type,
                num_attention_heads=self.config.num_rel_attention_heads,
                head_size=self.config.rel_attention_head_dims,
            )
            hidden_size = self.config.encoder_output_dim
            if task.type == RELATIONS:
                hidden_size = self.config.num_rel_attention_heads
                if self.config.use_prior_tasks:
                    hidden_size += total_prev_task_labels

            self.classifiers[task.name] = ClassificationHead(
                hidden_dropout_prob=self.config.encoder_dropout,
                hidden_size=hidden_size,
                num_labels=len(task.labels),
            )

            total_prev_task_labels += len(task.labels)

        self.class_weights = class_weights
        self.final_task_weight = final_task_weight
        self.use_prior_tasks = self.config.use_prior_tasks
        self.reg_temperature = 1.0

    def predict_relations_with_previous_logits(
        self, features: torch.Tensor, logits: list[torch.Tensor]
    ) -> torch.Tensor:
        """For the relation prediction task, use previous predictions of the tagging task as additional features in the
        representation used for making the relation prediction.

        Args:
            features: The existing feature vector for the relations
            logits: The predicted logits from the tagging task

        Returns:
            The augmented feature tensor
        """

        # features is (batch x seq x seq x n_heads)
        seq_len = features.shape[1]
        for prior_task_logits in logits:
            if len(features.shape) == 4:
                if len(prior_task_logits.shape) == 3:
                    # prior task is sequence tagging:
                    # we have batch x seq x num_classes.
                    # we want to concatenate the num_classes to the variables at each element of the sequence,
                    # but then need to broadcast it down all the rows of the matrix.
                    aug = prior_task_logits.unsqueeze(
                        2
                    )  # add another dimension to repeat along
                    aug = aug.repeat(
                        1, 1, seq_len, 1
                    )  # repeat along the new empty dimension so we have our seq logits repeated seq_len x seq_len
                    features = torch.cat(
                        (features, aug), 3
                    )  # concatenate the  relation matrix with the sequence matrix
                else:
                    logger.warning(
                        f"It is not implemented to add a task of shape {prior_task_logits.shape!s} to a relation matrix"
                    )
            elif len(features.shape) == 3:
                # sequence
                logger.warning(
                    "It is not implemented to add previous task of any type to a sequence task"
                )

        return features

    def compute_loss(
        self,
        task_logits: torch.FloatTensor,
        labels: torch.LongTensor,
        task_ind: int,
        task_num_labels: int,
        batch_size: int,
        seq_len: int,
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
        task = self.tasks[task_ind]
        if task_num_labels == 1:
            #  We are doing regression
            loss_fct = MSELoss()
            task_loss = loss_fct(task_logits.view(-1), labels.view(-1))
        else:
            loss_fct = CrossEntropyLoss(
                weight=self.class_weights[task.name]
                if self.class_weights is not None
                else None
            )

            if task.type == RELATIONS:
                task_labels = labels[
                    :, :, state["task_label_ind"] : state["task_label_ind"] + seq_len
                ]
                state["task_label_ind"] += seq_len
                task_loss = loss_fct(
                    task_logits.permute(0, 3, 1, 2),
                    task_labels.type(torch.LongTensor).to(labels.device),
                )
            elif task.type == TAGGING:
                # in cases where we are only given a single task the HF code will have one fewer dimension in the labels, so just add a dummy dimension to make our indexing work:
                if labels.ndim == 2:
                    task_labels = labels
                elif labels.ndim == 3:
                    # labels = labels.unsqueeze(1)
                    task_labels = labels[:, :, state["task_label_ind"]]
                else:
                    task_labels = labels[:, 0, state["task_label_ind"], :]

                state["task_label_ind"] += 1
                task_loss = loss_fct(
                    task_logits.view(-1, task_num_labels),
                    task_labels.reshape(
                        [
                            batch_size * seq_len,
                        ]
                    )
                    .type(torch.LongTensor)
                    .to(labels.device),
                )
            else:  # task.type == CLASSIFICATION
                if labels.ndim == 1:
                    task_labels = labels
                elif labels.ndim == 2:
                    task_labels = labels[:, task_ind]
                elif labels.ndim == 3:
                    task_labels = labels[:, 0, task_ind]
                else:
                    raise NotImplementedError(
                        "Have not implemented the case where a classification task "
                        "is part of an MTL setup with relations and sequence tagging"
                    )

                state["task_label_ind"] += 1
                task_loss = loss_fct(
                    task_logits, task_labels.type(torch.LongTensor).to(labels.device)
                )

        if state["loss"] is None:
            state["loss"] = task_loss
        else:
            task_weight = (
                1.0 if task_ind + 1 < len(self.tasks) else self.final_task_weight
            )
            state["loss"] += task_weight * task_loss

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
        event_tokens=None,
        **kwargs,
    ):
        r"""Forward method.

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
            event_tokens: a mask defining which tokens in the input are to be averaged for input to classifier head; only used when self.tokens==True.

        Returns: (`transformers.SequenceClassifierOutput`) the output of the model
        """

        kwargs = generalize_encoder_forward_kwargs(
            self.encoder,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        outputs = self.encoder(input_ids, **kwargs)

        batch_size, seq_len, _ = outputs.last_hidden_state.shape

        logits = []

        state = dict(loss=None, task_label_ind=0)

        for task in self.tasks:
            # hidden_states has shape (layers x batch x seq x hidden)

            # features shape:
            # for classification (including event tokens mode): (batch x hidden)
            # for tagging: (batch x seq x hidden)
            # for relations: (batch x seq x seq x n_heads)
            features = self.feature_extractors[task.name](
                outputs.hidden_states, event_tokens
            )
            if self.use_prior_tasks:
                # note: this specific way of incorporating previous logits doesn't help in my experiments with thyme/clinical tempeval
                if task.type == RELATIONS:
                    features = self.predict_relations_with_previous_logits(
                        features, logits
                    )
            task_logits = self.classifiers[task.name](features)
            logits.append(task_logits)

            if labels is not None:
                self.compute_loss(
                    task_logits,
                    labels,
                    task.index,
                    len(task.labels),
                    batch_size,
                    seq_len,
                    state,
                )
        output = SequenceClassifierOutput(loss=state["loss"], logits=logits)
        if self.config.output_hidden_states:
            output.hidden_states = outputs.hidden_states
        if self.config.output_attentions:
            output.attentions = outputs.attentions

        return output

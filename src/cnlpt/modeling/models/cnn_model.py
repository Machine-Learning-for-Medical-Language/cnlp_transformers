from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from ..config.cnn_config import CnnModelConfig


class CnnModel(PreTrainedModel):
    base_model_prefix = "cnlpt.cnn"
    config_class = CnnModelConfig

    def __init__(
        self,
        config: CnnModelConfig,
        *,
        class_weights: Union[dict[str, torch.FloatTensor], None] = None,
        **kwargs,
    ):
        super().__init__(config)
        self.config: CnnModelConfig

        self.embed = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=config.embed_dim,
                    out_channels=config.filters_per_size,
                    kernel_size=filter_size,
                )
                for filter_size in config.filter_sizes
            ]
        )

        self.loss_fns = {
            task.name: nn.CrossEntropyLoss(
                weight=class_weights[task.name] if class_weights is not None else None
            )
            for task in self.config.tasks
        }

        total_filters = len(self.config.filter_sizes) * self.config.filters_per_size

        self.fcs = nn.ModuleList()
        for task in self.config.tasks:
            self.fcs.append(nn.Linear(total_filters, len(task.labels)))

        if self.config.use_prior_tasks:
            self.intertask_matrices: list[list[nn.Linear]] = []
            for i in range(len(self.tasks)):
                matrices = []
                for j in range(len(self.tasks) - i):
                    matrices.append(nn.Linear(2, total_filters))
                self.intertask_matrices.append(matrices)

    def forward(
        self,
        input_ids: Union[torch.LongTensor, None] = None,
        labels: Union[torch.LongTensor, None] = None,
        output_hidden_states=False,
        **kwargs,
    ):
        embeddings: torch.Tensor = self.embed(input_ids)
        embeddings = embeddings.transpose(1, 2)
        all_convs: list[torch.Tensor] = [conv(embeddings) for conv in self.convs]
        pooled_convs = [
            F.max_pool1d(conv_out, conv_out.shape[2]) for conv_out in all_convs
        ]

        fc_in = torch.cat(pooled_convs, 1).squeeze(2)

        logits = []
        loss = 0
        for task, fc in zip(self.config.tasks, self.fcs):
            # get feaures from previous tasks using the world's tiniest linear layer
            if self.config.use_prior_tasks:
                for prev_task_ind in range(task.index):
                    prev_task_matrix = self.intertask_matrices[prev_task_ind][
                        task.index - prev_task_ind - 1
                    ]
                    prev_task_matrix = prev_task_matrix.to(logits[prev_task_ind].device)
                    prev_task_features = prev_task_matrix(logits[prev_task_ind])
                    fc_in = fc_in + prev_task_features
            task_logits: torch.Tensor = fc(fc_in)
            logits.append(task_logits)

            if labels is not None:
                if labels.ndim == 2:
                    # if len(self.fcs) == 1:
                    #     task_labels = labels[:,0]
                    task_labels = labels[:, task.index]
                elif labels.ndim == 3:
                    task_labels = labels[:, 0, task.index]
                loss += self.loss_fns[task.name](
                    task_logits, task_labels.type(torch.LongTensor).to(labels.device)
                )
        if output_hidden_states:
            return loss, logits, fc_in
        else:
            return loss, logits

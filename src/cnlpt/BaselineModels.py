import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from torch import nn


class CnnSentenceClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        vocab_size,
        task_names: list[str],
        num_labels_dict: dict[str, int],
        embed_dims=100,
        num_filters=25,
        dropout=0.2,
        filters=(1, 2, 3),
        use_prior_tasks=False,
        class_weights=None,
    ):
        super().__init__()
        self.dropout = dropout

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dims)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dims, num_filters, x) for x in filters]
        )
        # need separate loss functions with different weights
        if class_weights is not None:
            if class_weights.ndim > 1:
                self.loss_fn = {
                    task_name: nn.CrossEntropyLoss(weight=class_weights[i])
                    for i, task_name in enumerate(task_names)
                }
            else:
                self.loss_fn = {
                    task_name: nn.CrossEntropyLoss(weight=class_weights)
                    for task_name in task_names
                }
        else:
            self.loss_fn = {
                task_name: nn.CrossEntropyLoss() for task_name in task_names
            }
        self.fcs = nn.ModuleList()

        self.task_names = task_names
        for task_name in self.task_names:
            if task_name not in num_labels_dict:
                raise ValueError("Misalignment between task_names and num_labels_dict")
            self.fcs.append(
                nn.Linear(num_filters * len(filters), num_labels_dict[task_name])
            )

        self.use_prior_tasks = use_prior_tasks
        if self.use_prior_tasks:
            self.intertask_matrices = []
            for i in range(len(self.task_names)):
                matrices = []
                for j in range(len(self.task_names) - i):
                    matrices.append(nn.Linear(2, num_filters * len(filters)))
                self.intertask_matrices.append(matrices)
            # self.intertask_matrix = nn.Linear(2, num_filters * len(filters))
            # put logits for task a through intertask_matrix[a][b] to get features to add to features of task b

    def forward(
        self,
        input_ids=None,
        event_tokens=None,
        labels=None,
        output_hidden_states=False,
        **kwargs,
    ):
        embeddings = self.embed(input_ids)
        embeddings = embeddings.transpose(1, 2)
        all_convs = [conv(embeddings) for conv in self.convs]
        pooled_convs = [
            F.max_pool1d(conv_out, conv_out.shape[2]) for conv_out in all_convs
        ]

        fc_in = torch.cat(pooled_convs, 1).squeeze(2)

        logits = []
        loss = 0
        for task_ind, task_fc in enumerate(self.fcs):
            # get feaures from previous tasks using the world's tiniest linear layer
            if self.use_prior_tasks:
                for prev_task_ind in range(task_ind):
                    prev_task_matrix = self.intertask_matrices[prev_task_ind][
                        task_ind - prev_task_ind - 1
                    ]
                    prev_task_matrix = prev_task_matrix.to(logits[prev_task_ind].device)
                    prev_task_features = prev_task_matrix(logits[prev_task_ind])
                    fc_in = fc_in + prev_task_features
            task_logits = task_fc(fc_in)
            logits.append(task_logits)

            if labels is not None:
                if labels.ndim == 2:
                    # if len(self.fcs) == 1:
                    #     task_labels = labels[:,0]
                    task_labels = labels[:, task_ind]
                elif labels.ndim == 3:
                    task_labels = labels[:, 0, task_ind]
                loss += self.loss_fn[self.task_names[task_ind]](
                    task_logits, task_labels.type(torch.LongTensor).to(labels.device)
                )
        if output_hidden_states:
            return loss, logits, fc_in
        else:
            return loss, logits


class LstmSentenceClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        vocab_size,
        task_names: list[str],
        num_labels_dict: dict[str, int],
        embed_dims=100,
        dropout=0.2,
        hidden_size=100,
    ):
        super().__init__()
        self.dropout = dropout

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dims)
        self.lstm = nn.LSTM(
            input_size=embed_dims, hidden_size=hidden_size, bidirectional=True
        )
        self.loss_fn = nn.CrossEntropyLoss()

        self.fcs = nn.ModuleList()

        self.task_names = task_names
        for task_name in self.task_names:
            if task_name not in num_labels_dict:
                raise ValueError("Misalignment between task_names and num_labels_dict")
            self.fcs.append(nn.Linear(4 * hidden_size, num_labels_dict[task_name]))

    def forward(
        self,
        input_ids=None,
        event_tokens=None,
        labels=None,
        **kwargs,
    ):
        embeddings = self.embed(input_ids)
        # embeddings = embeddings.transpose(1,2)
        lstm_out = self.lstm(embeddings)[0]

        logits = []
        loss = 0
        for task_ind, task_fc in enumerate(self.fcs):
            features = torch.cat((lstm_out[:, 0, :], lstm_out[:, -1, :]), 1)
            task_logits = task_fc(features)
            logits.append(task_logits)

            if labels is not None:
                if labels.ndim == 2:
                    task_labels = labels[:, 0]
                elif labels.ndim == 3:
                    task_labels = labels[:, 0, task_ind]
                loss += self.loss_fn(
                    task_logits, task_labels.type(torch.LongTensor).to(labels.device)
                )
        return loss, logits

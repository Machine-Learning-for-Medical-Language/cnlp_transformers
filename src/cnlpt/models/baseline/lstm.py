import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn


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

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from ..config.lstm_config import LstmModelConfig


class LstmModel(PreTrainedModel):
    base_model_prefix = "cnlpt.lstm"
    config_class = LstmModelConfig

    def __init__(
        self,
        config: LstmModelConfig,
        *,
        class_weights: dict[str, torch.FloatTensor] | None = None,
        **kwargs,
    ):
        super().__init__(config)
        self.config = config
        self.embed = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
        )
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_size,
            bidirectional=True,
        )
        self.loss_fns = {
            task.name: nn.CrossEntropyLoss(
                weight=class_weights[task.name] if class_weights is not None else None
            )
            for task in self.config.tasks
        }

        self.fcs = nn.ModuleList()

        for task in config.tasks:
            self.fcs.append(nn.Linear(4 * config.hidden_size, len(task.labels)))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ):
        embeddings = self.embed(input_ids)
        lstm_out = self.lstm(embeddings)[0]

        logits: list[torch.Tensor] = []
        loss = 0
        for task, fc in zip(self.config.tasks, self.fcs):
            features = torch.cat((lstm_out[:, 0, :], lstm_out[:, -1, :]), 1)
            task_logits: torch.Tensor = fc(features)
            logits.append(task_logits)

            if labels is not None:
                if labels.ndim == 2:
                    task_labels = labels[:, 0]
                elif labels.ndim == 3:
                    task_labels = labels[:, 0, task.index]
                loss += self.loss_fns[task.name](
                    task_logits, task_labels.type(torch.LongTensor).to(labels.device)
                )
        return loss, logits

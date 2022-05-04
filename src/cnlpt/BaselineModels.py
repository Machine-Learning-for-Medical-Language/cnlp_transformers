import torch
from torch import nn
import torch.nn.functional as F

class CnnSentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dims=100, num_labels_list=[2,], num_filters=25, dropout=0.2, filters=(1,2,3)):
        super(CnnSentenceClassifier, self).__init__()
        self.dropout =  dropout

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dims)
        self.convs = nn.ModuleList( [nn.Conv1d(embed_dims, num_filters, x) for x in filters] )
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.fcs = nn.ModuleList()
        for task_num_labels in num_labels_list:
            self.fcs.append(nn.Linear(num_filters * len(filters), task_num_labels))


    def forward(
        self,
        input_ids=None,
        event_tokens=None,
        labels=None,
        **kwargs,
    ):
        embeddings = self.embed(input_ids)
        embeddings = embeddings.transpose(1,2)
        all_convs = [conv(embeddings) for conv in self.convs]
        pooled_convs = [F.max_pool1d(conv_out, conv_out.shape[2]) for conv_out in all_convs]

        fc_in = torch.cat(pooled_convs, 1).squeeze(2)

        logits = []
        loss = 0
        for task_ind,task_fc in enumerate(self.fcs):
            task_logits = task_fc(fc_in)
            logits.append(task_logits)

            if not labels is None:
                if labels.ndim == 2:
                    task_labels = labels[:,0]
                elif labels.ndim == 3:
                    task_labels = labels[:,0,task_ind]
                loss += self.loss_fn(task_logits, task_labels.type(torch.LongTensor).to(labels.device))
        
        return loss, logits


class LstmSentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dims=100, num_labels_list=[2,], dropout=0.2, hidden_size=100):
        super(LstmSentenceClassifier, self).__init__()
        self.dropout =  dropout

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dims)
        self.lstm = nn.LSTM(input_size = embed_dims, hidden_size=hidden_size, bidirectional=True)
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.fcs = nn.ModuleList()
        for task_num_labels in num_labels_list:
            self.fcs.append(nn.Linear(4*hidden_size, task_num_labels))

    def forward(
        self,
        input_ids = None,
        event_tokens = None,
        labels = None,
        **kwargs,
    ):
        embeddings = self.embed(input_ids)
        # embeddings = embeddings.transpose(1,2)
        lstm_out = self.lstm(embeddings)[0]

        logits = []
        loss = 0
        for task_ind, task_fc in enumerate(self.fcs):
            features = torch.cat( (lstm_out[:,0,:], lstm_out[:,-1,:]), 1)
            task_logits = task_fc(features)
            logits.append(task_logits)

            if not labels is None:
                if labels.ndim == 2:
                    task_labels = labels[:,0]
                elif labels.ndim == 3:
                    task_labels = labels[:,0,task_ind]
                loss += self.loss_fn(task_logits, task_labels.type(torch.LongTensor).to(labels.device))


        return loss, logits


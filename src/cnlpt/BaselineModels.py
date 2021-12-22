import torch
from torch import nn
import torch.nn.functional as F

class CnnSentenceClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dims=100, num_labels=2, num_filters=25, dropout=0.2, filters=(1,2,3)):
        super(CnnSentenceClassifier, self).__init__()
        self.dropout =  dropout

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dims)
        self.convs = nn.ModuleList( [nn.Conv1d(embed_dims, num_filters, x) for x in filters] )
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.fc = nn.Linear(num_filters * len(filters), num_labels)


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

        logits = self.fc(fc_in)

        if not labels is None:
            loss = self.loss_fn(logits, labels)
        
        return loss, logits



import time
import copy
import tempfile
import pickle
import shutil
import os

import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers.models.distilbert import DistilBertPreTrainedModel, DistilBertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score
# import wandb

# from transformer import EncoderLayer
# from utils import set_seed

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class HierarchicalTransformerConfig(object):
    def __init__(self,
                 n_layers,
                 d_model,
                 d_inner,
                 n_head,
                 d_k,
                 d_v,
                 dropout=0.1):
        self.n_layers = n_layers
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout


class HierarchicalTransformer(DistilBertPreTrainedModel):

    def __init__(self, distilbert_config, transformer_head_config):
        # when calling the from_pretrained method, the distilbert config is not necessary.
        super().__init__(config=distilbert_config)

        # Set up number of classes
        self.num_labels = distilbert_config.num_labels

        # distilbert model.
        self.distilbert = DistilBertModel(distilbert_config)

        # Transformer layer
        transformer_layer = EncoderLayer(d_model=transformer_head_config.d_model,
                                         d_inner=transformer_head_config.d_inner,
                                         n_head=transformer_head_config.n_head,
                                         d_k=transformer_head_config.d_k,
                                         d_v=transformer_head_config.d_v,
                                         dropout=transformer_head_config.dropout)
        self.transformer = nn.ModuleList(
            [copy.deepcopy(transformer_layer) for _ in range(transformer_head_config.n_layers)]
        )

        # document level
        self.classifier = nn.Sequential(
            nn.Linear(transformer_head_config.d_model,
                      distilbert_config.num_labels),
        )

        # weights
        self.init_weights()

    def forward(self, token_ids, attention_masks, head_mask=None):
        # BERT CLS outputs for each chunk
        logits = []

        # for each sample in a batch (B, n_chunks, max_len)
        for token_id, attention_mask in zip(token_ids, attention_masks):
            # Transform the word embeddings. (n_chunks, max_len)
            distilbert_output = self.distilbert(input_ids=token_id,
                                                attention_mask=attention_mask,
                                                head_mask=None,
                                                inputs_embeds=None)
            # Transformed word embeddings. (n_chunks, max_len, hidden_size)
            hidden_state = distilbert_output[0]

            # Extract the first token, CLS, embedding as chunk rep. (n_chunk, hidden_size)
            chunks_reps = hidden_state[:, 0]

            # Add addition dim. (1, n_chunk, hidden_size)
            chunks_reps = chunks_reps[None, :, :]

            # Use pre-trained model's position embedding
            seq_length = chunks_reps.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=chunks_reps.device)  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(chunks_reps[:, :, 0])  # (bs, max_seq_length)
            position_embeddings = self.distilbert.embeddings.position_embeddings(position_ids)
            chunks_reps += position_embeddings

            # document encoding
            for layer_module in self.transformer:
                chunks_reps, _ = layer_module(chunks_reps)

            # Remove the first dim. (n_chunk, hidden_size)
            chunks_reps = chunks_reps.squeeze()

            # extract first Documents as rep. (hidden_size)
            doc_rep = chunks_reps[0, :]

            # predict
            logits.append(self.classifier(doc_rep))

        # Batch outputs. (B, 2)
        logits = torch.stack(logits)  # batch predictions, forward pass

        outputs = (logits,)

        return outputs


# Train function
def train(model,
          dataset_train,
          return_best_model=False,
          dataset_val=None,
          per_gpu_train_batch_size=None,
          per_gpu_eval_batch_size=None,
          gradient_accumulation_steps=None,
          epochs=None,
          learning_rate=None,
          class_weight=None,
          seed=None,
          early_stopping_patient=None,
          early_stopping_delta=None,
          metric=None,
          labels_val=None,
          warmup=None
          ):
    """
    Default is using the first GPU to store model's parameters and all data.
    The early stop is using validation performance to decide when to stop.
    :returns best_model, train_loss,
    """
    # Metrics can be computed in the training.
    implemented_metrics = {"macro_f1", "roc_auc"}
    if metric is not None:
        if metric not in implemented_metrics:
            raise ValueError("The metric for validation is not implemented")

    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if class_weight is not None:
        class_weight = class_weight.to(device)
    train_batch_size = per_gpu_train_batch_size * max(1, n_gpus)
    train_sampler = RandomSampler(dataset_train)
    train_dataloader = DataLoader(dataset_train, sampler=train_sampler, batch_size=train_batch_size)
    epoch_steps = len(train_dataloader) // gradient_accumulation_steps
    t_total = epoch_steps * epochs
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_warmup_steps = int(t_total * warmup) if warmup <= 1 else warmup
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=t_total)
    # wandb.run.summary["num_warmup_steps"] = num_warmup_steps
    print(
        "| samples {} | total epochs {} | batch {} | gradient_accum {} | total steps {} | num_warmup_steps {} |".format(
            len(dataset_train), epochs, (train_batch_size * gradient_accumulation_steps),
            gradient_accumulation_steps, t_total, num_warmup_steps
        ))
    print('=' * 80)

    # ######################
    # Training
    # ######################

    global_step = 0
    tr_loss, epoch_loss = 0.0, 0.0
    best_val_performance = float('-inf')
    early_stop = False
    early_stopping_counter = 0
    best_epochs = None
    best_model_dir = tempfile.mkdtemp()
    train_start_time = time.time()
    model.zero_grad()
    set_seed(n_gpu=n_gpus, seed=seed)

    for epochs_i in range(epochs):
        new_best_loss_val = False
        epoch_start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "token_ids": batch[0],
                "attention_masks": batch[2],
            }
            logits = model(**inputs)[0]
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight)
            loss = loss_fct(logits, batch[4].view(-1))
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
        epoch_time = time.time() - epoch_start_time
        loss_val = None
        metric_val = None
        if dataset_val is not None:
            pred_probs_val, loss_val = predict(model, dataset_val, per_gpu_eval_batch_size,
                                               class_weight=class_weight)
            # Compute the metrics
            if metric is not None and metric == "macro_f1":
                metric_val = f1_score(y_true=labels_val, y_pred=np.argmax(pred_probs_val, axis=1), average="macro")
            if metric is not None and metric == "roc_auc":
                metric_val = roc_auc_score(y_true=labels_val,
                                           y_score=pred_probs_val[:, 1])

            # Test if the validation loss is the new best loss
            if metric_val > best_val_performance + early_stopping_delta:
                new_best_loss_val = True
                best_val_performance = metric_val
                early_stopping_counter = 0
                best_epochs = epochs_i + 1  # the epochs_i start from 0, so add 1

                # save the best model's checkpoint
                with open(os.path.join(best_model_dir, "best_model.pkl"), "wb") as f:
                    pickle.dump(model, f)
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patient:
                    early_stop = True
        # wandb.log({"loss_train": ((tr_loss - epoch_loss) / epoch_steps)})
        # if dataset_val is not None:
            # wandb.log({'loss_val': loss_val})
        if dataset_val is not None:
            print(
                "| epoch {}/{} | time {:.2f} s | train_loss {:.5f} | loss_val {:.5f} | {} {:.5f} |new_best_val {} |".format(
                    (epochs_i + 1),
                    epochs,
                    epoch_time,
                    ((tr_loss - epoch_loss) / epoch_steps),
                    loss_val,
                    metric,
                    metric_val,
                    new_best_loss_val
                ))
        else:
            print("| epoch {}/{} | time {:.2f} s | train_loss {:.5f} |".format(
                    (epochs_i + 1),
                    epochs,
                    epoch_time,
                    ((tr_loss - epoch_loss) / epoch_steps)))
        epoch_loss = tr_loss
        if early_stop:
            break
    if return_best_model:
        with open(os.path.join(best_model_dir, "best_model.pkl"), "rb") as f:
            model_to_return = pickle.load(f)
            model_to_return = model_to_return.module if hasattr(model, 'module') else model
        #shutil.rmtree(best_model_dir)  # clean up the cache dir
    else:
        model_to_return = model.module if hasattr(model, 'module') else model

    train_time = time.time() - train_start_time

    # print the summary of training.
    print("-" * 80)
    print("| epochs {} | time {:.2f} s| avg_train_loss {:.5f}".format(
        best_epochs if best_epochs is not None else epochs,
        train_time,
        (tr_loss / global_step)
    ))

    return model_to_return, tr_loss / global_step, train_time, best_epochs if best_epochs is not None else epochs


def predict(model,
            dataset,
            per_gpu_batch_size,
            class_weight=None):
    """
      Default is using the first GPU to store model's parameters and all data.

      returns logits_list, loss
      """
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = per_gpu_batch_size * max(1, n_gpus)

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    loss = 0.0
    steps = 0
    logits_list = None
    for batch in dataloader:
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "token_ids": batch[0],
                "attention_masks": batch[2],
            }
            # forward pass
            logits = model(**inputs)[0]

            # Compute the loss function
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weight)
            temp_loss = loss_fct(logits, batch[4].view(-1))
            loss += temp_loss.mean().item()
        steps += 1

        if logits_list is None:
            logits_list = logits.detach().cpu().numpy()
        else:
            logits_list = np.append(logits_list, logits.detach().cpu().numpy(), axis=0)

    return logits_list, loss / steps
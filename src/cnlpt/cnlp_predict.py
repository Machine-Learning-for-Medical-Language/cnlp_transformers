import numpy as np

from datasets import Dataset
from transformers import Trainer, PreTrainedTokenizer
from .cnlp_processors import tagging, relex, classification
from .cnlp_data import ClinicalNlpDataset
from seqeval.metrics.sequence_labeling import get_entities
from typing import Dict


def write_predictions_for_dataset(
    output_fn: str,
    trainer: Trainer,
    dataset: ClinicalNlpDataset,
    split_name: str,
    dataset_ind: int,
    output_mode: Dict[str, str],
    tokenizer: PreTrainedTokenizer,
    output_prob: bool = False,
    task_labels = dataset.get_labels()
    start_ind = end_ind = 0
    for ind in range(dataset_ind):
        start_ind += len(dataset.datasets[ind][split_name])
    end_ind = start_ind + len(dataset.datasets[dataset_ind][split_name])

    with open(output_fn, "w") as writer:
        eval_dataset = Dataset.from_dict(
            dataset.processed_dataset[split_name][start_ind:end_ind]
        )
        predictions = trainer.predict(test_dataset=eval_dataset).predictions
        for task_ind, task_name in enumerate(dataset.tasks):
            if output_prob and output_mode[task_name] != classification:
                raise NotImplementedError(
                    "Writing predictions is not implemented for this output_mode!"
                )

            if output_mode[task_name] == classification:
                task_predictions = predictions[task_ind]
                for index, logits in enumerate(task_predictions):
                    task_prediction_idx = np.argmax(logits, axis=1)
                    item = task_labels[task_name][task_prediction_idx]
                    prob_value = logits[task_prediction_idx]
                    if output_prob:
                        writer.write(
                            "Task %d (%s) - Index %d - %s - %.6f\n"
                            % (task_ind, task_name, index, item, prob_value)
                        )
                    else:
                        writer.write(
                            "Task %d (%s) - Index %d - %s\n"
                            % (task_ind, task_name, index, item)
                        )
            elif output_mode[task_name] == tagging:
                task_predictions = np.argmax(predictions[task_ind], axis=2)
                tagging_labels = task_labels[task_name]
                for index, seq_pair in enumerate(zip(task_predictions, tagging_labels)):
                    pred_seq, true_seq = seq_pair
                    wpind_to_ind = {}
                    chunk_labels = []

                    token_inds = eval_dataset["input_ids"][index]
                    text = eval_dataset["text"][index]
                    predicted_labels = [
                        tagging_labels[task_predictions[index][i[0]]]
                        for i in filter(
                            lambda s: not all(i == -100 for i in s[1]),
                            enumerate(eval_dataset["label"][index]),
                        )
                    ]
                    true_ner = eval_dataset[task_name][index]

                    writer.write(
                        f"{eval_dataset.column_names} {text} : {len(text.split())} true ner {true_ner}  {predicted_labels} {len(predicted_labels)} \n"
                    )
            elif output_mode[task_name] == relex:
                task_predictions = np.argmax(predictions[task_ind], axis=3)
                relex_labels = task_labels[task_name]
                none_index = (
                    relex_labels.index("None") if "None" in relex_labels else -1
                )

                for inst_ind in range(task_predictions.shape[0]):
                    inst_preds = task_predictions[inst_ind]
                    a1s, a2s = np.where(inst_preds != none_index)
                    for arg_ind in range(len(a1s)):
                        a1_ind = a1s[arg_ind]
                        a2_ind = a2s[arg_ind]
                        cat = relex_labels[inst_preds[a1_ind][a2_ind]]
                        writer.write(
                            "Task %d (%s) - Index %d - %s(%d, %d)\n"
                            % (task_ind, task_name, inst_ind, cat, a1_ind, a2_ind)
                        )
            else:
                raise NotImplementedError(
                    "Writing predictions is not implemented for this output_mode!"
                )

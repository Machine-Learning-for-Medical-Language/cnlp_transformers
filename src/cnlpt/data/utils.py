from datasets import Dataset as HFDataset

from .cnlp_datasets import ClinicalNlpDataset


def get_dataset_segment(
    split_name: str,
    dataset_ind: int,
    dataset: ClinicalNlpDataset,
):
    start_ind = end_ind = 0
    for ind in range(dataset_ind):
        start_ind += len(dataset.datasets[ind][split_name])
    end_ind = start_ind + len(dataset.datasets[dataset_ind][split_name])

    return HFDataset.from_dict(dataset.processed_dataset[split_name][start_ind:end_ind])

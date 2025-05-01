import copy
import json

import pytest

from cnlpt.new_data.data_reader import CnlpDataReader
from cnlpt.new_data.task_info import (
    CLASSIFICATION,
    RELATIONS,
    TAGGING,
    TaskInfo,
)

JSON_DATASET_OBJ = {
    "data": [
        {
            "text": "0_text",
            "id": "0_id",
            "task_a": "0_task_a",
            "task_b": "0_task_b",
            "task_c": "0_task_c",
        },
        {
            "text": "1_text",
            "id": "1_id",
            "task_a": "1_task_a",
            "task_b": "1_task_b",
            "task_c": "1_task_c",
        },
        {
            "text": "2_text",
            "id": "2_id",
            "task_a": "2_task_a",
            "task_b": "2_task_b",
            "task_c": "2_task_c",
        },
    ],
    "metadata": {
        "subtasks": [
            {"task_name": "task_a", "output_mode": "classification"},
            {"task_name": "task_b", "output_mode": "tagging"},
            {"task_name": "task_c", "output_mode": "relex"},
        ]
    },
}


def test_load_single_json_file(tmp_path):
    reader = CnlpDataReader()

    with open(tmp_path / "train.json", "w") as f:
        json.dump(JSON_DATASET_OBJ, f)

    reader.load_json(f.name)

    assert reader.split_names == {"train"}
    assert reader.get_tasks() == (
        TaskInfo(
            name="task_a",
            type=CLASSIFICATION,
            index=0,
            labels=("0_task_a", "1_task_a", "2_task_a"),
        ),
        TaskInfo(
            name="task_b",
            type=TAGGING,
            index=1,
            labels=("0_task_b", "1_task_b", "2_task_b"),
        ),
        TaskInfo(
            name="task_c",
            type=RELATIONS,
            index=2,
            labels=("0_task_c", "1_task_c", "2_task_c"),
        ),
    )


def test_load_multiple_json_files(tmp_path):
    reader = CnlpDataReader()

    for split in ("train", "test", "dev"):
        with open(tmp_path / f"{split}.json", "w") as f:
            json.dump(JSON_DATASET_OBJ, f)
        reader.load_json(f.name)

    assert reader.split_names == {"train", "test", "validation"}
    assert reader.get_tasks() == (
        TaskInfo(
            name="task_a",
            type=CLASSIFICATION,
            index=0,
            labels=("0_task_a", "1_task_a", "2_task_a"),
        ),
        TaskInfo(
            name="task_b",
            type=TAGGING,
            index=1,
            labels=("0_task_b", "1_task_b", "2_task_b"),
        ),
        TaskInfo(
            name="task_c",
            type=RELATIONS,
            index=2,
            labels=("0_task_c", "1_task_c", "2_task_c"),
        ),
    )


def test_load_dir_of_json_files(tmp_path):
    reader = CnlpDataReader()

    for split in ("train", "test", "dev"):
        with open(tmp_path / f"{split}.json", "w") as f:
            json.dump(JSON_DATASET_OBJ, f)

    reader.load_dir(tmp_path)

    assert reader.split_names == {"train", "test", "validation"}
    assert reader.get_tasks() == (
        TaskInfo(
            name="task_a",
            type=CLASSIFICATION,
            index=0,
            labels=("0_task_a", "1_task_a", "2_task_a"),
        ),
        TaskInfo(
            name="task_b",
            type=TAGGING,
            index=1,
            labels=("0_task_b", "1_task_b", "2_task_b"),
        ),
        TaskInfo(
            name="task_c",
            type=RELATIONS,
            index=2,
            labels=("0_task_c", "1_task_c", "2_task_c"),
        ),
    )


def test_ambiguous_split(tmp_path):
    reader = CnlpDataReader()

    with open(tmp_path / "foo.json", "w") as f:
        json.dump(JSON_DATASET_OBJ, f)

    with pytest.raises(ValueError, match="unable to infer split"):
        reader.load_json(f.name)


def test_incompatible_output_modes(tmp_path):
    reader = CnlpDataReader()

    with open(tmp_path / "train.json", "w") as f:
        json.dump(JSON_DATASET_OBJ, f)
    reader.load_json(f.name)

    incompatible_task_data = copy.deepcopy(JSON_DATASET_OBJ)
    incompatible_task_data["metadata"]["subtasks"][0]["output_mode"] = "tagging"
    with open(tmp_path / "test.json", "w") as f:
        json.dump(incompatible_task_data, f)

    with pytest.raises(
        ValueError, match="two different output modes in different datasets"
    ):
        reader.load_json(f.name)


def test_disjoint_labels(tmp_path):
    reader = CnlpDataReader()

    with open(tmp_path / "train.json", "w") as f:
        json.dump(JSON_DATASET_OBJ, f)
    reader.load_json(f.name)

    disjoint_labels_task_data = copy.deepcopy(JSON_DATASET_OBJ)
    for instance in disjoint_labels_task_data["data"]:
        instance["task_a"] = instance["task_a"] + "_foo"
    with open(tmp_path / "test.json", "w") as f:
        json.dump(disjoint_labels_task_data, f)

    with pytest.raises(
        ValueError, match="disjoint sets of labels in different datasets"
    ):
        reader.load_json(f.name)

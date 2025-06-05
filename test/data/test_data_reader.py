import copy
import json
from pathlib import Path

import pytest

from cnlpt.data.data_reader import CnlpDataReader
from cnlpt.data.task_info import TaskInfo


@pytest.fixture
def json_dataset_obj():
    json_dataset_path = (
        Path(__file__).parent.parent / "common" / "datasets" / "json" / "train.json"
    )
    with open(json_dataset_path) as f:
        data = json.load(f)
    return data


def test_load_single_json_file(tmp_path, json_dataset_obj):
    reader = CnlpDataReader()

    with open(tmp_path / "train.json", "w") as f:
        json.dump(json_dataset_obj, f)

    reader.load_json(f.name)

    assert reader.split_names == {"train"}
    assert reader.get_tasks() == (
        TaskInfo(
            name="task_a", type="classification", index=0, labels=("Bar", "Baz", "Foo")
        ),
        TaskInfo(name="task_b", type="tagging", index=1, labels=("bar", "baz", "foo")),
        TaskInfo(
            name="task_c",
            type="relations",
            index=2,
            labels=("BAR", "BAZ", "FOO", "None"),
        ),
    )


def test_load_multiple_json_files(tmp_path, json_dataset_obj):
    reader = CnlpDataReader()

    for split in ("train", "test", "dev"):
        with open(tmp_path / f"{split}.json", "w") as f:
            json.dump(json_dataset_obj, f)
        reader.load_json(f.name)

    assert reader.split_names == {"train", "test", "validation"}
    assert reader.get_tasks() == (
        TaskInfo(
            name="task_a", type="classification", index=0, labels=("Bar", "Baz", "Foo")
        ),
        TaskInfo(name="task_b", type="tagging", index=1, labels=("bar", "baz", "foo")),
        TaskInfo(
            name="task_c",
            type="relations",
            index=2,
            labels=("BAR", "BAZ", "FOO", "None"),
        ),
    )


def test_load_dir_of_json_files(tmp_path, json_dataset_obj):
    reader = CnlpDataReader()

    for split in ("train", "test", "dev"):
        with open(tmp_path / f"{split}.json", "w") as f:
            json.dump(json_dataset_obj, f)

    reader.load_dir(tmp_path)

    assert reader.split_names == {"train", "test", "validation"}
    assert reader.get_tasks() == (
        TaskInfo(
            name="task_a", type="classification", index=0, labels=("Bar", "Baz", "Foo")
        ),
        TaskInfo(name="task_b", type="tagging", index=1, labels=("bar", "baz", "foo")),
        TaskInfo(
            name="task_c",
            type="relations",
            index=2,
            labels=("BAR", "BAZ", "FOO", "None"),
        ),
    )


def test_ambiguous_split(tmp_path, json_dataset_obj):
    reader = CnlpDataReader()

    with open(tmp_path / "foo.json", "w") as f:
        json.dump(json_dataset_obj, f)

    with pytest.raises(ValueError, match="unable to infer split"):
        reader.load_json(f.name)


def test_incompatible_output_modes(tmp_path, json_dataset_obj):
    reader = CnlpDataReader()

    with open(tmp_path / "train.json", "w") as f:
        json.dump(json_dataset_obj, f)
    reader.load_json(f.name)

    incompatible_task_data = copy.deepcopy(json_dataset_obj)
    incompatible_task_data["metadata"]["subtasks"][0]["output_mode"] = "tagging"
    with open(tmp_path / "test.json", "w") as f:
        json.dump(incompatible_task_data, f)

    with pytest.raises(
        ValueError, match="two different output modes in different datasets"
    ):
        reader.load_json(f.name)


def test_disjoint_labels(tmp_path, json_dataset_obj):
    reader = CnlpDataReader()

    with open(tmp_path / "train.json", "w") as f:
        json.dump(json_dataset_obj, f)
    reader.load_json(f.name)

    disjoint_labels_task_data = copy.deepcopy(json_dataset_obj)
    for instance in disjoint_labels_task_data["data"]:
        instance["task_a"] = instance["task_a"] + "_foo"
    with open(tmp_path / "test.json", "w") as f:
        json.dump(disjoint_labels_task_data, f)

    with pytest.raises(
        ValueError, match="disjoint sets of labels in different datasets"
    ):
        reader.load_json(f.name)

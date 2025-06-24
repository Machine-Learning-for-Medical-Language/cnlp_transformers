import json
import os
import random
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pytest
from lorem_text import lorem
from transformers.models.auto.tokenization_auto import AutoTokenizer


@pytest.fixture(autouse=True)
def disable_mps(monkeypatch):
    """Disable MPS for CI"""
    if os.getenv("CI", False):
        monkeypatch.setattr("torch._C._mps_is_available", lambda: False)


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)


@pytest.fixture
def random_cnlp_data_dir(request):
    """Create a temporary directory with randomly generated CNLP data."""

    marker: pytest.Mark = request.node.get_closest_marker("random_cnlp_data")

    tasks: Iterable[tuple[str, int]] = marker.kwargs["tasks"]
    n_train: int = marker.kwargs["n_train"]
    n_test: int = marker.kwargs["n_test"]
    n_dev: int = marker.kwargs["n_dev"]
    min_seq: int = marker.kwargs["min_seq"]
    max_seq: int = marker.kwargs["max_seq"]
    min_relations: int = marker.kwargs["min_relations"]
    max_relations: int = marker.kwargs["max_relations"]

    split_data: dict[str, Any] = {}

    for split, n in (("train", n_train), ("test", n_test), ("dev", n_dev)):
        if n == 0:
            continue
        data = []
        for row in range(n):
            seq_len = random.randint(min_seq, max_seq)

            row = {
                "id": f"{split}_{row}_id",
                "text": lorem.words(seq_len),
            }
            for task_idx, (task_type, n_labels) in enumerate(tasks):
                if task_type == "classification":
                    task_data = (
                        f"task_{task_idx}_label_{random.randint(0, n_labels - 1)}"
                    )
                elif task_type == "tagging":
                    task_data = " ".join(
                        [
                            f"task_{task_idx}_label_{random.randint(0, n_labels - 1)}"
                            for _ in range(seq_len)
                        ]
                    )
                elif task_type == "relations":
                    relations = []

                    for _ in range(random.randint(min_relations, max_relations)):
                        start, end = sorted(random.choices(range(seq_len), k=2))
                        label = (
                            f"task_{task_idx}_label_{random.randint(0, n_labels - 1)}"
                        )
                        relations.append(f"({start},{end},{label})")

                    if len(relations) == 0:
                        task_data = "None"
                    else:
                        task_data = " , ".join(relations)

                else:
                    raise ValueError(
                        f'unknown task type {task_type}, must be one of ("classification", "tagging", "relations")'
                    )

                row[f"task_{task_idx}_{task_type}"] = task_data
            data.append(row)

        metadata = {
            "subtasks": [
                {"task_name": f"task_{task_idx}_{task_type}", "output_mode": task_type}
                for task_idx, (task_type, _) in enumerate(tasks)
            ]
        }

        split_data[split] = {"data": data, "metadata": metadata}

    with tempfile.TemporaryDirectory() as tmpdir:
        for split, data in split_data.items():
            with open(Path(tmpdir) / f"{split}.json", "w") as f:
                json.dump(data, f)

        yield tmpdir


def random_cnlp_data_options(
    *,
    tasks: Iterable[tuple[Literal["classification", "tagging", "relations"], int]],
    n_train: int = 0,
    n_test: int = 0,
    n_dev: int = 0,
    min_seq: int = 1,
    max_seq: int = 100,
    min_relations: int = 0,
    max_relations: int = 3,
):
    """Decorator to configure randomly generated CNLP data provided by the `random_cnlp_data_dir` fixture.

    Args:
        tasks: An iterable of (task_type, n_labels) pairs.
        n_train: Number of samples in `train.json`. Defaults to 0.
        n_test: Number of samples in `test.json`. Defaults to 0.
        n_dev: Number of samples in `dev.json`. Defaults to 0.
        min_seq: Minimum random text sequence length. Defaults to 1.
        max_seq: Maximum random text sequence length. Defaults to 100.
        min_relations: Minimum number of relations per sample for relations tasks. Defaults to 0.
        max_relations: Maximum number of relations per sample for relations tasks. Defaults to 3.
    """
    return pytest.mark.random_cnlp_data(
        tasks=tasks,
        n_train=n_train,
        n_test=n_test,
        n_dev=n_dev,
        min_seq=min_seq,
        max_seq=max_seq,
        min_relations=min_relations,
        max_relations=max_relations,
    )

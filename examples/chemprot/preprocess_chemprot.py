import bisect
import itertools
import os
import re
from dataclasses import dataclass
from sys import argv
from typing import Any, Union

import polars as pl
from datasets import load_dataset
from datasets.dataset_dict import Dataset, DatasetDict
from rich.console import Console


def load_chemprot_dataset(cache_dir="./cache") -> DatasetDict:
    return load_dataset("bigbio/chemprot", "chemprot_full_source", cache_dir=cache_dir)


def clean_text(text: str):
    return (
        text.replace("&#039;", "'")
        .replace("\n", " <cr> ")
        .replace("\r", " <cr> ")
        .replace("\t", " ")
    )


def expand_to_word_boundaries(start: int, end: int, text: str):
    while start > 0 and text[start] != " ":
        start -= 1
    while text[start] == " ":
        start += 1
    while end < len(text) and text[end - 1] != " ":
        end += 1
    while text[end - 1] == " ":
        end -= 1
    return start, end


@dataclass
class Entity:
    eid: str
    start_word: int
    end_word: int
    label: str

    def tags(self):
        return [f"B-{self.label}"] + (
            [f"I-{self.label}"] * (self.end_word - self.start_word - 1)
        )


@dataclass
class Relation:
    arg1: Entity
    arg2: Entity
    label: str

    def cnlp_str(self):
        return f"({self.arg1.start_word},{self.arg2.start_word},{self.label})"


class ChemprotRow:
    def __init__(self, row: dict[str, Any]):
        self.pmid: str = row["pmid"]

        raw_text: str = row["text"]

        boundaries = set([0, len(raw_text)])
        for offsets in row["entities"]["offsets"]:
            boundaries.update(offsets)

        spans: list[str] = []
        new_offsets: dict[int, int] = {0: 0}
        clean_len = 0
        for start, end in itertools.pairwise(sorted(boundaries)):
            span = clean_text(raw_text[start:end])
            clean_len += len(span)
            new_offsets[end] = clean_len
            spans.append(span)

        self.text = "".join(spans)
        self.n_words = len(self.text.split(" "))
        self.entities: dict[str, Entity] = {}

        space_idxs = [m.start() for m in re.finditer(" ", self.text)]

        for eid, ent_label, (old_start, old_end) in zip(
            row["entities"]["id"], row["entities"]["type"], row["entities"]["offsets"]
        ):
            start = new_offsets[old_start]
            end = new_offsets[old_end]
            start, end = expand_to_word_boundaries(start, end, self.text)

            start_word = bisect.bisect_right(space_idxs, start)
            end_word = bisect.bisect_right(space_idxs, end)

            entity = Entity(
                eid=eid, start_word=start_word, end_word=end_word, label=ent_label
            )
            self.entities[eid] = entity

        self.relations: list[Relation] = []

        for rel_label, eid1, eid2 in zip(
            row["relations"]["type"], row["relations"]["arg1"], row["relations"]["arg2"]
        ):
            self.relations.append(
                Relation(
                    arg1=self.entities[eid1], arg2=self.entities[eid2], label=rel_label
                )
            )

    def _ner_str(self, entities: list[Entity]):
        tags: list[str] = []

        for entity in sorted(entities, key=lambda e: e.start_word):
            if entity.start_word < len(tags):
                continue  # ignore overlapping entities for tagging
            while entity.start_word > len(tags):
                tags.append("O")
            tags.extend(entity.tags())

        while len(tags) < self.n_words:
            tags.append("O")

        return " ".join(tags)

    def chemical_ner_str(self):
        return self._ner_str(
            [e for e in self.entities.values() if e.label.startswith("CHEMICAL")]
        )

    def gene_ner_str(self):
        return self._ner_str(
            [e for e in self.entities.values() if e.label.startswith("GENE")]
        )

    def relations_str(self):
        if len(self.relations) == 0:
            return "None"
        return " , ".join([r.cnlp_str() for r in self.relations])


def preprocess_data(split: Dataset):
    rows = [ChemprotRow(row) for row in split]

    return pl.DataFrame(
        {
            "id": [r.pmid for r in rows],
            "text": [r.text for r in rows],
            "chemical_ner": [r.chemical_ner_str() for r in rows],
            "gene_ner": [r.gene_ner_str() for r in rows],
            "end_to_end": [r.relations_str() for r in rows],
        }
    )


def main(out_dir: Union[str, os.PathLike]):
    console = Console()

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    with console.status("Loading dataset...") as st:
        dataset = load_chemprot_dataset()
        for split in ("train", "test", "validation"):
            st.update(f"Preprocessing {split} data...")
            preprocessed = preprocess_data(dataset[split])
            preprocessed.write_csv(
                os.path.join(out_dir, f"{split}.tsv"), separator="\t"
            )

    console.print(
        f"[green i]Preprocessed chemprot data saved to [repr.filename]{out_dir}[/]."
    )


if __name__ == "__main__":
    main(argv[1])

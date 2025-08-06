import logging
import os
from collections.abc import Iterable
from typing import Union

import polars as pl
import torch
from datasets import Dataset
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from typing_extensions import Self

from ..args.data_args import CnlpDataArguments
from ..data.analysis import make_preds_df
from ..data.predictions import CnlpPredictions
from ..data.preprocess import preprocess_raw_data
from ..data.task_info import CLASSIFICATION, RELATIONS, TAGGING, TaskInfo


class InputDocument(BaseModel):
    text: str
    entity_spans: Union[list[tuple[int, int]], None] = None

    def to_text_list(self):
        if self.entity_spans is None:
            return [self.text]

        text_list: list[str] = []
        for entity_start, entity_end in self.entity_spans:
            start = max(0, entity_start - 100)
            end = min(len(self.text), entity_end + 100)
            text_list.append(
                "".join(
                    [
                        self.text[start:entity_start],
                        "<e>",
                        self.text[entity_start:entity_end],
                        "</e>",
                        self.text[entity_end:end],
                    ]
                )
            )
        return text_list


class CnlpRestApp:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.resolve_device(device)
        self.setup_logger(logging.DEBUG)
        self.load_model()

    def resolve_device(self, device: str):
        self.device = device.lower()
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            try:
                torch.tensor([1.0], device=self.device)
            except:  # noqa: E722
                self.logger.warning(
                    f"Device is set to '{self.device}' but was not available; setting to 'cpu' and proceeding. If you have a GPU you need to debug why pytorch cannot see it."
                )
                self.device = "cpu"

    def setup_logger(self, log_level):
        self.logger = logging.getLogger(self.__module__)
        self.logger.setLevel(log_level)

    def load_model(self):
        training_args = TrainingArguments(
            output_dir="save_run/",
            save_strategy="no",
            per_device_eval_batch_size=8,
            do_predict=True,
        )

        if self.device == "mps":
            # pin_memory is unsupported on MPS, but defaults to True,
            # so we'll explicitly turn it off to avoid a warning.
            training_args.dataloader_pin_memory = False

        self.config = AutoConfig.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, config=self.config
        )
        self.model = AutoModel.from_pretrained(
            self.model_path, cache_dir=os.getenv("HF_CACHE"), config=self.config
        ).to(self.device)
        self.trainer = Trainer(model=self.model, args=training_args)

        self.tasks: list[TaskInfo] = []
        for task_idx, task_name in enumerate(self.config.finetuning_task):
            if self.config.tagger[task_name]:
                task_type = TAGGING
            elif self.config.relations[task_name]:
                task_type = RELATIONS
            else:
                task_type = CLASSIFICATION

            self.tasks.append(
                TaskInfo(
                    name=task_name,
                    type=task_type,
                    index=task_idx,
                    labels=tuple(self.config.label_dictionary[task_name]),
                )
            )

    def create_prediction_dataset(
        self,
        text: list[str],
        data_args: CnlpDataArguments,
    ):
        dataset = Dataset.from_dict({"text": text})

        return dataset.map(
            preprocess_raw_data,
            batched=True,
            load_from_cache_file=False,
            desc="Preprocessing raw input",
            batch_size=100,
            fn_kwargs={
                "inference_only": True,
                "tokenizer": self.tokenizer,
                "tasks": None,
                "max_length": data_args.max_seq_length,
                "hierarchical": self.config.model_type == "hier",
                "chunk_len": data_args.chunk_len or -1,
                "num_chunks": data_args.num_chunks or -1,
                "insert_empty_chunk_at_beginning": data_args.insert_empty_chunk_at_beginning,
            },
        )

    def predict(self, dataset: Dataset, data_args: CnlpDataArguments):
        raw_predictions = self.trainer.predict(dataset)
        return CnlpPredictions(
            dataset,
            raw_prediction=raw_predictions,
            tasks=self.tasks,
            data_args=data_args,
        )

    def format_predictions(self, predictions: CnlpPredictions):
        df = make_preds_df(predictions).select(["text", *[t.name for t in self.tasks]])

        for task in self.tasks:
            if task.type == CLASSIFICATION:
                df = df.with_columns(
                    pl.struct(
                        prediction=pl.col(task.name)
                        .struct.field("predictions")
                        .struct.field("values"),
                        probs=pl.col(task.name)
                        .struct.field("model_output")
                        .struct.field("probs")
                        .arr.to_struct(fields=task.labels),
                    ).alias(task.name)
                )
            elif task.type == TAGGING:
                df = df.with_columns(
                    pl.struct(
                        pl.col(task.name)
                        .struct.field("predictions")
                        .struct.field("spans")
                    ).alias(task.name)
                )
            elif task.type == RELATIONS:
                df = df.with_columns(
                    pl.struct(
                        pl.col(task.name)
                        .struct.field("predictions")
                        .struct.field("relations")
                    ).alias(task.name)
                )

        return df.to_dicts()

    def process(
        self,
        input_doc: InputDocument,
        max_seq_length: int = 128,
        chunk_len: Union[int, None] = None,
        num_chunks: Union[int, None] = None,
        insert_empty_chunk_at_beginning: bool = False,
    ):
        data_args = CnlpDataArguments(
            data_dir=[],
            max_seq_length=max_seq_length,
            chunk_len=chunk_len,
            num_chunks=num_chunks,
            insert_empty_chunk_at_beginning=insert_empty_chunk_at_beginning,
        )

        dataset = self.create_prediction_dataset(input_doc.to_text_list(), data_args)
        predictions = self.predict(dataset, data_args)
        return self.format_predictions(predictions)

    def router(self, prefix: str = ""):
        router = APIRouter(prefix=prefix)
        router.add_api_route("/process", self.process, methods=["POST"])
        return router

    def fastapi(self, router_prefix: str = ""):
        app = FastAPI()
        app.include_router(self.router(prefix=router_prefix))
        return app

    @classmethod
    def multi_app(cls, apps: Iterable[tuple[Self, str]]):
        multi_app = FastAPI()
        for app, router_prefix in apps:
            multi_app.include_router(app.router(router_prefix))
        return multi_app

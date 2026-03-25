import logging
from collections.abc import Iterable

import polars as pl
import torch
from datasets import Dataset
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from transformers.models.auto.modeling_auto import AutoModel
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from typing_extensions import Self

from ..data.analysis import make_preds_df
from ..data.cnlp_dataset import HierarchicalDataConfig, load_tokenizer
from ..data.predictions import CnlpPredictions
from ..data.preprocess import preprocess_raw_data
from ..data.task_info import CLASSIFICATION, RELATIONS, TAGGING, TaskInfo
from ..modeling.config.hierarchical_config import HierarchicalModelConfig
from ..modeling.load import try_load_config


class InputDocument(BaseModel):
    text: str
    entity_spans: list[tuple[int, int]] | None = None

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
        self.setup_logger(logging.INFO)
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
            output_dir="cnlp_rest/",
            save_strategy="no",
            per_device_eval_batch_size=8,
            do_predict=True,
        )

        if self.device == "mps":
            # pin_memory is unsupported on MPS, but defaults to True,
            # so we'll explicitly turn it off to avoid a warning.
            training_args.dataloader_pin_memory = False

        self.config = try_load_config(self.model_path)
        try:
            self.tokenizer = load_tokenizer(
                self.model_path,
                character_level=self.config.character_level,
            )
        except KeyError:
            self.tokenizer = load_tokenizer(
                self.config.encoder_name,
                character_level=self.config.character_level,
            )

        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(
            self.model_path,
            config=self.config,
        ).to(self.device)
        self.trainer = Trainer(model=self.model, args=training_args)

        self.tasks: list[TaskInfo] = self.config.tasks

    def create_prediction_dataset(
        self,
        text: list[str],
        max_seq_length: int = 128,
        hier_data_config: HierarchicalDataConfig | None = None,
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
                "max_length": max_seq_length,
                "hier_config": hier_data_config,
            },
        )

    def predict(self, dataset: Dataset, max_seq_length: int):
        raw_predictions = self.trainer.predict(dataset)
        return CnlpPredictions(
            dataset,
            raw_prediction=raw_predictions,
            tasks=self.tasks,
            max_seq_length=max_seq_length,
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
        chunk_len: int | None = None,
        num_chunks: int | None = None,
        prepend_empty_chunk: bool = False,
    ):
        if isinstance(self.config, HierarchicalModelConfig):
            hier_data_config = HierarchicalDataConfig(
                chunk_len=chunk_len,
                num_chunks=num_chunks,
                prepend_empty_chunk=prepend_empty_chunk,
            )
        else:
            hier_data_config = None

        dataset = self.create_prediction_dataset(
            input_doc.to_text_list(), max_seq_length, hier_data_config
        )
        predictions = self.predict(dataset, max_seq_length)
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

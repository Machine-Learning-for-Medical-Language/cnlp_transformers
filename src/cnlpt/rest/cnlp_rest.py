import logging
import re
from collections.abc import Iterable
from importlib.resources import files
from typing import Union

import polars as pl
import torch
from datasets import Dataset
from fastapi import APIRouter, FastAPI
from fastapi.responses import HTMLResponse
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

# Splits on sentence-ending punctuation followed by whitespace + uppercase
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


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
        hier_data_config: Union[HierarchicalDataConfig, None] = None,
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

    def _compute_attributions(
        self,
        text_list: list[str],
        max_seq_length: int = 128,
    ) -> list[dict[str, list[dict]]]:
        """Compute signed per-class input x gradient token-level attributions.

        Runs one forward pass, then one backward pass per class label.
        Per-token scores are the dot product of (gradient x embedding)
        summed over the hidden dimension, normalized per class to [-1, 1]
        within each sample.

        - Positive score: token pushes the model toward that class.
        - Negative score: token pushes the model away from that class.

        Only classification tasks are supported.

        Returns a list (one entry per input text) of dicts mapping task name
        to a list of {"token_id": int, "start": int, "end": int,
        "scores": {"label_a": float, ...}} objects, one per non-padding token.
        start/end are character offsets into the original input string, so
        text[start:end] gives the corresponding substring and
        "".join(text[s:e] for s, e in ...) reconstructs the original
        (excluding special tokens, which have start == end == 0).
        """
        if not hasattr(self.model.encoder, "embeddings"):
            raise NotImplementedError(
                "Input x gradient attribution is only supported for encoder "
                "architectures that expose an 'embeddings' submodule (e.g. BERT)."
            )

        encoding = self.tokenizer(
            text_list,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(self.device)
        # token offsets stay on CPU
        offset_mapping = encoding["offset_mapping"]  # (batch, seq, 2)

        # compute embeddings first without grad
        embed_kwargs: dict = {"input_ids": input_ids}
        if token_type_ids is not None:
            embed_kwargs["token_type_ids"] = token_type_ids
        with torch.no_grad():
            base_embeds = self.model.encoder.embeddings(**embed_kwargs)

        # forward pass from embeddings
        self.model.eval()
        inputs_embeds = base_embeds.detach().requires_grad_(True)
        output = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        batch_size = len(text_list)

        results_per_task: dict[str, list] = {}
        for task_ind, task in enumerate(self.tasks):
            if task.type != CLASSIFICATION:
                self.logger.warning(
                    f"Skipping token attributions for task {task.name}, since it is not a classification task."
                )
                continue

            task_logits = output.logits[task_ind]  # (batch, num_labels)

            # One backward pass per class, reusing the graph via retain_graph=True.
            class_attrs: list[torch.Tensor] = []
            for c in range(len(task.labels)):
                if inputs_embeds.grad is not None:
                    inputs_embeds.grad.zero_()
                task_logits[:, c].sum().backward(retain_graph=True)
                # Signed dot product over hidden dim → (batch, seq)
                signed_attr = (inputs_embeds.grad * inputs_embeds.detach()).sum(dim=-1)
                class_attrs.append(signed_attr.detach())

            # Stack to (batch, seq, num_labels); mask padding.
            attrs = torch.stack(class_attrs, dim=-1)
            attrs = attrs * attention_mask.float().unsqueeze(-1)

            # Normalize each class independently per sample to [-1, 1].
            norm_denom = attrs.abs().max(dim=1, keepdim=True).values.clamp(min=1e-10)
            attrs = (attrs / norm_denom).cpu()

            input_ids_cpu = input_ids.cpu()
            attn_cpu = attention_mask.cpu()

            task_results = []
            for i in range(batch_size):
                task_results.append(
                    [
                        {
                            "token_id": input_ids_cpu[i, j].item(),
                            "start": offset_mapping[i, j, 0].item(),
                            "end": offset_mapping[i, j, 1].item(),
                            "scores": {
                                label: attrs[i, j, c].item()
                                for c, label in enumerate(task.labels)
                            },
                        }
                        for j, m in enumerate(attn_cpu[i])
                        if m == 1
                    ]
                )
            results_per_task[task.name] = task_results

        return [
            {
                task_name: task_results[i]
                for task_name, task_results in results_per_task.items()
            }
            for i in range(batch_size)
        ]

    def _compute_sentence_attributions(
        self,
        text_list: list[str],
        max_seq_length: int = 128,
    ) -> list[dict[str, list[dict]]]:
        """Leave-one-sentence-out ablation for each task.

        For each sentence and each class label, removes the sentence from the
        input and computes:
            score = p(class | full text) - p(class | text without sentence)

        Positive: the sentence supports that class.
        Negative: the sentence suppresses it.

        Only classification tasks are supported.

        Returns a list (one per input text) of dicts mapping task name to a list
        of {"sentence": str, "scores": {"label_a": float, ...}} objects.
        """
        self.model.eval()
        all_results: list[dict[str, list[dict]]] = []

        for text in text_list:
            sentences = _split_sentences(text)

            if len(sentences) < 2:
                all_results.append(
                    {
                        task.name: [
                            {
                                "sentence": s,
                                "scores": {label: 0.0 for label in task.labels},
                            }
                            for s in sentences
                        ]
                        for task in self.tasks
                        if task.type == CLASSIFICATION
                    }
                )
                continue

            # index 0 = full text, indices 1..N = each sentence removed
            ablated = [
                " ".join(s for j, s in enumerate(sentences) if j != i)
                for i in range(len(sentences))
            ]
            batch_texts = [text, *ablated]

            encoding = self.tokenizer(
                batch_texts,
                max_length=max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)

            task_results: dict[str, list[dict]] = {}
            for task_ind, task in enumerate(self.tasks):
                if task.type != CLASSIFICATION:
                    self.logger.warning(
                        f"Skipping sentence attributions for task {task.name}, since it is not a classification task."
                    )
                    continue

                # probs[0] = full text, probs[1..N] = each sentence ablated
                probs = torch.softmax(
                    output.logits[task_ind], dim=-1
                )  # (batch, num_labels)
                baseline = probs[0]  # (num_labels,)

                task_results[task.name] = [
                    {
                        "sentence": sent,
                        "scores": {
                            label: round((baseline[c] - probs[i + 1, c]).item(), 4)
                            for c, label in enumerate(task.labels)
                        },
                    }
                    for i, sent in enumerate(sentences)
                ]

            all_results.append(task_results)

        return all_results

    def process(
        self,
        input_doc: InputDocument,
        max_seq_length: int = 128,
        chunk_len: Union[int, None] = None,
        num_chunks: Union[int, None] = None,
        prepend_empty_chunk: bool = False,
        return_attributions: bool = False,
        return_sentence_attributions: bool = False,
    ):
        if isinstance(self.config, HierarchicalModelConfig):
            hier_data_config = HierarchicalDataConfig(
                chunk_len=chunk_len,
                num_chunks=num_chunks,
                prepend_empty_chunk=prepend_empty_chunk,
            )
        else:
            hier_data_config = None

        text_list = input_doc.to_text_list()
        dataset = self.create_prediction_dataset(
            text_list, max_seq_length, hier_data_config
        )
        predictions = self.predict(dataset, max_seq_length)
        results = self.format_predictions(predictions)

        if return_attributions:
            attributions = self._compute_attributions(text_list, max_seq_length)
            for result, doc_attrs in zip(results, attributions):
                for task_name, token_scores in doc_attrs.items():
                    if task_name in result:
                        result[task_name]["attributions"] = token_scores

        if return_sentence_attributions:
            sent_attrs = self._compute_sentence_attributions(text_list, max_seq_length)
            for result, doc_attrs in zip(results, sent_attrs):
                for task_name, sentence_scores in doc_attrs.items():
                    if task_name in result:
                        result[task_name]["sentence_attributions"] = sentence_scores

        return results

    def router(self, prefix: str = ""):
        _visualizer_html = files("cnlpt.rest").joinpath("visualizer.html").read_text()

        async def visualizer():
            return HTMLResponse(_visualizer_html)

        router = APIRouter(prefix=prefix)
        router.add_api_route("/", visualizer, methods=["GET"], include_in_schema=False)
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

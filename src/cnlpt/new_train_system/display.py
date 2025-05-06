import os
from typing import Any, Union

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from ..args import CnlpDataArguments, CnlpModelArguments, CnlpTrainingArguments

console = Console()


def _val_fmt(x):
    ndigits = 4
    if isinstance(x, (float, int)):
        return round(x, ndigits)
    elif isinstance(x, (list, np.ndarray)):
        return np.asarray(np.round(x, ndigits)).tolist()
    else:
        return x


class TrainSystemDisplay:
    def __init__(
        self,
        model_args: CnlpModelArguments,
        data_args: CnlpDataArguments,
        training_args: CnlpTrainingArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.train_metrics: list[dict[str, float]] = []
        self.eval_metrics: Union[dict[str, Any], None] = None
        self.best_eval_metrics: Union[dict[str, Any], None] = None

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
            auto_refresh=False,
            transient=True,
        )

        self.live = Live(self.panel(), console=console)

    @property
    def title(self):
        return "CNLP Transformers - Train System"

    @property
    def subtitle(self):
        if self.training_args.output_dir is None:
            return "?"
        logfile = os.path.join(
            os.path.abspath(self.training_args.output_dir), "train_system.log"
        )
        return f"Training log: {logfile}"

    def eval_metrics_table(self, metrics: Union[dict[str, Any], None]):
        if metrics is None:
            return "[dim italic]waiting for evaluation"
        grid = Table.grid(padding=(0, 1))

        general_metrics: dict[str, Any] = {}
        task_metrics: dict[str, dict[str, Any]] = {}

        for k, v in metrics.items():
            if isinstance(v, dict):
                task_metrics[k] = v
            else:
                general_metrics[k] = v

        general_metrics = {"epoch": general_metrics.pop("epoch")} | general_metrics
        task_metrics = {"general_metrics": general_metrics} | task_metrics

        for task_name, m in task_metrics.items():
            items = []
            for k, v in m.items():
                items.append(
                    f"[json.key]{k}[/json.key]: [json.number]{_val_fmt(v)}[/json.number]"
                )
            color = (
                "[bold magenta]" if task_name == "general_metrics" else "[bold yellow]"
            )
            grid.add_row(
                f"{color}{task_name}",
                " | ".join(items),
            )
        return grid

    def format_train_metrics(self):
        if len(self.train_metrics) == 0:
            return "[dim italic]waiting for training log step"
        elif len(self.train_metrics) == 1:
            return " | ".join(
                [
                    f"[json.key]{k}[/json.key]: [json.number]{_val_fmt(v)}[/json.number]"
                    for k, v in self.train_metrics[-1].items()
                ]
            )
        else:
            cur, prev = self.train_metrics[-1], self.train_metrics[-2]
            items = []
            for key in cur:
                delta = cur[key] - prev[key]
                if delta != 0 and key != "epoch":
                    prefix = "+" if delta > 0 else ""
                    items.append(
                        f"[json.key]{key}[/json.key]: [json.number]{_val_fmt(cur[key])}[/json.number] ([json.number]{prefix}{_val_fmt(delta)}[/json.number])"
                    )
                else:
                    items.append(
                        f"[json.key]{key}[/json.key]: [json.number]{_val_fmt(cur[key])}[/json.number]"
                    )
            return " | ".join(items)

    def body(self):
        meta = Table.grid(padding=(0, 1))
        meta.add_column(style="blue", justify="right")
        meta.add_column()
        out_dir_abspath = (
            os.path.abspath(self.training_args.output_dir)
            if self.training_args.output_dir is not None
            else "?"
        )
        meta.add_row("Output dir:", out_dir_abspath)
        meta.add_row("Dataset:", ", ".join(self.data_args.data_dir))
        meta.add_row("Encoder:", str(self.model_args.encoder_name))
        meta.add_row("Model type:", str(self.model_args.model))

        stats = Table.grid(padding=(1, 1))
        stats.add_column(style="blue", justify="right")
        stats.add_column()
        stats.add_row("Train metrics:", self.format_train_metrics())
        stats.add_row("Eval metrics:", self.eval_metrics_table(self.eval_metrics))
        stats.add_row(
            "Best eval metrics:", self.eval_metrics_table(self.best_eval_metrics)
        )

        body = Table.grid(expand=True, padding=(1, 0))
        body.add_column()
        body.add_row(meta)
        body.add_row(self.progress)
        body.add_row(stats)

        return body

    def panel(self):
        panel = Panel(
            self.body(),
            title=self.title,
            subtitle=self.subtitle,
            title_align="left",
            subtitle_align="left",
            padding=(1, 2),
            expand=True,
        )
        return panel

    def update(self):
        self.progress.refresh()
        self.live.update(self.panel())

    def __enter__(self):
        self.live.__enter__()

        return self

    def __exit__(self, *args):
        return self.live.__exit__(*args)

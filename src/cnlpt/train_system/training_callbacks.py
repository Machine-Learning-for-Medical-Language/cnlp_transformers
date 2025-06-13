from dataclasses import asdict
from typing import Union

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.training_args import TrainingArguments

from ..args import CnlpDataArguments, CnlpModelArguments, CnlpTrainingArguments
from .display import TrainSystemDisplay
from .log import logger


class DisplayCallback(TrainerCallback):
    def __init__(
        self,
        display: TrainSystemDisplay,
    ):
        self.display = display
        self.progress = self.display.progress
        self.training_task = None
        self.epoch_task = None
        self.eval_task = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        self.training_task = self.progress.add_task("Training")
        self.epoch_task = self.progress.add_task(
            f"Epoch 1/{int(args.num_train_epochs)}"
        )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.epoch_task is not None:
            self.progress.remove_task(self.epoch_task)
            self.epoch_task = None

        if self.training_task is not None:
            self.progress.remove_task(self.training_task)
            self.training_task = None

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return

        steps_per_epoch = state.max_steps // args.num_train_epochs

        if self.training_task is not None:
            self.progress.update(
                self.training_task,
                total=state.max_steps,
                completed=state.global_step,
            )

        if self.epoch_task is not None:
            cur_epoch, epoch_progress = divmod(state.global_step, steps_per_epoch)

            # unless we're just starting (N == 0), prefer rendering
            # 100% for epoch N instead of 0% for epoch N + 1
            if epoch_progress == 0 and cur_epoch > 0:
                epoch_progress = steps_per_epoch
                cur_epoch -= 1

            self.progress.update(
                task_id=self.epoch_task,
                total=steps_per_epoch,
                completed=int(epoch_progress),
                description=f"Epoch {int(cur_epoch + 1)}/{int(args.num_train_epochs)}",
            )

        self.display.update()

    def on_prediction_step(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        eval_dataloader=None,
        **kwargs,
    ):
        if not state.is_world_process_zero or eval_dataloader is None:
            return

        if self.eval_task is None:
            self.eval_task = self.progress.add_task(
                "Evaluation", total=len(eval_dataloader)
            )

        self.progress.advance(self.eval_task, advance=1)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Union[dict[str, float], None] = None,
        **kwargs,
    ):
        if logs is not None and "loss" in logs:
            self.display.train_metrics.append(logs)
            self.display.update()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        if self.eval_task is not None:
            self.progress.remove_task(self.eval_task)
            self.eval_task = None
        self.display.eval_metrics = metrics

        best = self.display.best_eval_metrics
        tgt_metric = args.metric_for_best_model
        if (
            len(best) == 0
            or (args.greater_is_better and metrics[tgt_metric] > best[tgt_metric])
            or (not args.greater_is_better and metrics[tgt_metric] < best[tgt_metric])
        ):
            self.display.best_eval_metrics = metrics

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        if not state.is_world_process_zero:
            return
        if self.eval_task is not None:
            self.progress.remove_task(self.eval_task)
            self.eval_task = None


class BasicLoggingCallback(TrainerCallback):
    def __init__(
        self,
        model_args: CnlpModelArguments,
        data_args: CnlpDataArguments,
        training_args: CnlpTrainingArguments,
    ):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logger.info("*** TRAIN SYSTEM ARGS ***")
        for args_data, prefix in (
            (self.model_args, "model_args"),
            (self.data_args, "data_args"),
            (self.training_args, "training_args"),
        ):
            for arg, val in sorted(asdict(args_data).items()):
                logger.info("%s.%s: %s", prefix, arg, val)
        logger.info("*** STARTING TRAINING ***")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Union[dict[str, float], None] = None,
        **kwargs,
    ):
        if logs is not None:
            logger.info(logs)

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.epoch is None:
            return
        logger.info("Starting epoch %s", int(state.epoch) + 1)

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.epoch is None:
            return
        logger.info("Epoch %s complete", int(state.epoch))

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logger.info("*** TRAINING COMPLETE ***")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        logger.info("New best model saved! Current step: %s", state.global_step)

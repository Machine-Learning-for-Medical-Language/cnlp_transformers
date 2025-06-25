from dataclasses import asdict
from typing import Union

from transformers.trainer_callback import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, SaveStrategy
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
        self.current_eval_step: int = 0

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.display.start_training(total_epochs=int(args.num_train_epochs))
        self.display.update()

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.display.finish_training()
        self.display.update()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.display.training_progress(
            completed_steps=state.global_step,
            total_steps=state.max_steps,
        )

        steps_per_epoch = state.max_steps // args.num_train_epochs
        cur_epoch, epoch_progress = divmod(state.global_step, steps_per_epoch)

        # unless we're just starting (N == 0), prefer rendering
        # 100% for epoch N instead of 0% for epoch N + 1
        if epoch_progress == 0 and cur_epoch > 0:
            epoch_progress = steps_per_epoch
            cur_epoch -= 1

        self.display.epoch_progress(
            epoch=int(cur_epoch + 1),
            total_epochs=int(args.num_train_epochs),
            completed_steps=int(epoch_progress),
            total_steps=steps_per_epoch,
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
        if eval_dataloader is None:
            return

        self.current_eval_step += 1
        self.display.eval_progress(
            completed_steps=self.current_eval_step,
            total_steps=len(eval_dataloader),
        )
        self.display.update()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float],
        **kwargs,
    ):
        self.display.finish_eval()
        self.current_eval_step = 0

        self.display.eval_metrics = metrics
        best = self.display.best_eval_metrics
        tgt_metric = args.metric_for_best_model
        if (
            len(best) == 0
            or (args.greater_is_better and metrics[tgt_metric] > best[tgt_metric])
            or (not args.greater_is_better and metrics[tgt_metric] < best[tgt_metric])
        ):
            self.display.best_eval_metrics = metrics

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

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if args.save_strategy == SaveStrategy.BEST:
            self.display.best_checkpoint = (
                f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
        else:
            self.display.best_checkpoint = state.best_model_checkpoint

    def on_predict(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        **kwargs,
    ):
        self.display.finish_eval()
        self.current_eval_step = 0


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
        logger.info("Model saved! Current step: %s", state.global_step)

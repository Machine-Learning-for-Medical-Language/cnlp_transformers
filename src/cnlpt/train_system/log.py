import logging
import logging.config
import os
import re

from transformers import logging as transformers_logging

from ..args import CnlpTrainingArguments

logger = logging.getLogger("cnlpt.train_system")


class CleanWhitespaceFormatter(logging.Formatter):
    def format(self, record):
        original = super().format(record)
        cleaned = re.sub(r"\s+", " ", original).strip()
        return cleaned


def configure_logger_for_training(training_args: CnlpTrainingArguments):
    assert training_args.output_dir is not None
    log_file = os.path.join(training_args.output_dir, "train_system.log")
    level = "INFO" if training_args.local_rank in (-1, 0) else "WARNING"
    handlers = ["logfile"]
    if not training_args.rich_display:
        handlers.append("stdout")
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
            "no_newlines": {
                "()": CleanWhitespaceFormatter,
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            },
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "logfile": {
                "class": "logging.FileHandler",
                "formatter": "no_newlines",
                "filename": log_file,
                "mode": "a",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "root": {
                "level": level,
                "handlers": handlers,
            },
            "transformers": {"propagate": True},
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)
    transformers_logging.set_verbosity_info()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

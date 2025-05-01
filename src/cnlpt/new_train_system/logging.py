import logging
import logging.config
import os

from ..args import CnlpTrainingArguments

logger = logging.getLogger("cnlp_train_system")


def configure_logger_for_training(training_args: CnlpTrainingArguments):
    log_file = os.path.join(training_args.output_dir, "train_system.log")
    level = "INFO" if training_args.local_rank in (-1, 0) else "WARNING"
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            "logfile": {
                "class": "logging.FileHandler",
                "formatter": "simple",
                "filename": log_file,
                "mode": "a",
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "root": {
                "level": level,
                "handlers": [
                    # "stdout",
                    "logfile",
                ],
            }
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

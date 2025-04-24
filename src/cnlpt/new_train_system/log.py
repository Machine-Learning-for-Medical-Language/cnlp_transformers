import logging
import logging.config

from ..args import CnlpTrainingArguments

logger = logging.getLogger("cnlp_train_system")
# TODO https://github.com/huggingface/transformers/blob/main/docs/source/en/trainer.md#logging


def configure_logger_for_training(training_args: CnlpTrainingArguments):
    level = "INFO" if training_args.local_rank in (-1, 0) else "WARNING"
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "format": "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S%z",
            }
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "root": {
                "level": level,
                "handlers": ["stdout"],
            }
        },
    }
    logging.config.dictConfig(LOGGING_CONFIG)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )

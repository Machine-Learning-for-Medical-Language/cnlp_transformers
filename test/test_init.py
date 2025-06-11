"""
Test suite for initializing the library
"""


def test_init():
    """
    Test cnlpt can be imported and is the correct package (to be refined later)
    """
    import cnlpt

    assert cnlpt.__package__ == "cnlpt"


def test_init_models():
    import cnlpt.models

    assert cnlpt.models.__package__ == "cnlpt.models"
    assert cnlpt.models.__all__ == [
        "CnlpConfig",
        "CnlpModelForClassification",
        "HierarchicalModel",
    ]

    import cnlpt.models.baseline

    assert cnlpt.models.baseline.__package__ == "cnlpt.models.baseline"
    assert cnlpt.models.baseline.__all__ == [
        "CnnSentenceClassifier",
        "LstmSentenceClassifier",
    ]


def test_init_train_system():
    import cnlpt.train_system

    assert cnlpt.train_system.__package__ == "cnlpt.train_system"
    assert cnlpt.train_system.__all__ == ["CnlpTrainSystem"]


def test_init_data():
    import cnlpt.data

    assert cnlpt.data.__package__ == "cnlpt.data"
    assert cnlpt.data.__all__ == [
        "CLASSIFICATION",
        "RELATIONS",
        "TAGGING",
        "CnlpDataset",
        "TaskInfo",
        "get_task_type",
        "preprocess_raw_data",
    ]


def test_init_args():
    import cnlpt.args

    assert cnlpt.args.__package__ == "cnlpt.args"
    assert cnlpt.args.__all__ == [
        "CnlpDataArguments",
        "CnlpModelArguments",
        "CnlpTrainingArguments",
        "parse_args_dict",
        "parse_args_from_argv",
        "parse_args_json_file",
        "preprocess_args",
    ]


def test_init_api():
    import cnlpt.api

    assert cnlpt.api.__package__ == "cnlpt.api"
    assert cnlpt.api.__all__ == ["MODEL_TYPES", "get_rest_app"]

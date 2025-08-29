# ruff: noqa: F401
"""
Test suite for initializing the library
"""


def test_init():
    """
    Test cnlpt can be imported and is the correct package (to be refined later)
    """
    import cnlpt


def test_init_models():
    import cnlpt.modeling
    import cnlpt.modeling.config
    import cnlpt.modeling.models


def test_init_train_system():
    import cnlpt.train_system


def test_init_data():
    import cnlpt.data


def test_init_rest():
    import cnlpt.rest

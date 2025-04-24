"""
Test suite for initializing the library
"""


def test_init():
    """
    Test cnlpt can be imported and is the correct package (to be refined later)
    """
    import cnlpt

    assert cnlpt.__package__ == "cnlpt"


def test_torch_numpy_available():
    """
    Test that NumPy is available to PyTorch (#91)
    """
    import torch

    t = torch.tensor(
        [[3.5026, -3.2149], [3.5615, -3.3653], [-2.5377, 1.8398], [-2.5859, 2.0284]]
    )
    try:
        t.numpy()
    except RuntimeError as e:
        assert False, e.args[0] + " (check for UserWarning)"


def test_import_sklearn():
    """
    Test that sklearn can be imported (#85)
    """
    from importlib.util import find_spec

    assert find_spec("sklearn") is not None


class TestSubmodulesPresent:
    """
    Test that all submodules are defined and can be imported (#88)
    """

    def test_BaselineModels_present(self):
        import cnlpt.models.baseline

        assert cnlpt.models.baseline.__package__ == "cnlpt.models.baseline"

    def test_cnlp_data_present(self):
        import cnlpt.data

        assert cnlpt.data.__package__ == "cnlpt.data"

    def test_cnlp_processors_present(self):
        import cnlpt.data.cnlp_datasets

        assert cnlpt.data.cnlp_datasets.__package__ == "cnlpt.data.cnlp_datasets"

    def test_CnlpModelForClassification_present(self):
        import cnlpt.models.cnlp

        assert cnlpt.models.cnlp.__package__ == "cnlpt.models"

    def test_HierarchicalTransformer_present(self):
        import cnlpt.models.hierarchical

        assert cnlpt.models.hierarchical.__package__ == "cnlpt.models"

    def test_thyme_eval_present(self):
        import cnlpt.thyme_eval

        assert cnlpt.thyme_eval.__package__ == "cnlpt"

    def test_train_system_present(self):
        import cnlpt.train_system

        assert cnlpt.train_system.__package__ == "cnlpt"

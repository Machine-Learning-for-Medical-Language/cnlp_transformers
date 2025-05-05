"""
Test suite for initializing the library
"""


def test_init():
    """
    Test cnlpt can be imported and is the correct package (to be refined later)
    """
    import cnlpt

    assert cnlpt.__package__ == "cnlpt"


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

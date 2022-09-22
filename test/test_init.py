"""
Test suite for initializing the library
"""


def test_init():
    """
    Test cnlpt can be imported and is the correct package (to be refined later)
    """
    import cnlpt
    assert cnlpt.__package__ == 'cnlpt'


class TestSubmodulesPresent:
    """
    Test that all submodules are defined and can be imported
    """
    def test_BaselineModels_present(self):
        import cnlpt.BaselineModels
        assert cnlpt.BaselineModels.__package__ == 'cnlpt'

    def test_cnlp_data_present(self):
        import cnlpt.cnlp_data
        assert cnlpt.cnlp_data.__package__ == 'cnlpt'

    def test_cnlp_processors_present(self):
        import cnlpt.cnlp_processors
        assert cnlpt.cnlp_processors.__package__ == 'cnlpt'

    def test_CnlpModelForClassification_present(self):
        import cnlpt.CnlpModelForClassification
        assert cnlpt.CnlpModelForClassification.__package__ == 'cnlpt'

    def test_HierarchicalTransformer_present(self):
        import cnlpt.HierarchicalTransformer
        assert cnlpt.HierarchicalTransformer.__package__ == 'cnlpt'
    def test_thyme_eval_present(self):
        import cnlpt.thyme_eval
        assert cnlpt.thyme_eval.__package__ == 'cnlpt'

    def test_train_system_present(self):
        import cnlpt.train_system
        assert cnlpt.train_system.__package__ == 'cnlpt'

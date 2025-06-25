from ..common.fixtures import random_cnlp_data_options


@random_cnlp_data_options(
    tasks=(("classification", 3), ("tagging", 3), ("relations", 3))
)
def test_init_train_system(random_cnlp_train_system):
    pass

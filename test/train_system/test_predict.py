from ..common.fixtures import random_cnlp_data_options


@random_cnlp_data_options(
    tasks=(("classification", 3), ("tagging", 3), ("relations", 3)),
    n_train=5,
    n_test=5,
    n_dev=5,
)
def test_predict(random_cnlp_train_system):
    random_cnlp_train_system.predict()

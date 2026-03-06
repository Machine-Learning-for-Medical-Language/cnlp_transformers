from enum import Enum

from .config import (
    CnnModelConfig,
    HierarchicalModelConfig,
    LstmModelConfig,
    ProjectionModelConfig,
)
from .models import CnnModel, HierarchicalModel, LstmModel, ProjectionModel


class ModelType(str, Enum):
    CNN = "cnn"
    LSTM = "lstm"
    HIER = "hier"
    PROJ = "proj"

    @property
    def config_class(self):
        if self == ModelType.CNN:
            return CnnModelConfig
        if self == ModelType.LSTM:
            return LstmModelConfig
        if self == ModelType.HIER:
            return HierarchicalModelConfig
        if self == ModelType.PROJ:
            return ProjectionModelConfig

    @property
    def model_class(self):
        if self == ModelType.CNN:
            return CnnModel
        if self == ModelType.LSTM:
            return LstmModel
        if self == ModelType.HIER:
            return HierarchicalModel
        if self == ModelType.PROJ:
            return ProjectionModel


class ClassificationMode(str, Enum):
    CLS = "cls"
    TAGGED = "tagged"

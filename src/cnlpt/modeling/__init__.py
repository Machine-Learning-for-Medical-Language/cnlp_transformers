from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModel

from .config import (
    CnnModelConfig,
    HierarchicalModelConfig,
    LstmModelConfig,
    ProjectionModelConfig,
)
from .models import CnnModel, HierarchicalModel, LstmModel, ProjectionModel
from .types import ClassificationMode, ModelType

__all__ = [
    "ClassificationMode",
    "CnnModel",
    "CnnModelConfig",
    "HierarchicalModel",
    "HierarchicalModelConfig",
    "LstmModel",
    "LstmModelConfig",
    "ModelType",
    "ProjectionModel",
    "ProjectionModelConfig",
]


AutoConfig.register("cnlpt.proj", ProjectionModelConfig)
AutoModel.register(ProjectionModelConfig, ProjectionModel)

AutoConfig.register("cnlpt.cnn", CnnModelConfig)
AutoModel.register(CnnModelConfig, CnnModel)

AutoConfig.register("cnlpt.hier", HierarchicalModelConfig)
AutoModel.register(HierarchicalModelConfig, HierarchicalModel)

AutoConfig.register("cnlpt.lstm", LstmModelConfig)
AutoModel.register(LstmModelConfig, LstmModel)

# TODO(ian) It would be REALLY nice if we could load legacy models...

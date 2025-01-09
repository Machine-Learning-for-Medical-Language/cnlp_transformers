from transformers import AutoConfig, AutoModel

from .cnlp import CnlpConfig, CnlpModelForClassification
from .hierarchical import HierarchicalModel

__all__ = ["CnlpConfig", "CnlpModelForClassification", "HierarchicalModel"]

AutoConfig.register("cnlpt", CnlpConfig)
AutoModel.register(CnlpConfig, (CnlpModelForClassification, HierarchicalModel))

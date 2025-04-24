from typing import Literal

MTL = "mtl"
CLASSIFICATION = "classification"
TAGGING = "tagging"
RELEX = "relations"

# For backwards compatibility
mtl = MTL
classification = CLASSIFICATION
tagging = TAGGING
relex = RELEX

TaskType = Literal["classification", "tagging", "relations"]

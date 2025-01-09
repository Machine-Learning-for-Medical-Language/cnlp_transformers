import json
from dataclasses import asdict, dataclass
from typing import Union


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        event_tokens: (Optional)
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: list[int]
    attention_mask: Union[list[int], None] = None
    token_type_ids: Union[list[int], None] = None
    event_tokens: Union[list[int], None] = None
    label: list[Union[int, float, list[int], list[tuple[int]], None]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(asdict(self)) + "\n"


@dataclass(frozen=True)
class HierarchicalInputFeatures:
    """
    A single set of features of data for the hierarchical model.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        event_tokens: (Optional)
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: list[list[int]]
    attention_mask: Union[list[list[int]], None] = None
    token_type_ids: Union[list[list[int]], None] = None
    event_tokens: Union[list[list[int]], None] = None
    label: list[Union[int, float, list[int], list[tuple[int]], None]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(asdict(self)) + "\n"

"""
Tests for per-class token and sentence attribution methods on the REST API.
"""

import pytest
from fastapi.testclient import TestClient

from cnlpt.rest.cnlp_rest import CnlpRestApp, InputDocument

NEGATION_MODEL = "mlml-chip/negation_pubmedbert_sharpseed"

# Single sentence — used for token attribution tests and the single-sentence
# edge case in sentence attribution tests.
SINGLE_SENTENCE = "The patient denied any chest pain."

# Three sentences — used to verify sentence attribution returns one entry per
# sentence and that multi-sentence scores are non-trivially computed.
MULTI_SENTENCE = (
    "The patient denied any chest pain. "
    "She reported no shortness of breath. "
    "There were no signs of infection."
)


@pytest.fixture(scope="module")
def negation_client():
    with TestClient(CnlpRestApp(NEGATION_MODEL).fastapi()) as client:
        yield client


# ---------------------------------------------------------------------------
# Token attributions
# ---------------------------------------------------------------------------


class TestTokenAttributions:
    @pytest.fixture(scope="class")
    def response(self, negation_client):
        doc = InputDocument(text=SINGLE_SENTENCE)
        r = negation_client.post(
            "/process?return_attributions=true",
            json=doc.model_dump(),
        )
        r.raise_for_status()
        return r.json()

    def test_attributions_present(self, response):
        assert "attributions" in _task(response)

    def test_attributions_absent_by_default(self, negation_client):
        doc = InputDocument(text=SINGLE_SENTENCE)
        r = negation_client.post("/process", json=doc.model_dump())
        r.raise_for_status()
        assert "attributions" not in _task(r.json())

    def test_token_dict_keys(self, response):
        for tok in _task(response)["attributions"]:
            assert set(tok.keys()) == {"token_id", "start", "end", "scores"}

    def test_token_id_is_int(self, response):
        for tok in _task(response)["attributions"]:
            assert isinstance(tok["token_id"], int)

    def test_offsets_are_non_negative_and_ordered(self, response):
        for tok in _task(response)["attributions"]:
            assert tok["start"] >= 0
            assert tok["end"] >= tok["start"]

    def test_all_labels_have_a_score_per_token(self, response):
        result = response[0]
        task_name = _task_name(result)
        expected_labels = set(result[task_name]["probs"].keys())
        for tok in result[task_name]["attributions"]:
            assert set(tok["scores"].keys()) == expected_labels

    def test_scores_are_in_range(self, response):
        for tok in _task(response)["attributions"]:
            for score in tok["scores"].values():
                assert -1.0 <= score <= 1.0

    def test_special_tokens_have_zero_offsets(self, response):
        # [CLS] and [SEP] are always present and should have start == end == 0.
        special = [
            t
            for t in _task(response)["attributions"]
            if t["start"] == 0 and t["end"] == 0
        ]
        assert len(special) >= 2

    def test_non_special_token_offsets_are_within_input(self, response):
        text = response[0]["text"]
        real_tokens = [
            t
            for t in _task(response)["attributions"]
            if not (t["start"] == 0 and t["end"] == 0)
        ]
        assert len(real_tokens) > 0
        for tok in real_tokens:
            assert 0 <= tok["start"] < tok["end"] <= len(text)


# ---------------------------------------------------------------------------
# Sentence attributions
# ---------------------------------------------------------------------------


class TestSentenceAttributions:
    @pytest.fixture(scope="class")
    def response(self, negation_client):
        doc = InputDocument(text=MULTI_SENTENCE)
        r = negation_client.post(
            "/process?return_sentence_attributions=true",
            json=doc.model_dump(),
        )
        r.raise_for_status()
        return r.json()

    def test_sentence_attributions_present(self, response):
        assert "sentence_attributions" in _task(response)

    def test_sentence_attributions_absent_by_default(self, negation_client):
        doc = InputDocument(text=MULTI_SENTENCE)
        r = negation_client.post("/process", json=doc.model_dump())
        r.raise_for_status()
        assert "sentence_attributions" not in _task(r.json())

    def test_sentence_dict_keys(self, response):
        for sent in _task(response)["sentence_attributions"]:
            assert set(sent.keys()) == {"sentence", "scores"}

    def test_all_labels_have_a_score_per_sentence(self, response):
        result = response[0]
        task_name = _task_name(result)
        expected_labels = set(result[task_name]["probs"].keys())
        for sent in result[task_name]["sentence_attributions"]:
            assert set(sent["scores"].keys()) == expected_labels

    def test_sentence_count_matches_input(self, response):
        assert len(_task(response)["sentence_attributions"]) == 3

    def test_single_sentence_scores_are_zero(self, negation_client):
        # With only one sentence, ablation has nothing to remove — all scores
        # should be 0.0.
        doc = InputDocument(text=SINGLE_SENTENCE)
        r = negation_client.post(
            "/process?return_sentence_attributions=true",
            json=doc.model_dump(),
        )
        r.raise_for_status()
        sents = _task(r.json())["sentence_attributions"]
        assert len(sents) == 1
        for score in sents[0]["scores"].values():
            assert score == 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_name(result: dict) -> str:
    return next(k for k in result if k != "text")


def _task(data: list) -> dict:
    result = data[0]
    return result[_task_name(result)]

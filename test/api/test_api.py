"""
Test suite for running the API models
"""

import pytest
from fastapi.testclient import TestClient

from cnlpt.api.utils import EntityDocument


@pytest.fixture(autouse=True)
def disable_mps(monkeypatch):
    """Disable MPS for all tests"""
    monkeypatch.setattr("torch._C._mps_is_available", lambda: False)


class TestNegation:
    @pytest.fixture
    def test_client(self):
        from cnlpt.api.negation_rest import app

        with TestClient(app) as client:
            yield client

    def test_negation_startup(self, test_client):
        pass

    def test_negation_process(self, test_client: TestClient):
        from cnlpt.api.negation_rest import NegationResults

        doc = EntityDocument(
            doc_text="The patient has a sore knee and headache "
            "but denies nausea and has no anosmia.",
            entities=[[18, 27], [32, 40], [52, 58], [70, 77]],
        )
        response = test_client.post("/negation/process", content=doc.json())
        response.raise_for_status()
        assert response.json() == NegationResults.parse_obj(
            {"statuses": [-1, -1, 1, 1]}
        )


class TestTemporal:
    @pytest.fixture
    def test_client(self):
        from cnlpt.api.temporal_rest import app

        with TestClient(app) as client:
            yield client

    def test_temporal_startup(self, test_client: TestClient):
        pass

    def test_temporal_process_sentence(self, test_client: TestClient):
        from cnlpt.api.temporal_rest import (
            SentenceDocument,
            TemporalResults,
        )

        doc = SentenceDocument(
            sentence="The patient was diagnosed with adenocarcinoma "
            "March 3, 2010 and will be returning for "
            "chemotherapy next week."
        )
        response = test_client.post("/temporal/process_sentence", content=doc.json())
        response.raise_for_status()
        out = response.json()
        expected_out = TemporalResults.parse_obj(
            {
                "events": [
                    [
                        {"begin": 3, "dtr": "BEFORE", "end": 3},
                        {"begin": 5, "dtr": "BEFORE", "end": 5},
                        {"begin": 13, "dtr": "AFTER", "end": 13},
                        {"begin": 15, "dtr": "AFTER", "end": 15},
                    ]
                ],
                "relations": [
                    [
                        {
                            "arg1": "TIMEX-0",
                            "arg1_start": 6,
                            "arg2": "EVENT-0",
                            "arg2_start": 3,
                            "category": "CONTAINS",
                        },
                        {
                            "arg1": "TIMEX-1",
                            "arg1_start": 16,
                            "arg2": "EVENT-2",
                            "arg2_start": 13,
                            "category": "CONTAINS",
                        },
                    ]
                ],
                "timexes": [
                    [
                        {"begin": 6, "end": 9, "timeClass": "DATE"},
                        {"begin": 16, "end": 17, "timeClass": "DATE"},
                    ]
                ],
            }
        )
        assert out == expected_out

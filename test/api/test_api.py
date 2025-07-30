"""
Test suite for running the API models
"""

import pytest
from fastapi.testclient import TestClient

from cnlpt.rest.cnlp_rest import CnlpRestApp, InputDocument


class TestNegation:
    @pytest.fixture
    def test_client(self):
        with TestClient(
            CnlpRestApp("mlml-chip/negation_pubmedbert_sharpseed").fastapi()
        ) as client:
            yield client

    def test_negation_startup(self, test_client):
        pass

    def test_negation_process(self, test_client: TestClient):
        doc = InputDocument(
            text="The patient has a sore knee and headache "
            "but denies nausea and has no anosmia.",
            entity_spans=[(18, 27), (32, 40), (52, 58), (70, 77)],
        )
        response = test_client.post("/process", content=doc.json())
        response.raise_for_status()
        assert response.json() == [
            {
                "text": "The patient has a <e>sore knee</e> and headache but denies nausea and has no anosmia.",
                "Negation": {
                    "prediction": "-1",
                    "probs": {
                        "1": pytest.approx(0.0002379878715146333),
                        "-1": pytest.approx(0.9997619986534119),
                    },
                },
            },
            {
                "text": "The patient has a sore knee and <e>headache</e> but denies nausea and has no anosmia.",
                "Negation": {
                    "prediction": "-1",
                    "probs": {
                        "1": pytest.approx(0.0004393413255456835),
                        "-1": pytest.approx(0.9995606541633606),
                    },
                },
            },
            {
                "text": "The patient has a sore knee and headache but denies <e>nausea</e> and has no anosmia.",
                "Negation": {
                    "prediction": "1",
                    "probs": {
                        "1": pytest.approx(0.9921413660049438),
                        "-1": pytest.approx(0.007858583703637123),
                    },
                },
            },
            {
                "text": "The patient has a sore knee and headache but denies nausea and has no <e>anosmia</e>.",
                "Negation": {
                    "prediction": "1",
                    "probs": {
                        "1": pytest.approx(0.9928833246231079),
                        "-1": pytest.approx(0.0071166763082146645),
                    },
                },
            },
        ]


class TestTemporal:
    @pytest.fixture
    def test_client(self):
        with TestClient(CnlpRestApp("mlml-chip/thyme2_colon_e2e").fastapi()) as client:
            yield client

    def test_temporal_startup(self, test_client: TestClient):
        pass

    def test_temporal_process_sentence(self, test_client: TestClient):
        doc = InputDocument(
            text="The patient was diagnosed with adenocarcinoma "
            "March 3, 2010 and will be returning for "
            "chemotherapy next week."
        )
        response = test_client.post("/process", content=doc.json())
        response.raise_for_status()
        assert response.json() == [
            {
                "text": "The patient was diagnosed with adenocarcinoma March 3, 2010 and will be returning for chemotherapy next week.",
                "timex": {
                    "spans": [
                        {
                            "text": "March 3, 2010 ",
                            "tag": "DATE",
                            "start": 6,
                            "end": 8,
                            "valid": True,
                        },
                        {
                            "text": "next week.",
                            "tag": "DATE",
                            "start": 15,
                            "end": 16,
                            "valid": True,
                        },
                    ]
                },
                "event": {
                    "spans": [
                        {
                            "text": "diagnosed ",
                            "tag": "BEFORE",
                            "start": 3,
                            "end": 3,
                            "valid": True,
                        },
                        {
                            "text": "adenocarcinoma ",
                            "tag": "BEFORE",
                            "start": 5,
                            "end": 5,
                            "valid": True,
                        },
                        {
                            "text": "returning ",
                            "tag": "AFTER",
                            "start": 12,
                            "end": 12,
                            "valid": True,
                        },
                        {
                            "text": "chemotherapy ",
                            "tag": "AFTER",
                            "start": 14,
                            "end": 14,
                            "valid": True,
                        },
                    ]
                },
                "tlinkx": {
                    "relations": [
                        {
                            "arg1_wid": 6,
                            "arg1_text": "March",
                            "arg2_wid": 3,
                            "arg2_text": "diagnosed",
                            "label": "CONTAINS",
                        },
                        {
                            "arg1_wid": 15,
                            "arg1_text": "next",
                            "arg2_wid": 12,
                            "arg2_text": "returning",
                            "label": "CONTAINS",
                        },
                    ]
                },
            }
        ]

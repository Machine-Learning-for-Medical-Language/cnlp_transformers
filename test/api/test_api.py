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
            text="The patient has a sore knee and headache but denies nausea and has no anosmia.",
            entity_spans=[(18, 27), (32, 40), (52, 58), (70, 77)],
        )
        response = test_client.post("/process", content=doc.json())
        response.raise_for_status()
        response_json = response.json()
        assert [record["Negation"]["prediction"] for record in response_json] == [
            "-1",  # sore knee (not negated)
            "-1",  # headache (not negated)
            "1",  # nausea (negated)
            "1",  # anosmia (negated)
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
            text="The patient was diagnosed with adenocarcinoma March 3, 2010 and will be returning for chemotherapy next week."
        )
        response = test_client.post("/process", content=doc.json())
        response.raise_for_status()
        response_json = response.json()
        assert all(
            span in response_json[0]["timex"]["spans"]
            for span in [
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
        )

        assert all(
            span in response_json[0]["event"]["spans"]
            for span in [
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
        )

        assert all(
            span in response_json[0]["tlinkx"]["relations"]
            for span in [
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
        )

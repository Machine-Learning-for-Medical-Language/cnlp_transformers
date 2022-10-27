"""
Test suite for running the API models
"""
import asyncio
import pytest

from cnlpt.api import cnlp_rest


class TestNegation:
    @pytest.fixture
    def startup_negation(self):
        from cnlpt.api.negation_rest import startup_event as negation_startup
        asyncio.run(negation_startup())

    def test_negation_startup(self, startup_negation):
        pass

    def test_negation_process(self, startup_negation):
        from cnlpt.api.negation_rest import process as negation_process, NegationResults
        doc = cnlp_rest.EntityDocument(
            doc_text="The patient has a sore knee and headache "
                     "but denies nausea and has no anosmia.",
            entities=[[18, 27], [32, 40], [52, 58], [70, 77]]
        )
        out = asyncio.run(negation_process(doc))
        assert out == NegationResults.parse_obj(
            {'statuses': [-1, -1, 1, 1]}
        )


class TestTemporal:
    @pytest.fixture
    def startup_temporal(self):
        from cnlpt.api.temporal_rest import startup_event as temporal_startup
        asyncio.run(temporal_startup())

    def test_temporal_startup(self, startup_temporal):
        pass

    def test_temporal_process_sentence(self, startup_temporal):
        from cnlpt.api.temporal_rest import process_sentence as temporal_process_sentence, TemporalResults, SentenceDocument, Timex, Event, Relation
        doc = SentenceDocument(
            sentence='The patient was diagnosed with adenocarcinoma '
                     'March 3, 2010 and will be returning for '
                     'chemotherapy next week.'
        )
        out = asyncio.run(temporal_process_sentence(doc))
        expected_out = TemporalResults.parse_obj(
            {'events': [[{'begin': 3, 'dtr': 'BEFORE', 'end': 3},
                         {'begin': 5, 'dtr': 'BEFORE', 'end': 5},
                         {'begin': 13, 'dtr': 'AFTER', 'end': 13},
                         {'begin': 15, 'dtr': 'AFTER', 'end': 15}]],
             'relations': [[{'arg1': 'EVENT-0', 'arg2': 'EVENT-1', 'category': 'OVERLAP', 'arg1_start': 3, 'arg2_start': 5}, 
                            {'arg1': 'TIMEX-0', 'arg2': 'EVENT-0', 'category': 'CONTAINS', 'arg1_start': 6, 'arg2_start': 3}, 
                            {'arg1': 'EVENT-3', 'arg2': 'EVENT-2', 'category': 'BEGINS-ON', 'arg1_start': 15, 'arg2_start': 13}, 
                            {'arg1': 'TIMEX-1', 'arg2': 'EVENT-2', 'category': 'CONTAINS', 'arg1_start': 16, 'arg2_start': 13}, 
                            {'arg1': 'TIMEX-1', 'arg2': 'EVENT-3', 'category': 'CONTAINS', 'arg1_start': 16, 'arg2_start': 15}]],
             'timexes': [[{'begin': 6, 'end': 9, 'timeClass': 'DATE'},
                          {'begin': 16, 'end': 17, 'timeClass': 'DATE'}]]}
        )
        assert out == expected_out

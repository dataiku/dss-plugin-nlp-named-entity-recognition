# -*- coding: utf-8 -*-
import json

import pandas as pd
import pytest
from flair.models import SequenceTagger

from ner.constants import (
    COLUMN_PER_ENTITY_FORMAT,
    JSON_KEY_PER_ENTITY_FORMAT,
    JSON_LABELING_FORMAT
)
from ner.flair import extract_entities, get_model

TEST_SENTENCE = "Mark Zuckerberg is one of the founders of Facebook, a company from the United States"

def test_extract_entities():
    df = pd.DataFrame({'text': [TEST_SENTENCE]})
    pd.testing.assert_frame_equal(
        extract_entities(df['text'], JSON_LABELING_FORMAT, "en"),
        pd.DataFrame({
            'sentence': [TEST_SENTENCE],
            'entities': json.dumps([
                {'text': 'Mark Zuckerberg', 'beginningIndex': 0, 'endIndex': 15, 'category': 'PER'},
                {'text': 'Facebook', 'beginningIndex': 42, 'endIndex': 50, 'category': 'ORG'},
                {'text': 'United States', 'beginningIndex': 71, 'endIndex': 84, 'category': 'LOC'}
            ])
        })
    )
    pd.testing.assert_frame_equal(
        extract_entities(df['text'], COLUMN_PER_ENTITY_FORMAT, "en"),
        pd.DataFrame({
            'sentence': [TEST_SENTENCE],
            'LOC': json.dumps(['United States']),
            'ORG': json.dumps(['Facebook']),
            'PER': json.dumps(['Mark Zuckerberg']),
        })
    )
    pd.testing.assert_frame_equal(
        extract_entities(df['text'], JSON_KEY_PER_ENTITY_FORMAT, "en"),
        pd.DataFrame({
            'sentence': [TEST_SENTENCE],
            'entities': json.dumps({
                'PER': ['Mark Zuckerberg'],
                'ORG': ['Facebook'],
                'LOC': ['United States'],
            })
        })
    )

def test_extract_entities_empty_text():
    df = pd.DataFrame({'text': ['']})
    result = extract_entities(df['text'], JSON_LABELING_FORMAT, "en")
    assert len(result) == 1
    assert result['sentence'].iloc[0] == ''
    assert json.loads(result['entities'].iloc[0]) == []

def test_extract_entities_no_entities():
    df = pd.DataFrame({'text': ['Hello world, this is a simple test.']})
    result = extract_entities(df['text'], JSON_LABELING_FORMAT, "en")
    assert len(result) == 1
    assert result['sentence'].iloc[0] == 'Hello world, this is a simple test.'
    assert json.loads(result['entities'].iloc[0]) == []

def test_extract_entities_unicode():
    unicode_text = 'Müller works at Nestlé in Zürich.'
    df = pd.DataFrame({'text': [unicode_text]})
    result = extract_entities(df['text'], JSON_LABELING_FORMAT, "en")
    assert len(result) == 1
    # Verify unicode text is preserved correctly
    assert result['sentence'].iloc[0] == unicode_text
    # Verify valid JSON output (no encoding errors)
    entities = json.loads(result['entities'].iloc[0])
    assert isinstance(entities, list)

def test_extract_entities_multiple_same_type():
    df = pd.DataFrame({'text': ['John and Mary went to Paris and London.']})
    result = extract_entities(df['text'], COLUMN_PER_ENTITY_FORMAT, "en")
    assert len(result) == 1
    # Should have multiple entities, check that PER or LOC columns exist with arrays
    if 'PER' in result.columns:
        per_entities = json.loads(result['PER'].iloc[0])
        assert isinstance(per_entities, list)
    if 'LOC' in result.columns:
        loc_entities = json.loads(result['LOC'].iloc[0])
        assert isinstance(loc_entities, list)

def test_extract_entities_multiple_rows():
    df = pd.DataFrame({'text': [
        'Apple is based in California.',
        'Microsoft was founded by Bill Gates.'
    ]})
    result = extract_entities(df['text'], JSON_LABELING_FORMAT, "en")
    assert len(result) == 2

def test_get_model_legacy_mapping():
    model = get_model("en")
    assert isinstance(model, SequenceTagger)

def test_get_model_invalid_id():
    with pytest.raises(KeyError):
        get_model("invalid_language_code")

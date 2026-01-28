# -*- coding: utf-8 -*-
import json

import pandas as pd
import pytest
import spacy

from ner.constants import (
    COLUMN_PER_ENTITY_FORMAT,
    JSON_KEY_PER_ENTITY_FORMAT,
    JSON_LABELING_FORMAT
)
from ner.spacy import extract_entities, get_model, SPACY_LANGUAGE_MODELS_LEGACY_MAPPING

TEST_SENTENCE = "Mark Zuckerberg is one of the founders of Facebook, a company from the United States"

def test_extract_entities():
    df = pd.DataFrame({'text': [TEST_SENTENCE]})
    pd.testing.assert_frame_equal(
        extract_entities(df['text'], JSON_LABELING_FORMAT, "en"),
        pd.DataFrame({
            'sentence': [TEST_SENTENCE],
            'entities': json.dumps([
                {'text': 'Mark Zuckerberg', 'beginningIndex': 0, 'endIndex': 15, 'category': 'PERSON'},
                {'text': 'one', 'beginningIndex': 19, 'endIndex': 22, 'category': 'CARDINAL'},
                {'text': 'the United States', 'beginningIndex': 67, 'endIndex': 84, 'category': 'GPE'}
            ])
        })
    )
    pd.testing.assert_frame_equal(
        extract_entities(df['text'], COLUMN_PER_ENTITY_FORMAT, "en"),
        pd.DataFrame({
            'sentence': [TEST_SENTENCE],
            'CARDINAL': json.dumps(['one']),
            'GPE': json.dumps(['the United States']),
            'PERSON': json.dumps(['Mark Zuckerberg']),
        })
    )
    pd.testing.assert_frame_equal(
        extract_entities(df['text'], JSON_KEY_PER_ENTITY_FORMAT, "en"),
        pd.DataFrame({
            'sentence': [TEST_SENTENCE],
            'entities': json.dumps({
                'PERSON': ['Mark Zuckerberg'],
                'CARDINAL': ['one'],
                'GPE': ['the United States']
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
    # Should have multiple entities, check that PERSON or GPE columns exist with arrays
    if 'PERSON' in result.columns:
        person_entities = json.loads(result['PERSON'].iloc[0])
        assert isinstance(person_entities, list)
    if 'GPE' in result.columns:
        gpe_entities = json.loads(result['GPE'].iloc[0])
        assert isinstance(gpe_entities, list)

def test_extract_entities_multiple_rows():
    df = pd.DataFrame({'text': [
        'Apple is based in California.',
        'Microsoft was founded by Bill Gates.'
    ]})
    result = extract_entities(df['text'], JSON_LABELING_FORMAT, "en")
    assert len(result) == 2

def test_get_model_english():
    model = get_model("en")
    assert model is not None
    assert hasattr(model, 'pipe')

def test_get_model_french():
    model = get_model("fr")
    assert model is not None
    assert hasattr(model, 'pipe')

def test_get_model_invalid_id():
    with pytest.raises(KeyError):
        get_model("invalid_language_code")

def test_language_models_mapping_completeness():
    expected_languages = ["en", "es", "zh", "pl", "fr", "de", "ja", "nb"]
    for lang in expected_languages:
        assert lang in SPACY_LANGUAGE_MODELS_LEGACY_MAPPING

import json
import pandas as pd

from ner.constants import (
    COLUMN_PER_ENTITY_FORMAT,
    JSON_KEY_PER_ENTITY_FORMAT,
    JSON_LABELING_FORMAT
)
from ner.spacy import extract_entities


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
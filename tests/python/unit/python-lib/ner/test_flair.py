import json

import pandas as pd

from ner.constants import (
    COLUMN_PER_ENTITY_FORMAT,
    JSON_KEY_PER_ENTITY_FORMAT,
    JSON_LABELING_FORMAT
)
from ner.flair import extract_entities

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
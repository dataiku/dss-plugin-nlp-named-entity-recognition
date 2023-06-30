# -*- coding: utf-8 -*-
import json

from flair.data import Sentence
import pandas as pd

from .constants import (
    COLUMN_PER_ENTITY_FORMAT,
    JSON_KEY_PER_ENTITY_FORMAT,
    JSON_LABELING_FORMAT
)


def extract_entities(text_column, format, tagger):
    # Create Sentences
    sentences = [Sentence(text, use_tokenizer=True) for text in text_column.values]

    # Tag Sentences
    tagger.predict(sentences)

    # Extract entities
    extract = {
        COLUMN_PER_ENTITY_FORMAT: get_columns_per_entity_rows,
        JSON_KEY_PER_ENTITY_FORMAT: get_json_key_per_entity_rows,
        JSON_LABELING_FORMAT: get_json_labeling_rows
    }[format]
    rows = extract(sentences)

    entity_df = pd.DataFrame(rows)

    # Put 'sentence' column first
    cols = sorted(list(entity_df.columns))
    cols.insert(0, cols.pop(cols.index("sentence")))
    entity_df = entity_df[cols]
    return entity_df

def get_json_key_per_entity_rows(sentences):
    rows = []
    for sentence in sentences:
        entities = {}
        for span in sentence.get_spans('ner'):
            label = span.get_label("ner").value
            if label not in entities:
                entities[label] = []
            entities[label].append(span.text)
        rows.append({"sentence": sentence.to_plain_string(), "entities": json.dumps(entities)})
    return rows

def get_columns_per_entity_rows(sentences):
    rows = []
    for sentence in sentences:
        labels = {}
        for span in sentence.get_spans('ner'):
            label = span.get_label("ner").value
            if label not in labels:
                labels[label] = []
            labels[label].append(span.text)
        row = {"sentence": sentence.to_plain_string()}
        for k, v in labels.items():
            row[k] = json.dumps(v)
        rows.append(row)
    return rows

def get_json_labeling_rows(sentences):
    rows = []
    for sentence in sentences:
        entities = []
        for span in sentence.get_spans('ner'):
            entities.append({
                "text": span.text,
                "beginningIndex": span.start_position,
                "endIndex": span.end_position,
                "category": span.get_label("ner").value
            })
        rows.append({"sentence": sentence.to_plain_string(), "entities": json.dumps(entities)})
    return rows

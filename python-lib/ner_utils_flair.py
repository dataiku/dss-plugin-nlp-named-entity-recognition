# -*- coding: utf-8 -*-
import json

from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd


FLAIR_LANGUAGE_MODELS_LEGACY_MAPPING = {
    "en": "flair/ner-english-fast@3d3d35790f78a00ef319939b9004209d1d05f788",
}


def get_model(model_id: str):
    return SequenceTagger.load(FLAIR_LANGUAGE_MODELS_LEGACY_MAPPING.get(model_id, None))

def extract_entities(text_column, format, model_id):
    # Create Sentences
    tagger = get_model(model_id)
    sentences = [Sentence(text, use_tokenizer=True) for text in text_column.values]

    # Tag Sentences
    tagger.predict(sentences)

    # Extract entities
    rows = {
        "standard": get_standard_rows,
        "labeling": get_labeling_rows
    }.get(format, get_default_rows)(sentences)

    entity_df = pd.DataFrame(rows)

    # Put 'sentence' column first
    cols = sorted(list(entity_df.columns))
    cols.insert(0, cols.pop(cols.index("sentence")))
    entity_df = entity_df[cols]
    return entity_df

def get_standard_rows(sentences):
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

def get_default_rows(sentences):
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

def get_labeling_rows(sentences):
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

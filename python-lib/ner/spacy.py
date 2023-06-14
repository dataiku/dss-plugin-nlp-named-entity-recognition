# -*- coding: utf-8 -*-
import json

import pandas as pd
import spacy

from .constants import (
    COLUMN_PER_ENTITY_FORMAT,
    JSON_KEY_PER_ENTITY_FORMAT,
    JSON_LABELING_FORMAT
)


SPACY_LANGUAGE_MODELS_LEGACY_MAPPING = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "zh": "zh_core_web_sm",
    "pl": "nb_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "ja": "ja_core_news_sm",
    "nb": "nb_core_news_sm",
}

def get_model(model_id: str):
    return spacy.load(SPACY_LANGUAGE_MODELS_LEGACY_MAPPING.get(model_id, None), exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    
def extract_entities(text_column, format, model_id: str):
    # Tag sentences
    nlp = get_model(model_id=model_id)
    docs = nlp.pipe(text_column.values, n_process=-1, batch_size=100)
    # Extract entities
    extract = {
        COLUMN_PER_ENTITY_FORMAT: get_columns_per_entity_rows,
        JSON_KEY_PER_ENTITY_FORMAT: get_json_key_per_entity_rows,
        JSON_LABELING_FORMAT: get_json_labeling_rows
    }[format]
    rows = extract(docs)

    entity_df = pd.DataFrame(rows)

    # Put 'sentence' column first
    cols = sorted(list(entity_df.columns))
    cols.insert(0, cols.pop(cols.index("sentence")))
    entity_df = entity_df[cols]
    return entity_df

def get_json_key_per_entity_rows(docs):
    rows = []
    for doc in docs:
        entities = {}
        for span in doc.ents:
            if span.label_ not in entities:
                entities[span.label_] = []
            entities[span.label_].append(span.text)
        rows.append({"sentence": doc.text, "entities": json.dumps(entities)})
    return rows

def get_columns_per_entity_rows(docs):
    rows = []
    for doc in docs:
        labels = {}
        for span in doc.ents:
            if span.label_ not in labels:
                labels[span.label_] = []
            labels[span.label_].append(span.text)
        row = {"sentence": doc.text}
        for k, v in labels.items():
            row[k] = json.dumps(v)
        rows.append(row)
    return rows

def get_json_labeling_rows(docs):
    rows = []
    for doc in docs:
        entities = []
        for span in doc.ents:
            entities.append({
                "text": span.text,
                "beginningIndex": span.start_char,
                "endIndex": span.end_char,
                "category": span.label_
            })
        rows.append({"sentence": doc.text, "entities": json.dumps(entities)})
    return rows

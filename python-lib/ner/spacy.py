# -*- coding: utf-8 -*-
import json

import pandas as pd
import spacy

from .constants import (
    COLUMN_PER_ENTITY_FORMAT,
    JSON_KEY_PER_ENTITY_FORMAT,
    JSON_LABELING_FORMAT
)


SPACY_LANGUAGE_MODELS = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "zh": "zh_core_web_sm",
    "pl": "nb_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "ja": "ja_core_news_sm",
    "nb": "nb_core_news_sm",
}

def get_spacy_model(language: str):
    language_model = SPACY_LANGUAGE_MODELS.get(language, None)
    if language_model is None:
        raise ValueError(f"The language {language} is not available. \
                        You can add the language & corresponding model name by editing the code.")
    try:
        nlp = spacy.load(language_model, exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    except OSError:
        # Raising ValueError instead of OSError so it shows up at the top of the log
        raise ValueError(f"Could not find spaCy model for the language {language}. \
                        Maybe you need to edit the requirements.txt file to enable it.")
    return nlp

def extract_entities(text_column, format, language: str):
    # Tag sentences
    nlp = get_spacy_model(language=language)
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

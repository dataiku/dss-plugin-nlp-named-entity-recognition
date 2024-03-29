# -*- coding: utf-8 -*-
import json

from flair.data import Sentence
from flair.models import SequenceTagger
import pandas as pd

from .constants import (
    COLUMN_PER_ENTITY_FORMAT,
    JSON_KEY_PER_ENTITY_FORMAT,
    JSON_LABELING_FORMAT
)

# backward compatibility
model_provider = None

try:
    from dataiku.core.model_provider import get_model_provider
    model_provider = get_model_provider()
except ImportError:
    pass


FLAIR_LANGUAGE_MODELS_LEGACY_MAPPING = {
    "en": "flair/ner-english-fast@3d3d35790f78a00ef319939b9004209d1d05f788",
}

FLAIR_MODEL_PROVIDER_MAPPING = {
    "ner_english_ontonotes_fast": "flair/ner-english-ontonotes-fast@38a8eb6a720791da55e15962c36a37dd8d8270b2",
}


def get_model(model_id: str):
    language_model = FLAIR_LANGUAGE_MODELS_LEGACY_MAPPING.get(model_id, None)
    if language_model is not None:
        # those models are downloaded on resources init
        model_path = language_model
    elif model_provider is not None:
        # dl model with provider at hugging face proper location, return pytorch bin path
        model_path = "%s/pytorch_model.bin" % model_provider.get_or_download_model(FLAIR_MODEL_PROVIDER_MAPPING[model_id])
    return SequenceTagger.load(model_path)

def extract_entities(text_column, format, model_id):
    # Create Sentences
    tagger = get_model(model_id)
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
